from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from baselines import resolve_baseline_token_count, run_gaussian_baseline
from clustering import (
    HiddenStateRecorder,
    LayerHidden,
    adjusted_rand_index,
    analyze_hidden_state_umap,
    require_hdbscan,
)
from model_loading import load_dsv4
from plotting import sample_summary, save_hidden_state_umap_plot
from prompts import (
    attach_snapshot_tokens,
    encode_prompt,
    jsonify,
    limit_prompt_tokens,
    load_prompt_jobs,
    normalize_rows,
)


@torch.inference_mode()
def capture_prompt_hidden_states(
    model: torch.nn.Module,
    input_ids: Sequence[int],
    max_tokens: int,
    num_observation_layers: int = 12,
) -> Dict[int, LayerHidden]:
    tokens = torch.tensor([list(input_ids)], dtype=torch.long, device="cuda")
    recorder = HiddenStateRecorder(
        model,
        max_tokens=max_tokens,
        num_observation_layers=num_observation_layers,
    )
    recorder.install()
    try:
        model.forward(tokens, 0)
        torch.cuda.synchronize()
        captures = dict(recorder.layers)
    finally:
        recorder.uninstall()
    return captures

def command_self_test(args: argparse.Namespace) -> None:
    require_hdbscan()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    true_labels = torch.arange(3, dtype=torch.long).repeat_interleave(40)
    centers = normalize_rows(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-0.5, 0.8660254, 0.0, 0.0],
                [-0.5, -0.8660254, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )
    early = normalize_rows(
        torch.randn(120, 4, generator=generator, dtype=torch.float32)
    )
    middle = normalize_rows(
        0.9 * centers[true_labels]
        + 0.55
        * torch.randn(120, 4, generator=generator, dtype=torch.float32)
    )
    final = normalize_rows(
        3.0 * centers[true_labels]
        + 0.14
        * torch.randn(120, 4, generator=generator, dtype=torch.float32)
    )
    captures = {
        0: LayerHidden(layer_id=0, hidden=early),
        21: LayerHidden(layer_id=21, hidden=middle),
        42: LayerHidden(layer_id=42, hidden=final),
    }
    result = analyze_hidden_state_umap(
        captures,
        cluster_umap_components=4,
        cluster_fit_tokens_per_layer=120,
        spherical_cluster_space="umap",
        umap_n_neighbors=10,
        umap_min_dist=0.1,
        umap_metric="cosine",
        hdbscan_min_cluster_size=20,
        hdbscan_min_samples=5,
        hdbscan_cluster_selection_method="eom",
        hdbscan_allow_single_cluster=True,
        assign_all_tokens=True,
        plot_tokens=120,
        seed=int(args.seed),
    )
    final_clustering = result.get("final_layer_clustering", {})
    inferred_labels = torch.tensor(
        final_clustering.get("token_labels", []),
        dtype=torch.long,
    )
    trace = result.get("layer_trace") or []
    snapshots = result.get("snapshots_sphere") or []
    fixed_snapshot_labels = all(
        snapshot.get("final_cluster_label")
        == inferred_labels[
            torch.tensor(snapshot.get("token_index", []), dtype=torch.long)
        ].tolist()
        for snapshot in snapshots
    )
    checks = {
        "analysis_available": bool(result.get("available")),
        "hdbscan_infers_k3": int(final_clustering.get("n_clusters", 0)) == 3,
        "every_token_has_a_cluster": (
            inferred_labels.numel() == true_labels.numel()
            and bool((inferred_labels >= 0).all())
            and int(final_clustering.get("noise_count", -1)) == 0
        ),
        "final_partition_matches_known_groups": (
            inferred_labels.numel() == true_labels.numel()
            and adjusted_rand_index(inferred_labels, true_labels) > 0.95
        ),
        "labels_are_reused_without_reclustering": bool(trace)
        and all(row.get("labels_reused_without_reclustering") for row in trace),
        "final_cohorts_separate_more_than_early": bool(trace)
        and float(trace[-1]["fixed_label_cosine_silhouette"])
        > float(trace[0]["fixed_label_cosine_silhouette"]),
        "snapshot_layers_are_0_21_42": [
            int(snapshot["layer"]) for snapshot in snapshots
        ]
        == [0, 21, 42],
        "snapshot_colors_use_final_labels": fixed_snapshot_labels,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "tracking_summary": result.get("tracking_summary"),
    }
    print(json.dumps(jsonify(report), ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("Final-layer HDBSCAN tracking self-test failed")

def command_evaluate(args: argparse.Namespace) -> None:
    _, hdbscan_version = require_hdbscan()
    analysis_config = {
        "analysis": "final_layer_anchored_hdbscan_tracking",
        "hdbscan_version": hdbscan_version,
        "cluster_space": args.spherical_cluster_space,
        "umap_components": int(args.cluster_umap_components),
        "umap_fit_tokens_per_layer": int(args.cluster_fit_tokens_per_layer),
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "umap_metric": str(args.umap_metric),
        "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
        "hdbscan_min_cluster_fraction": float(args.hdbscan_min_cluster_fraction),
        "hdbscan_min_samples": int(args.hdbscan_min_samples),
        "hdbscan_min_samples_fraction": float(args.hdbscan_min_samples_fraction),
        "hdbscan_cluster_selection_method": args.hdbscan_cluster_selection_method,
        "hdbscan_cluster_selection_epsilon": float(args.hdbscan_cluster_selection_epsilon),
        "hdbscan_alpha": float(args.hdbscan_alpha),
        "hdbscan_allow_single_cluster": bool(args.hdbscan_allow_single_cluster),
        "assign_all_tokens": bool(args.assign_all_tokens),
        "num_observation_layers": int(args.num_observation_layers),
        "plot_tokens": int(args.plot_tokens),
        "sample_seed": int(args.sample_seed),
        "max_tokens": int(args.max_tokens),
        "max_seq_len": int(args.max_seq_len),
        "long_prompt_policy": args.long_prompt_policy,
        "truncation_head_fraction": float(args.truncation_head_fraction),
    }
    jobs = load_prompt_jobs(args)
    model, tokenizer, rank, world_size = load_dsv4(
        args.ckpt_path,
        args.config,
        max_sequence_length=args.max_seq_len,
        tokenizer_path=args.tokenizer_path,
        tilelang_backend=args.tilelang_backend,
    )
    is_main_process = rank == 0
    output_directory = Path(args.out_dir)
    if is_main_process:
        output_directory.mkdir(parents=True, exist_ok=True)
        group_counts = {
            group: sum(job.group == group for job in jobs)
            for group in dict.fromkeys(job.group for job in jobs)
        }
        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": "final_layer_anchored_hdbscan_tracking",
            "analysis_config": analysis_config,
            "sample_seed": int(args.sample_seed),
            "n_samples": len(jobs),
            "samples_by_group": group_counts,
            "samples": [
                {
                    "sample_index": index,
                    "tag": job.tag,
                    "group": job.group,
                    "text": job.text,
                    "n_characters": len(job.text),
                    "benchmark": job.metadata,
                }
                for index, job in enumerate(jobs)
            ],
        }
        manifest_path = output_directory / "sample_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {manifest_path}")

    sample_summaries: List[Dict[str, Any]] = []
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        dist.barrier()

    capture_limit = min(int(args.max_tokens), int(args.max_seq_len))
    baseline_length_stats = resolve_baseline_token_count(
        jobs,
        tokenizer,
        max_tokens=capture_limit,
        long_prompt_policy=args.long_prompt_policy,
        truncation_head_fraction=args.truncation_head_fraction,
    )
    analysis_config["gaussian_baseline"] = {
        "enabled": True,
        "n_tokens": int(baseline_length_stats["n_tokens"]),
        "length_stats": baseline_length_stats,
    }
    baseline_summary = run_gaussian_baseline(
        model=model,
        args=args,
        analysis_config=analysis_config,
        output_directory=output_directory,
        n_tokens=int(baseline_length_stats["n_tokens"]),
        baseline_length_stats=baseline_length_stats,
        is_main_process=is_main_process,
    )
    gaussian_embedding_cosine_silhouette: Optional[float] = None
    if is_main_process and baseline_summary is not None:
        sample_summaries.append(baseline_summary)
        raw_gaussian_silhouette = baseline_summary.get(
            "initialized_gaussian_cosine_silhouette"
        )
        if raw_gaussian_silhouette is not None:
            gaussian_embedding_cosine_silhouette = float(raw_gaussian_silhouette)
    if distributed:
        dist.barrier()

    for sample_index, job in enumerate(jobs):
        tag, group, text = job.tag, job.group, job.text
        json_path = output_directory / f"{tag}_hidden_umap.json"
        resume_record: Optional[Dict[str, Any]] = None
        resume_sample = False
        if is_main_process and args.resume and json_path.is_file():
            try:
                resume_record = json.loads(json_path.read_text(encoding="utf-8"))
                resume_sample = (
                    resume_record.get("hidden_state_umap", {}).get("analysis")
                    == "final_layer_anchored_hdbscan_tracking"
                    and resume_record.get("analysis_config") == analysis_config
                )
            except (OSError, json.JSONDecodeError):
                resume_sample = False
            if not resume_sample:
                print(
                    f"[sample {sample_index + 1}/{len(jobs)}] {tag}: existing "
                    "result is not compatible with HDBSCAN tracking; recomputing",
                    flush=True,
                )
        if distributed:
            resume_object = [resume_sample]
            dist.broadcast_object_list(resume_object, src=0)
            resume_sample = bool(resume_object[0])
        if resume_sample:
            if is_main_process:
                assert resume_record is not None
                record = resume_record
                sample_summaries.append(sample_summary(record))
                print(f"[sample {sample_index + 1}/{len(jobs)}] resume {tag}", flush=True)
            continue

        if is_main_process:
            print(
                f"\n=== [{sample_index + 1}/{len(jobs)}] {tag} ({group}) ===",
                flush=True,
            )
        encoded_ids = encode_prompt(tokenizer, text)
        capture_limit = min(int(args.max_tokens), int(args.max_seq_len))
        input_ids, tokenization = limit_prompt_tokens(
            encoded_ids,
            max_tokens=capture_limit,
            policy=args.long_prompt_policy,
            head_fraction=args.truncation_head_fraction,
        )
        if len(input_ids) < 4:
            raise ValueError(f"Prompt {tag} produced fewer than four tokens")
        if is_main_process and tokenization["truncated"]:
            print(
                f"warning: {tag} was truncated from "
                f"{tokenization['original_n_tokens']} to {tokenization['n_tokens']} "
                "tokens; this is an adapted rather than full-context benchmark item",
                flush=True,
            )
        start_time = time.time()
        captures = capture_prompt_hidden_states(
            model,
            input_ids,
            max_tokens=capture_limit,
            num_observation_layers=args.num_observation_layers,
        )

        if is_main_process:

            with torch.device("cpu"):
                result = analyze_hidden_state_umap(
                    captures,
                    cluster_umap_components=args.cluster_umap_components,
                    cluster_fit_tokens_per_layer=args.cluster_fit_tokens_per_layer,
                    spherical_cluster_space=args.spherical_cluster_space,
                    umap_n_neighbors=args.umap_n_neighbors,
                    umap_min_dist=args.umap_min_dist,
                    umap_metric=args.umap_metric,
                    hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
                    hdbscan_min_cluster_fraction=args.hdbscan_min_cluster_fraction,
                    hdbscan_min_samples=args.hdbscan_min_samples,
                    hdbscan_min_samples_fraction=args.hdbscan_min_samples_fraction,
                    hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
                    hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
                    hdbscan_alpha=args.hdbscan_alpha,
                    hdbscan_allow_single_cluster=args.hdbscan_allow_single_cluster,
                    assign_all_tokens=args.assign_all_tokens,
                    plot_tokens=args.plot_tokens,
                    seed=int(args.sample_seed) + sample_index * 1000003,
                )
            attach_snapshot_tokens(result, input_ids, tokenizer)
            record = {
                "meta": {
                    "tag": tag,
                    "group": group,
                    "sample_index": sample_index,
                    "n_tokens": len(input_ids),
                    "wall_time_s": time.time() - start_time,
                    "prompt_preview": text[:240],
                },
                "benchmark": dict(job.metadata),
                "analysis_config": analysis_config,
                "tokenization": tokenization,
                "hidden_state_umap": result,
            }
            plot_path = output_directory / f"{tag}_hidden_umap.png"
            record["hidden_state_umap"]["plot"] = save_hidden_state_umap_plot(
                result,
                plot_path,
                gaussian_embedding_cosine_silhouette=gaussian_embedding_cosine_silhouette,
            )
            json_path.write_text(
                json.dumps(jsonify(record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary_record = sample_summary(record)
            sample_summaries.append(summary_record)
            print(json.dumps(jsonify(summary_record), ensure_ascii=False, indent=2))
            print(f"wrote {json_path}")
            plot_result = record["hidden_state_umap"]["plot"]
            if plot_result.get("written"):
                print(f"wrote {plot_path}")
            else:
                print("Riemann UMAP plot skipped: " + str(plot_result.get("reason")))

        del captures
        gc.collect()
        torch.cuda.empty_cache()
        if distributed:
            dist.barrier()

    if is_main_process:
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": "final_layer_anchored_hdbscan_tracking",
            "analysis_config": analysis_config,
            "n_samples": len(sample_summaries),
            "n_truncated_samples": sum(
                bool(sample.get("truncated")) for sample in sample_summaries
            ),
            "samples": sample_summaries,
            "gaussian_baseline": analysis_config.get("gaussian_baseline"),
            "benchmark_scope": {
                "task": "HellaSwag commonsense prompts as hidden-state probes",
                "model_output_scoring": False,
            },
            "decision_rule": {
                "partition": "Final-layer HDBSCAN; earlier layers reuse labels",
                "metrics": "Fixed-label cosine silhouette vs depth",
                "visualization": "Dedicated shared Riemann UMAP sphere (haversine output)",
            },
        }
        summary_path = output_directory / "hidden_umap_summary.json"
        summary_path.write_text(
            json.dumps(jsonify(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {summary_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
