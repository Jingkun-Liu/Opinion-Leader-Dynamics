from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from clustering import (
    fit_cluster_umap,
    fit_final_layer_hdbscan,
    fixed_label_directional_metrics,
    serializable_partition,
    transform_cluster_umap,
)
from plotting import sample_summary
from prompts import (
    PromptJob,
    encode_prompt,
    jsonify,
    limit_prompt_tokens,
    normalize_rows,
)


@torch.no_grad()
def estimate_embedding_scale(
    model: torch.nn.Module,
    max_rows: int = 8192,
    seed: int = 0,
) -> Dict[str, Any]:
    if not hasattr(model, "embed") or not hasattr(model.embed, "weight"):
        raise RuntimeError("Model does not expose embed.weight for scale estimation")
    weight = model.embed.weight.detach()
    if weight.ndim != 2 or min(weight.shape) < 1:
        raise RuntimeError(f"Unexpected embedding weight shape: {tuple(weight.shape)}")
    n_rows, hidden_dim = int(weight.shape[0]), int(weight.shape[1])
    sample_count = min(n_rows, max(1, int(max_rows)))
    if sample_count < n_rows:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        index = torch.randperm(
            n_rows, generator=generator, device="cpu"
        )[:sample_count]
        sample = weight[index.to(device=weight.device)].to(dtype=torch.float32)
    else:
        sample = weight.to(dtype=torch.float32)
    sigma = float(sample.std(unbiased=False).clamp_min(1e-8))
    sigma_device = weight.device if weight.is_cuda else torch.device(
        "cuda", torch.cuda.current_device()
    )
    sigma_tensor = torch.tensor([sigma], device=sigma_device, dtype=torch.float32)
    import torch.distributed as dist

    synced = False
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sigma_tensor, op=dist.ReduceOp.SUM)
        sigma_tensor /= float(dist.get_world_size())
        sigma = float(sigma_tensor.item())
        synced = True
    return {
        "sigma": sigma,
        "method": "embedding_table_elementwise_std",
        "n_rows_available_on_rank": n_rows,
        "n_rows_sampled_on_rank": sample_count,
        "hidden_dim": hidden_dim,
        "distributed_sigma_allreduce_mean": synced,
    }

@torch.inference_mode()
def capture_gaussian_baseline_hidden_states(
    model: torch.nn.Module,
    n_tokens: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    n_tokens = int(n_tokens)
    if n_tokens < 4:
        raise ValueError("Gaussian baseline requires at least four tokens")
    if not hasattr(model, "embed"):
        raise RuntimeError("Model does not expose .embed for Gaussian baseline injection")
    hidden_dim = int(getattr(model.embed, "dim", 0) or getattr(getattr(model, "args", None), "dim", 0))
    if hidden_dim <= 0:
        raise RuntimeError("Unable to resolve model embedding dimension")

    scale_info = {
        "sigma": 1.0,
        "method": "standard_normal",
        "hidden_dim": hidden_dim,
        "scaled": False,
    }
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    gaussian = torch.randn(
        n_tokens,
        hidden_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return gaussian.detach().to(device="cpu", dtype=torch.float32), scale_info

@torch.no_grad()
def analyze_initialized_gaussian_embedding(
    embeddings: torch.Tensor,
    cluster_umap_components: int = 128,
    cluster_fit_tokens_per_layer: int = 256,
    spherical_cluster_space: str = "umap",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    hdbscan_min_cluster_size: int = 0,
    hdbscan_min_cluster_fraction: float = 0.05,
    hdbscan_min_samples: int = 0,
    hdbscan_min_samples_fraction: float = 0.01,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_alpha: float = 1.0,
    hdbscan_allow_single_cluster: bool = False,
    assign_all_tokens: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    hidden = embeddings.detach().to(device="cpu", dtype=torch.float32)
    if hidden.ndim != 2 or hidden.shape[0] < 4:
        return {
            "available": False,
            "reason": "Initialized Gaussian embedding requires at least four tokens",
        }
    cluster_umap_fit: Optional[Dict[str, Any]] = None
    if spherical_cluster_space == "hidden":
        coordinates = normalize_rows(hidden)
        clustering_space_name = "original_row_l2_normalized_gaussian_embedding"
    elif spherical_cluster_space == "umap":
        cluster_components = max(2, int(cluster_umap_components))
        try:
            reducer, cluster_umap_fit = fit_cluster_umap(
                [hidden],
                n_components=cluster_components,
                fit_tokens_per_layer=cluster_fit_tokens_per_layer,
                seed=int(seed),
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
            )
        except (RuntimeError, ValueError) as exc:
            return {
                "available": False,
                "reason": f"Gaussian clustering UMAP failed: {exc}",
            }
        coordinates = normalize_rows(transform_cluster_umap(reducer, hidden))
        clustering_space_name = "row_l2_normalized_shared_umap_coordinates"
    else:
        raise ValueError("spherical_cluster_space must be 'umap' or 'hidden'")
    partition = fit_final_layer_hdbscan(
        coordinates,
        min_cluster_size=hdbscan_min_cluster_size,
        min_cluster_fraction=hdbscan_min_cluster_fraction,
        min_samples=hdbscan_min_samples,
        min_samples_fraction=hdbscan_min_samples_fraction,
        cluster_selection_method=hdbscan_cluster_selection_method,
        cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        alpha=hdbscan_alpha,
        allow_single_cluster=hdbscan_allow_single_cluster,
        assign_all_tokens=assign_all_tokens,
    )
    metrics = fixed_label_directional_metrics(coordinates, partition["labels"])
    cosine_silhouette = metrics.get("fixed_label_cosine_silhouette")
    if cosine_silhouette is None:
        cosine_silhouette = 0.0
    n_clusters = int(partition["n_clusters"])
    return {
        "available": True,
        "analysis": "initialized_gaussian_embedding_cosine_silhouette",
        "interpretation_scope": "hdbscan_on_initialized_gaussian_embeddings_without_llm_evolution",
        "n_layers": 1,
        "n_aligned_tokens": int(hidden.shape[0]),
        "captured_layers": [-1],
        "embedding_baseline_available": True,
        "clustering_space": clustering_space_name,
        "cluster_space_mode": spherical_cluster_space,
        "cluster_algorithm": "hdbscan",
        "distance_metric": "precomputed_cosine_distance",
        "cluster_umap_fit": cluster_umap_fit,
        "visualization_method": "sample_curve_horizontal_reference",
        "initialized_gaussian_cosine_silhouette": float(cosine_silhouette),
        "tracking_summary": {
            "first_fixed_label_cosine_silhouette": float(cosine_silhouette),
            "middle_fixed_label_cosine_silhouette": float(cosine_silhouette),
            "final_fixed_label_cosine_silhouette": float(cosine_silhouette),
            "final_minus_first_cosine_silhouette": 0.0,
            "inferred_final_n_clusters": n_clusters,
            "final_noise_fraction": float(partition["noise_fraction"]),
        },
        "final_layer_clustering": {
            "reference_layer": -1,
            "reference_stage": "embedding",
            "n_clusters": n_clusters,
            "status": (
                "multi_cluster"
                if n_clusters > 1
                else "single_cluster"
                if n_clusters == 1
                else "no_density_cluster"
            ),
            "is_density_supported_multicluster": bool(n_clusters > 1),
            "noise_label": None if bool(assign_all_tokens) else -1,
            "token_labels": partition["labels"].tolist(),
            **serializable_partition(partition),
            "fixed_label_cosine_silhouette": float(cosine_silhouette),
        },
        "warnings": [
            "Gaussian baseline uses initialized embeddings only; no LLM evolution or sphere plot.",
            (
                f"Cosine silhouette is computed in the same {spherical_cluster_space} "
                "clustering space as samples."
            ),
            (
                "Noise reassigned to nearest cluster."
                if assign_all_tokens
                else "Label -1 retained as noise and excluded from cosine silhouette."
            ),
        ],
    }

def attach_gaussian_snapshot_tokens(result: Dict[str, Any], n_tokens: int) -> None:
    n_tokens = int(n_tokens)
    clustering = result.get("final_layer_clustering", {})
    final_labels = [int(x) for x in clustering.get("token_labels", [])]
    probs = list(clustering.get("membership_probabilities", []))
    outliers = list(clustering.get("outlier_scores", []))
    result["tokens"] = [
        {
            "token_index": i,
            "token_id": -1,
            "token_text": f"<gauss:{i}>",
            "final_cluster_label": final_labels[i] if i < len(final_labels) else None,
            "final_membership_probability": probs[i] if i < len(probs) else None,
            "final_outlier_score": outliers[i] if i < len(outliers) else None,
        }
        for i in range(n_tokens)
    ]
    for snapshot in result.get("snapshots_sphere", []):
        indices = [int(i) for i in snapshot.get("token_index", [])]
        snapshot["token_id"] = [-1 for _ in indices]
        snapshot["token_text"] = [f"<gauss:{i}>" for i in indices]

def resolve_baseline_token_count(
    jobs: Sequence[PromptJob],
    tokenizer,
    max_tokens: int,
    long_prompt_policy: str,
    truncation_head_fraction: float,
) -> Dict[str, Any]:
    capture_limit = int(max_tokens)
    lengths: List[int] = []
    for job in jobs:
        encoded = encode_prompt(tokenizer, job.text)
        limited, _ = limit_prompt_tokens(
            encoded,
            max_tokens=capture_limit,
            policy=long_prompt_policy,
            head_fraction=truncation_head_fraction,
        )
        if len(limited) >= 4:
            lengths.append(len(limited))
    if not lengths:
        n_tokens = max(4, min(capture_limit, 512))
        return {
            "n_tokens": n_tokens,
            "n_jobs_measured": 0,
            "median_tokens": n_tokens,
            "min_tokens": n_tokens,
            "max_tokens": n_tokens,
            "selection": "fallback_512_or_limit",
        }
    median = int(np.median(np.asarray(lengths, dtype=np.int64)))
    n_tokens = max(4, min(capture_limit, median))
    return {
        "n_tokens": n_tokens,
        "n_jobs_measured": len(lengths),
        "median_tokens": median,
        "min_tokens": int(min(lengths)),
        "max_tokens": int(max(lengths)),
        "selection": "median_eval_set_token_length",
    }

def run_gaussian_baseline(
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    analysis_config: Dict[str, Any],
    output_directory: Path,
    n_tokens: int,
    baseline_length_stats: Dict[str, Any],
    is_main_process: bool,
) -> Optional[Dict[str, Any]]:
    tag = "gaussian_baseline"
    json_path = output_directory / f"{tag}_hidden_umap.json"
    start_time = time.time()
    if is_main_process:
        print(
            f"\n=== [baseline] N(0, I) Gaussian embeddings "
            f"(n_tokens={n_tokens}, dim=model.embed; no LLM evolution) ===",
            flush=True,
        )
    embeddings, scale_info = capture_gaussian_baseline_hidden_states(
        model,
        n_tokens=n_tokens,
        seed=int(args.sample_seed) + 17,
    )
    if is_main_process:
        print(
            f"[baseline] no embedding-scale matching "
            f"(method={scale_info['method']})",
            flush=True,
        )
    summary = None
    if is_main_process:
        with torch.device("cpu"):
            result = analyze_initialized_gaussian_embedding(
                embeddings,
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
                seed=int(args.sample_seed) + 17,
            )
        cosine_silhouette = result.get("initialized_gaussian_cosine_silhouette")
        umap_fit = result.get("cluster_umap_fit") or {}
        gaussian_config = {
            **dict(analysis_config.get("gaussian_baseline") or {}),
            "scale": scale_info,
            "initialization": "iid_gaussian_embeddings",
            "initialized_cosine_silhouette": cosine_silhouette,
            "clustering_space": result.get("clustering_space"),
            "cluster_space_mode": result.get("cluster_space_mode"),
            "cluster_umap_n_components": umap_fit.get("n_components"),
            "visualization": "horizontal_reference_on_sample_silhouette_curves",
            "skip_evolution_plot": True,
        }
        analysis_config["gaussian_baseline"] = gaussian_config
        baseline_analysis_config = {
            **analysis_config,
            "gaussian_baseline": gaussian_config,
        }
        record = {
            "meta": {
                "tag": tag,
                "group": "baseline",
                "sample_index": -1,
                "n_tokens": n_tokens,
                "wall_time_s": time.time() - start_time,
                "prompt_preview": (
                    "N(0, I) embeddings clustered at initialization (no LLM forward, "
                    "no embedding-scale matching)"
                ),
            },
            "benchmark": {
                "dataset": "gaussian_baseline",
                "baseline": True,
                "initialization": "iid_gaussian_embeddings",
                "length_stats": baseline_length_stats,
                "scale": scale_info,
            },
            "analysis_config": baseline_analysis_config,
            "tokenization": {
                "original_n_tokens": n_tokens,
                "n_tokens": n_tokens,
                "truncated": False,
                "truncation_policy": "none",
                "synthetic": True,
            },
            "hidden_state_umap": result,
        }
        json_path.write_text(
            json.dumps(jsonify(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary = sample_summary(record)
        summary["initialized_gaussian_cosine_silhouette"] = cosine_silhouette
        print(json.dumps(jsonify(summary), ensure_ascii=False, indent=2))
        print(f"wrote {json_path}")
        print(
            "Gaussian evolution/sphere plot skipped; "
            f"initialized cosine silhouette={cosine_silhouette} "
            "will be drawn on sample curves",
            flush=True,
        )
    del embeddings
    gc.collect()
    torch.cuda.empty_cache()
    return summary
