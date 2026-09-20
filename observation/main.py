from __future__ import annotations

import argparse

from prompts import HELLASWAG_DEFAULT_PATH
from runner import command_evaluate, command_self_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 HellaSwag hidden-state HDBSCAN tracking + Riemann UMAP viz"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    self_test = subparsers.add_parser(
        "self-test",
        help="Verify final-layer HDBSCAN and fixed-label backward tracking",
    )
    self_test.add_argument("--seed", type=int, default=0)
    self_test.set_defaults(function=command_self_test)

    evaluate = subparsers.add_parser(
        "eval",
        help="HellaSwag prefill capture, final-layer HDBSCAN, backward label tracking",
    )
    evaluate.add_argument("--ckpt-path", required=True)
    evaluate.add_argument("--config", default="./llm/DS_V4_Flash/inference/config.json")
    evaluate.add_argument("--tokenizer-path", default=None)
    evaluate.add_argument(
        "--out-dir", default="./hellaswag_riemann_umap_hdbscan_tracking_results"
    )
    evaluate.add_argument(
        "--tilelang-backend",
        choices=("auto", "tvm_ffi", "nvrtc"),
        default="auto",
        help="auto|tvm_ffi|nvrtc; auto prefers tvm_ffi when CUDA toolkit exists",
    )
    prompt_source = evaluate.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--hellaswag-path",
        default=None,
        help=f"HellaSwag parquet path (default: {HELLASWAG_DEFAULT_PATH})",
    )
    prompt_source.add_argument("--prompt-file", default=None)
    prompt_source.add_argument(
        "--prompt-jsonl",
        default=None,
        help="JSONL with text (+ optional tag/id, group)",
    )
    prompt_source.add_argument(
        "--prompt-dir",
        default=None,
        help="Recursive .txt/.md; first subdir = group",
    )
    evaluate.add_argument(
        "--hellaswag-activities",
        default="all",
        help="Comma-separated HellaSwag activity labels or 'all'",
    )
    evaluate.add_argument("--samples-per-group", type=int, default=0)
    evaluate.add_argument("--sample-seed", type=int, default=0)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--max-tokens", type=int, default=2048)
    evaluate.add_argument("--max-seq-len", type=int, default=2048)
    evaluate.add_argument(
        "--long-prompt-policy",
        choices=("error", "head_tail"),
        default="error",
        help="error|head_tail for overlong prompts",
    )
    evaluate.add_argument(
        "--truncation-head-fraction",
        type=float,
        default=0.50,
        help="Head fraction for head_tail truncation",
    )
    evaluate.add_argument(
        "--num-observation-layers",
        type=int,
        default=12,
        help="Observation layers for metric curves (always includes 0/mid/final)",
    )

    evaluate.add_argument(
        "--umap-components",
        "--cluster-umap-components",
        dest="cluster_umap_components",
        type=int,
        default=128,
        help="Shared Riemann UMAP dims for clustering; sphere viz uses haversine S^2 UMAP",
    )
    evaluate.add_argument(
        "--umap-fit-tokens-per-layer",
        "--cluster-fit-tokens-per-layer",
        dest="cluster_fit_tokens_per_layer",
        type=int,
        default=256,
    )
    evaluate.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="Riemann UMAP n_neighbors (clamped to fit-set size)",
    )
    evaluate.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="Riemann UMAP min_dist",
    )
    evaluate.add_argument(
        "--umap-metric",
        default="cosine",
        help="Riemann UMAP input metric (default: cosine on unit sphere)",
    )
    evaluate.add_argument(
        "--spherical-cluster-space",
        choices=("umap", "hidden"),
        default="umap",
        help="Cluster in riemann-umap (default) or full hidden space",
    )
    evaluate.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=0,
        help="Absolute min cluster size; 0 => fraction (floor 10)",
    )
    evaluate.add_argument(
        "--hdbscan-min-cluster-fraction",
        type=float,
        default=0.05,
        help="min_cluster_size fraction when absolute is 0",
    )
    evaluate.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=0,
        help="Absolute min_samples; 0 => fraction (floor 5)",
    )
    evaluate.add_argument(
        "--hdbscan-min-samples-fraction",
        type=float,
        default=0.01,
        help="min_samples fraction when absolute is 0",
    )
    evaluate.add_argument(
        "--hdbscan-cluster-selection-method",
        choices=("eom", "leaf"),
        default="eom",
        help="HDBSCAN cluster selection: eom|leaf",
    )
    evaluate.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="cluster_selection_epsilon",
    )
    evaluate.add_argument(
        "--hdbscan-alpha",
        type=float,
        default=1.0,
        help="HDBSCAN alpha",
    )
    evaluate.add_argument(
        "--hdbscan-allow-single-cluster",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow HDBSCAN K=1 root cluster",
    )
    evaluate.add_argument(
        "--assign-all-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map HDBSCAN noise to nearest cluster (default on)",
    )
    evaluate.add_argument("--plot-tokens", type=int, default=512)
    evaluate.set_defaults(function=command_evaluate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
