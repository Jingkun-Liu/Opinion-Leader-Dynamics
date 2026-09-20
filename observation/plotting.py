from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def sample_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the compact, JSON-ready fields used in run summaries."""
    result = record.get("hidden_state_umap", {})
    final_clustering = result.get("final_layer_clustering", {})
    tracking = result.get("tracking_summary", {})
    benchmark = record.get("benchmark", {})
    tokenization = record.get("tokenization", {})
    return {
        "tag": record.get("meta", {}).get("tag"),
        "group": record.get("meta", {}).get("group"),
        "dataset": benchmark.get("dataset"),
        "activity_label": benchmark.get("activity_label"),
        "source_id": benchmark.get("source_id"),
        "n_tokens": record.get("meta", {}).get("n_tokens"),
        "truncated": tokenization.get("truncated"),
        "gold_label_index": benchmark.get("gold_label_index"),
        "unique_id": benchmark.get("source_id", benchmark.get("unique_id")),
        "available": bool(result.get("available")),
        "n_observations": result.get("n_layers"),
        "final_n_clusters": final_clustering.get("n_clusters"),
        "final_cluster_status": final_clustering.get("status"),
        "final_noise_fraction": final_clustering.get("noise_fraction"),
        "mean_cluster_persistence": final_clustering.get(
            "mean_cluster_persistence"
        ),
        "final_reference_layer": final_clustering.get("reference_layer"),
        "first_fixed_label_cosine_silhouette": tracking.get(
            "first_fixed_label_cosine_silhouette"
        ),
        "final_fixed_label_cosine_silhouette": tracking.get(
            "final_fixed_label_cosine_silhouette"
        ),
        "final_minus_first_cosine_silhouette": tracking.get(
            "final_minus_first_cosine_silhouette"
        ),
        "interpretation_scope": result.get("interpretation_scope"),
        "clustering_space": result.get("clustering_space"),
    }


def _sphere_wireframe() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude, colatitude = np.mgrid[0 : 2 * math.pi : 31j, 0 : math.pi : 17j]
    return (
        np.cos(longitude) * np.sin(colatitude),
        np.sin(longitude) * np.sin(colatitude),
        np.cos(colatitude),
    )

def _snapshot_sphere_xyz(snapshot: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(key in snapshot for key in ("sphere_x", "sphere_y", "sphere_z")):
        return (
            np.asarray(snapshot["sphere_x"], dtype=np.float64),
            np.asarray(snapshot["sphere_y"], dtype=np.float64),
            np.asarray(snapshot["sphere_z"], dtype=np.float64),
        )
    axis1 = np.asarray(snapshot["axis1"], dtype=np.float64)
    axis2 = np.asarray(snapshot.get("axis2", np.zeros_like(axis1)), dtype=np.float64)
    axis3 = np.asarray(snapshot.get("axis3", np.zeros_like(axis1)), dtype=np.float64)
    coords = np.stack([axis1, axis2, axis3], axis=1)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    sphere = coords / norms
    return sphere[:, 0], sphere[:, 1], sphere[:, 2]

def save_hidden_state_umap_plot(
    result: Dict[str, Any],
    output_path: Path,
    gaussian_embedding_cosine_silhouette: Optional[float] = None,
) -> Dict[str, Any]:
    if not result.get("available"):
        return {
            "written": False,
            "reason": result.get("reason", "Hidden-state analysis is unavailable"),
        }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"written": False, "reason": f"matplotlib is unavailable: {exc}"}

    snapshots = result.get("snapshots_sphere") or []
    if not snapshots:
        return {"written": False, "reason": "No sphere snapshots are available"}
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    final_clustering = result["final_layer_clustering"]
    final_k = int(final_clustering["n_clusters"])
    base_color_map = plt.get_cmap("tab10" if final_k <= 10 else "tab20")
    cluster_colors = [
        base_color_map(cluster_id % base_color_map.N)
        for cluster_id in range(final_k)
    ]
    discrete_color_map = ListedColormap(cluster_colors) if cluster_colors else None

    def metric_text(value: Any, pattern: str) -> str:
        return "n/a" if value is None else format(float(value), pattern)

    def panel_title(snapshot: Dict[str, Any]) -> str:
        layer_id = int(snapshot["layer"])
        return "Embedding" if layer_id < 0 else f"Layer {layer_id}"

    n_columns = max(3, len(snapshots))
    figure = plt.figure(figsize=(4.6 * n_columns, 7.2))
    grid = figure.add_gridspec(
        2,
        n_columns,
        height_ratios=(3.2, 1.35),
        hspace=0.34,
        wspace=0.28,
    )
    final_snapshot = next(
        (
            snapshot
            for snapshot in snapshots
            if str(snapshot.get("snapshot_role", "")).strip().lower()
            in {"final", "last"}
        ),
        snapshots[-1],
    )
    final_metric_line = (
        f"$s_{{\\mathrm{{cos}}}}$="
        f"{metric_text(final_snapshot.get('fixed_label_cosine_silhouette'), '.2f')}"
    )
    wire_x, wire_y, wire_z = _sphere_wireframe()
    for panel_index, snapshot in enumerate(snapshots):
        axis = figure.add_subplot(grid[0, panel_index], projection="3d")
        sx, sy, sz = _snapshot_sphere_xyz(snapshot)
        labels = np.asarray(snapshot["final_cluster_label"], dtype=np.int64)
        axis.plot_wireframe(
            wire_x,
            wire_y,
            wire_z,
            color="#a8a8a8",
            alpha=0.18,
            linewidth=0.45,
            rstride=2,
            cstride=2,
        )
        assigned = labels >= 0
        noise = ~assigned
        if assigned.any() and final_k > 0:
            axis.scatter(
                sx[assigned],
                sy[assigned],
                sz[assigned],
                c=labels[assigned],
                cmap=discrete_color_map,
                vmin=-0.5,
                vmax=final_k - 0.5,
                s=22,
                alpha=0.86,
                linewidths=0,
                depthshade=False,
            )
        if noise.any():
            axis.scatter(
                sx[noise],
                sy[noise],
                sz[noise],
                c="#9aa0a6",
                s=16,
                alpha=0.42,
                linewidths=0,
                depthshade=False,
            )
        axis.set_xlim(-1.12, 1.12)
        axis.set_ylim(-1.12, 1.12)
        axis.set_zlim(-1.12, 1.12)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=18.0, azim=35.0)
        axis.set_xlabel(r"$S_x$")
        axis.set_ylabel(r"$S_y$")
        axis.set_zlabel(r"$S_z$", labelpad=2)
        axis.set_title(panel_title(snapshot), fontsize=10, pad=8)
        axis.set_xticks([-1.0, 0.0, 1.0])
        axis.set_yticks([-1.0, 0.0, 1.0])
        axis.set_zticks([-1.0, 0.0, 1.0])

    trace = [
        row
        for row in (result.get("layer_trace") or [])
        if int(row["layer"]) >= 0
    ]
    x_values = np.arange(len(trace), dtype=np.int64)
    x_labels = [str(int(row["layer"])) for row in trace]
    metric_axis = figure.add_subplot(grid[1, :])
    y_values = np.asarray(
        [
            (
                np.nan
                if row.get("fixed_label_cosine_silhouette") is None
                else float(row["fixed_label_cosine_silhouette"])
            )
            for row in trace
        ],
        dtype=np.float64,
    )
    if np.isfinite(y_values).any():
        metric_axis.plot(
            x_values,
            y_values,
            color="#276fbf",
            marker="o",
            markersize=3.8,
            linewidth=1.8,
        )
    else:
        metric_axis.text(
            0.5,
            0.5,
            "Undefined",
            ha="center",
            va="center",
            transform=metric_axis.transAxes,
            fontsize=9,
            color="#666666",
        )
    metric_axis.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
    if (
        gaussian_embedding_cosine_silhouette is not None
        and np.isfinite(float(gaussian_embedding_cosine_silhouette))
    ):
        metric_axis.axhline(
            float(gaussian_embedding_cosine_silhouette),
            color="#c44e52",
            linestyle="--",
            linewidth=1.6,
            zorder=3,
            label=r"Gaussian embedding $s_{\mathrm{cos}}$",
        )
        metric_axis.legend(loc="best", fontsize=8, frameon=False)
    metric_axis.set_title("Cosine Silhouette", fontsize=10)
    metric_axis.set_ylabel(r"$s_{\mathrm{cos}}$", fontsize=9)
    metric_axis.set_xlabel("Layer", fontsize=9)
    metric_axis.set_xticks(x_values)
    metric_axis.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=8)
    metric_axis.grid(alpha=0.22)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=cluster_colors[cluster_id],
            label=f"Cluster {cluster_id}",
            markersize=6,
        )
        for cluster_id in range(final_k)
    ]
    if int(final_clustering.get("noise_count", 0)) > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="#9aa0a6",
                label="Noise",
                markersize=6,
            )
        )
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905),
            ncol=max(1, min(len(legend_handles), 8)),
            frameon=False,
            fontsize=9,
        )

    figure.suptitle(
        "Token Trajectories on the Shared Riemann-UMAP Sphere",
        fontsize=11,
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        final_metric_line,
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.subplots_adjust(
        left=0.045, right=0.985, bottom=0.08, top=0.80, wspace=0.26, hspace=0.34
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    return {"written": True, "path": str(output_path)}
