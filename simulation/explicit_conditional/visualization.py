from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from simulation_core import RunConfig

_MPL_CACHE = Path(tempfile.gettempdir()) / "opinion_leader_matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


MODEL_TITLES = {
    "explicit": "Conditional Explicit Opinion Leader Dynamics",
}


def token_colors(count: int, labels: Optional[np.ndarray]) -> np.ndarray:
    if labels is None:
        return np.repeat(np.asarray(plt.cm.Set2(0.0))[None, :], count, axis=0)
    labels = np.asarray(labels, dtype=int)
    number_of_groups = int(labels.max()) + 1
    cmap = plt.cm.Set2 if number_of_groups <= 8 else plt.cm.tab20
    return cmap((labels % cmap.N) / max(1, cmap.N - 1))


def draw_circle(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), color="#8d8d8d", lw=0.8, alpha=0.55)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.set_axis_off()


def sphere_wireframe() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude, colatitude = np.mgrid[0 : 2 * math.pi : 31j, 0 : math.pi : 17j]
    return (
        np.cos(longitude) * np.sin(colatitude),
        np.sin(longitude) * np.sin(colatitude),
        np.cos(colatitude),
    )


def draw_sphere(ax: plt.Axes) -> None:
    sx, sy, sz = sphere_wireframe()
    ax.plot_wireframe(
        sx, sy, sz, color="#a8a8a8", alpha=0.16, linewidth=0.45, rstride=2, cstride=2
    )
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_zlim(-1.12, 1.12)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.view_init(elev=18.0, azim=35.0)


def leader_flag_marker() -> MplPath:
    return MplPath(
        vertices=[
            (-0.14, -1.12),
            (0.10, -1.12),
            (0.10, 0.10),
            (1.48, 0.52),
            (0.10, 0.94),
            (0.10, 1.12),
            (-0.14, 1.12),
            (-0.14, -1.12),
            (0.0, 0.0),
        ],
        codes=[
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ],
    )


LEADER_FLAG_MARKER = leader_flag_marker()
LEADER_FLAG_STYLE = {
    "marker": LEADER_FLAG_MARKER,
    "s": 260,
    "facecolor": "white",
    "edgecolor": "#252525",
    "linewidth": 1.05,
}


def draw_explicit_representatives_2d(ax: plt.Axes, result: dict) -> None:
    leaders = result["metadata"]["leader_directions"].detach().cpu().numpy()
    ax.scatter(
        leaders[:, 0],
        leaders[:, 1],
        zorder=6,
        **LEADER_FLAG_STYLE,
    )


def draw_explicit_representatives_3d(ax: plt.Axes, result: dict) -> None:
    leaders = result["metadata"]["leader_directions"].detach().cpu().numpy()
    ax.scatter(
        leaders[:, 0],
        leaders[:, 1],
        leaders[:, 2],
        depthshade=False,
        **LEADER_FLAG_STYLE,
    )


def model_subtitle(result: dict) -> str:
    config: RunConfig = result["config"]
    initialization = result["metadata"]["initialization"]
    return (
        f"n={config.n}, beta={config.beta:g}, representatives={config.leaders}, "
        f"initialization={initialization}"
    )


def save_snapshot_montage(
    result: dict,
    output_path: Path,
    columns: int,
    dpi: int,
) -> None:
    steps = result["montage_steps"]
    config: RunConfig = result["config"]
    columns = min(max(1, columns), len(steps))
    rows = int(math.ceil(len(steps) / columns))
    subplot_size = 3.45 if config.dim == 2 else 4.0
    fig = plt.figure(figsize=(subplot_size * columns, subplot_size * rows + 0.75))

    for position, step in enumerate(steps, start=1):
        if config.dim == 2:
            ax = fig.add_subplot(rows, columns, position)
            draw_circle(ax)
            state = result["snapshots"][step]
            ax.scatter(
                state[:, 0],
                state[:, 1],
                s=22,
                c=result["colors"],
                alpha=0.86,
                linewidths=0.0,
                zorder=3,
            )
            draw_explicit_representatives_2d(ax, result)
        else:
            ax = fig.add_subplot(rows, columns, position, projection="3d")
            draw_sphere(ax)
            state = result["snapshots"][step]
            ax.scatter(
                state[:, 0],
                state[:, 1],
                state[:, 2],
                s=19,
                c=result["colors"],
                alpha=0.9,
                linewidths=0.0,
                depthshade=False,
            )
            draw_explicit_representatives_3d(ax, result)
        ax.set_title(f"t = {step * result['dt']:.3f}", fontsize=16, pad=6)

    fig.suptitle(
        f"{MODEL_TITLES[result['model']]}\n{model_subtitle(result)}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945), pad=0.7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[{result['model']}] snapshot montage -> {output_path}")


def save_gif(result: dict, output_path: Path, fps: int, dpi: int) -> None:
    config: RunConfig = result["config"]
    steps = result["gif_steps"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.dim == 2:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        draw_circle(ax)
        initial = result["snapshots"][steps[0]]
        scatter = ax.scatter(
            initial[:, 0],
            initial[:, 1],
            s=24,
            c=result["colors"],
            alpha=0.86,
            linewidths=0.0,
            zorder=3,
        )
        draw_explicit_representatives_2d(ax, result)
        title = ax.set_title("", fontsize=16, pad=8)

        def update(frame_index: int):
            step = steps[frame_index]
            state = result["snapshots"][step]
            scatter.set_offsets(state[:, :2])
            title.set_text(
                f"{MODEL_TITLES[result['model']]}\n"
                f"t = {step * result['dt']:.3f}"
            )
            return scatter, title

        movie = animation.FuncAnimation(
            fig, update, frames=len(steps), interval=1000.0 / fps, blit=True
        )
    else:
        fig = plt.figure(figsize=(6.6, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        draw_sphere(ax)
        initial = result["snapshots"][steps[0]]
        scatter = ax.scatter(
            initial[:, 0],
            initial[:, 1],
            initial[:, 2],
            s=22,
            c=result["colors"],
            alpha=0.9,
            linewidths=0.0,
            depthshade=False,
        )
        draw_explicit_representatives_3d(ax, result)
        title = ax.set_title("", fontsize=16, pad=8)

        def update(frame_index: int):
            step = steps[frame_index]
            state = result["snapshots"][step]
            scatter._offsets3d = (state[:, 0], state[:, 1], state[:, 2])
            title.set_text(
                f"{MODEL_TITLES[result['model']]}\n"
                f"t = {step * result['dt']:.3f}"
            )
            return scatter, title

        movie = animation.FuncAnimation(
            fig, update, frames=len(steps), interval=1000.0 / fps, blit=False
        )

    movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[{result['model']}] GIF -> {output_path}")


def resolve_output_paths(args: argparse.Namespace, model: str) -> tuple[Path, Path]:
    if args.save is None:
        stem = f"{model}_attention_dynamics"
        base = Path(args.output_dir) / stem
        return base.with_suffix(".gif"), base.with_name(base.name + "_snapshots.png")

    path = Path(args.save)
    suffix = path.suffix.lower()
    if suffix == ".gif":
        gif_path = path
        png_path = path.with_name(path.stem + "_snapshots.png")
    elif suffix in {".png", ".pdf", ".svg"}:
        png_path = path
        gif_path = path.with_suffix(".gif")
    elif suffix:
        raise ValueError("--save must end in .gif, .png, .pdf, or .svg.")
    else:
        gif_path = path.with_suffix(".gif")
        png_path = path.with_name(path.name + "_snapshots.png")
    return gif_path, png_path
