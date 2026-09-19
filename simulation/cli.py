from __future__ import annotations

import argparse
import time

import torch

from simulation_core import RunConfig, run_simulation
from visualization import resolve_output_paths, save_gif, save_snapshot_montage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate Standard, Explicit Opinion Leader, and Implicit Opinion "
            "Leader attention dynamics on S^{d-1}; save GIFs and time-snapshot montages."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["standard", "explicit", "implicit"],
        default="standard",
        help="Attention model to simulate.",
    )
    parser.add_argument("--n", type=int, default=180, help="Number of token particles.")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=20.0, help="Final continuous time.")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Maximum RK4 step; it is reduced slightly when needed so the final frame is exactly T.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device.",
    )

    parser.add_argument("--leaders", type=int, default=3, help="Number m of fixed representatives.")
    parser.add_argument(
        "--leader-radii",
        type=str,
        default=None,
        help="Comma list of r_a values; omitted means all radii are 1.",
    )
    parser.add_argument("--leader-weights", type=str, default=None, help="Comma list of rho_a values.")

    parser.add_argument("--components", type=int, default=3, help="Requested radius-graph component count K.")
    parser.add_argument(
        "--implicit-component-sizes",
        type=str,
        default=None,
        help="Comma list of K component sizes summing to n; omitted means balanced components.",
    )
    parser.add_argument("--interaction-radius", type=float, default=4.05, help="Radius R in coordinate space.")
    parser.add_argument("--coordinate-spacing", type=float, default=1.0, help="Spacing of p_i inside a component.")
    parser.add_argument(
        "--component-gap",
        type=float,
        default=6.0,
        help="Coordinate distance between the last node of one component and the first of the next.",
    )
    parser.add_argument(
        "--implicit-cap-angle",
        type=float,
        default=0.22,
        help="Initial cap radius for each graph component, in radians; must be below pi/4.",
    )

    parser.add_argument(
        "--standard-init",
        choices=["uniform", "l2", "cap"],
        default="uniform",
        help=(
            "Use 'l2' for an empirical approximation of a smooth vMF density "
            "with nonzero population mean (Theorem 2.2 initialization), or "
            "'cap' to match the localized theorem assumption."
        ),
    )
    parser.add_argument(
        "--standard-l2-kappa",
        type=float,
        default=1.0,
        help=(
            "Positive vMF concentration used by --standard-init l2. Larger "
            "values give a larger nonzero population mean."
        ),
    )
    parser.add_argument(
        "--standard-cap-angle",
        type=float,
        default=0.02,
        help="Cap radius used when --standard-init cap.",
    )

    parser.add_argument("--gif-frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=12,
        help="Number of GIF frames placed in the static montage; use 0 for every GIF frame.",
    )
    parser.add_argument("--snapshot-columns", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=135)
    parser.add_argument("--output-dir", type=str, default="attention_dynamics_outputs")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "Optional output path. The paired GIF/PNG name is derived automatically."
        ),
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Backward-compatible no-op: GIF generation is enabled by default.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the ODE only and do not write GIF/figure files.",
    )
    return parser


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(spec)


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.dpi <= 0 or args.snapshot_columns <= 0:
        raise ValueError("--fps, --dpi, and --snapshot-columns must be positive.")
    device = choose_device(args.device)
    total_start = time.perf_counter()

    config = RunConfig(
        model=args.model,
        n=args.n,
        dim=args.dim,
        beta=args.beta,
        final_time=args.T,
        maximum_dt=args.dt,
        seed=args.seed,
        leaders=args.leaders,
        components=args.components,
        standard_init=args.standard_init,
        standard_l2_kappa=args.standard_l2_kappa,
        standard_cap_angle=args.standard_cap_angle,
        implicit_cap_angle=args.implicit_cap_angle,
        implicit_component_sizes=args.implicit_component_sizes,
        leader_radii_spec=args.leader_radii,
        leader_weights_spec=args.leader_weights,
        interaction_radius=args.interaction_radius,
        coordinate_spacing=args.coordinate_spacing,
        component_gap=args.component_gap,
        gif_frames=args.gif_frames,
        snapshot_count=args.snapshot_count,
    )
    result = run_simulation(config, device=device, verbose=True)
    if not args.no_plot:
        gif_path, montage_path = resolve_output_paths(args, args.model)
        if not args.no_gif:
            save_gif(result, gif_path, fps=args.fps, dpi=args.dpi)
        if not args.no_snapshots:
            save_snapshot_montage(
                result,
                montage_path,
                columns=args.snapshot_columns,
                dpi=args.dpi,
            )

    print(f"\nSimulation completed in {time.perf_counter() - total_start:.2f}s.")
