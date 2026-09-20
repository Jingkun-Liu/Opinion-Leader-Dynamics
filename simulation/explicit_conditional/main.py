from __future__ import annotations

import argparse
import time

import torch

from simulation_core import RunConfig, run_simulation
from visualization import resolve_output_paths, save_gif, save_snapshot_montage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate conditional Explicit Opinion Leader attention dynamics "
            "on S^{d-1}."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=180)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--leaders", type=int, default=6)
    parser.add_argument(
        "--leader-radii",
        type=str,
        default="auto",
        help="Comma list of r_a values or 'auto'.",
    )
    parser.add_argument("--leader-weights", type=str, default=None)
    parser.add_argument("--explicit-group-sizes", type=str, default=None)
    parser.add_argument("--explicit-cap-angle", type=float, default=0.25)
    parser.add_argument("--explicit-theorem-cap-angle", type=float, default=0.28)
    parser.add_argument(
        "--explicit-theory-check",
        choices=["strict", "warn", "off"],
        default="strict",
    )
    parser.add_argument(
        "--explicit-init-mode",
        choices=["paired", "iid"],
        default="paired",
    )
    parser.add_argument("--explicit-auto-margin-ratio", type=float, default=0.05)
    parser.add_argument("--gif-frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--snapshot-count", type=int, default=5)
    parser.add_argument("--snapshot-columns", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=135)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
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

    config = RunConfig(
        n=args.n,
        dim=args.dim,
        beta=args.beta,
        final_time=args.T,
        maximum_dt=args.dt,
        seed=args.seed,
        leaders=args.leaders,
        explicit_cap_angle=args.explicit_cap_angle,
        explicit_theorem_cap_angle=args.explicit_theorem_cap_angle,
        explicit_theory_check=args.explicit_theory_check,
        explicit_init_mode=args.explicit_init_mode,
        explicit_auto_margin_ratio=args.explicit_auto_margin_ratio,
        explicit_group_sizes=args.explicit_group_sizes,
        leader_radii_spec=args.leader_radii,
        leader_weights_spec=args.leader_weights,
        gif_frames=args.gif_frames,
        snapshot_count=args.snapshot_count,
    )
    device = choose_device(args.device)
    start = time.perf_counter()
    result = run_simulation(config, device=device, verbose=True)
    if not args.no_plot:
        gif_path, montage_path = resolve_output_paths(args, "explicit_conditional")
        if not args.no_gif:
            save_gif(result, gif_path, fps=args.fps, dpi=args.dpi)
        if not args.no_snapshots:
            save_snapshot_montage(
                result,
                montage_path,
                columns=args.snapshot_columns,
                dpi=args.dpi,
            )
    print(f"\nSimulation completed in {time.perf_counter() - start:.2f}s.")


if __name__ == "__main__":
    main()
