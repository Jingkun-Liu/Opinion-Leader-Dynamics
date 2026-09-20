from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from geometry import (
    TensorRHS,
    explicit_assumption_diagnostics,
    explicit_attention_rhs,
    make_separated_directions,
    maximum_pairwise_angle_deg,
    parse_float_list,
    parse_size_list,
    rk4_step_on_sphere,
    sample_grouped_caps,
    sample_grouped_paired_caps,
    select_theory_compatible_common_radius,
)
from visualization import token_colors


@dataclass
class RunConfig:
    n: int
    dim: int
    beta: float
    final_time: float
    maximum_dt: float
    seed: int
    leaders: int
    explicit_cap_angle: float
    explicit_theorem_cap_angle: float
    explicit_theory_check: str
    explicit_init_mode: str
    explicit_auto_margin_ratio: float
    explicit_group_sizes: Optional[str]
    leader_radii_spec: Optional[str]
    leader_weights_spec: Optional[str]
    gif_frames: int
    snapshot_count: int


def evenly_spaced_steps(last_step: int, count: int) -> list[int]:
    if last_step < 0:
        raise ValueError("last_step must be nonnegative.")
    if count <= 1:
        return [0] if last_step == 0 else [0, last_step]
    return sorted(set(np.rint(np.linspace(0, last_step, count)).astype(int).tolist()))


def subset_steps(steps: list[int], count: int) -> list[int]:
    if count <= 0 or count >= len(steps):
        return list(steps)
    positions = np.rint(np.linspace(0, len(steps) - 1, count)).astype(int)
    return [steps[index] for index in sorted(set(positions.tolist()))]


@torch.no_grad()
def prepare_model(
    config: RunConfig,
    device: torch.device,
) -> tuple[torch.Tensor, TensorRHS, dict]:
    if config.leaders < 2:
        raise ValueError("The conditional explicit model requires --leaders >= 2.")
    if not (0.0 < config.explicit_cap_angle < math.pi / 2.0):
        raise ValueError("--explicit-cap-angle (delta) must lie in (0, pi/2).")
    if not (0.0 < config.explicit_theorem_cap_angle < math.pi / 2.0):
        raise ValueError("--explicit-theorem-cap-angle (alpha) must lie in (0, pi/2).")

    directions = make_separated_directions(config.leaders, config.dim, device)
    sizes = parse_size_list(
        config.explicit_group_sizes,
        config.n,
        config.leaders,
        "--explicit-group-sizes",
    )
    if config.explicit_init_mode == "paired":
        x, labels = sample_grouped_paired_caps(
            directions, sizes, config.explicit_cap_angle, config.seed
        )
    elif config.explicit_init_mode == "iid":
        x, labels = sample_grouped_caps(
            directions, sizes, config.explicit_cap_angle, config.seed
        )
    else:
        raise ValueError("--explicit-init-mode must be 'paired' or 'iid'.")

    weight_values = parse_float_list(
        config.leader_weights_spec,
        config.leaders,
        "--leader-weights",
    ) or [1.0 / config.leaders] * config.leaders
    weights = torch.tensor(weight_values, dtype=torch.float32, device=device)
    if torch.any(weights <= 0):
        raise ValueError("All representative field weights must be positive.")
    weights = weights / weights.sum()

    radii_spec = config.leader_radii_spec
    if radii_spec is None or radii_spec.strip().lower() == "auto":
        common_radius, radius_selection = select_theory_compatible_common_radius(
            directions,
            weights,
            config.beta,
            config.explicit_theorem_cap_angle,
            config.explicit_auto_margin_ratio,
        )
        radii = torch.full(
            (config.leaders,), common_radius, dtype=torch.float32, device=device
        )
        radii_option = "--leader-radii auto"
    else:
        radii_values = parse_float_list(
            radii_spec, config.leaders, "--leader-radii"
        )
        if radii_values is None:
            raise RuntimeError("Failed to parse explicit leader radii.")
        radii = torch.tensor(radii_values, dtype=torch.float32, device=device)
        radii_option = "--leader-radii"
        radius_selection = {
            "selection": "user-specified radii",
            "beta_times_max_radius": float(config.beta * radii.max().item()),
        }
    if torch.any(radii <= 0):
        raise ValueError("All representative radii must be positive.")
    if config.beta * float(radii.max().item()) >= 80.0:
        raise ValueError(
            "beta * max(leader radius) must be below 80 for stable float32 exponentials."
        )

    diagnostics = explicit_assumption_diagnostics(
        x,
        labels,
        directions,
        radii,
        weights,
        config.beta,
        config.explicit_cap_angle,
        config.explicit_theorem_cap_angle,
    )
    diagnostics["initialization_sampler"] = config.explicit_init_mode
    diagnostics["radius_selection"] = radius_selection
    if not diagnostics["verified"]:
        failed = [name for name, passed in diagnostics["checks"].items() if not passed]
        message = (
            "Explicit initialization does not satisfy Assumption 3.1. "
            f"Failed checks: {failed}. "
            f"delta={diagnostics['delta_witness']:.6g}, "
            f"eta={diagnostics['eta_actual_max']:.6g}, "
            f"alpha={diagnostics['theorem_alpha']:.6g}, "
            f"Delta={diagnostics['Delta_witness']:.6g}, "
            f"min I={diagnostics['minimum_inward_margin']:.6g}, "
            f"min H={diagnostics['minimum_curvature_margin']:.6g}."
        )
        if config.explicit_theory_check == "strict":
            raise ValueError(message)
        if config.explicit_theory_check == "warn":
            print(f"[explicit_conditional] WARNING: {message}")

    rhs = lambda state: explicit_attention_rhs(
        state, directions, radii, weights, config.beta
    )
    initial_velocity = rhs(x)
    radial_error = torch.abs((x * initial_velocity).sum(dim=-1))
    speed = initial_velocity.norm(dim=-1)
    absolute_tangency_error = float(radial_error.max().item())
    active = speed > 1e-8
    relative_tangency_error = (
        float((radial_error[active] / speed[active]).max().item())
        if torch.any(active)
        else 0.0
    )
    if absolute_tangency_error > 2e-4 and relative_tangency_error > 1e-5:
        raise RuntimeError(
            "Initial RHS is not numerically tangent to the sphere: "
            f"absolute error={absolute_tangency_error:.3e}, "
            f"relative error={relative_tangency_error:.3e}."
        )

    metadata = {
        "labels": labels,
        "leader_directions": directions,
        "leader_radii": radii,
        "leader_radii_source": radii_option,
        "leader_weights": weights,
        "group_sizes": sizes,
        "initialization": f"grouped spherical caps ({config.explicit_init_mode})",
        "theory_diagnostics": diagnostics,
        "radius_selection": radius_selection,
        "explicit_init_mode": config.explicit_init_mode,
        "initial_tangency_error": absolute_tangency_error,
        "initial_relative_tangency_error": relative_tangency_error,
    }
    return x, rhs, metadata


@torch.no_grad()
def run_simulation(config: RunConfig, device: torch.device, verbose: bool = True) -> dict:
    if config.n <= 0:
        raise ValueError("--n must be positive.")
    if config.beta <= 0.0:
        raise ValueError("--beta must be positive.")
    if config.final_time <= 0.0 or config.maximum_dt <= 0.0:
        raise ValueError("--T and --dt must be positive.")
    if config.gif_frames < 2:
        raise ValueError("--gif-frames must be at least 2.")

    x, rhs, metadata = prepare_model(config, device)
    x_initial = x.detach().cpu().numpy().copy()
    number_of_steps = int(math.ceil(config.final_time / config.maximum_dt))
    dt = config.final_time / number_of_steps
    gif_steps = evenly_spaced_steps(number_of_steps, config.gif_frames)
    montage_steps = subset_steps(gif_steps, config.snapshot_count)
    steps_to_store = set(gif_steps) | set(montage_steps)
    snapshots: dict[int, np.ndarray] = {}

    start = time.perf_counter()
    progress_stride = max(1, number_of_steps // 10)
    if verbose:
        print(
            f"\n[explicit_conditional] device={device}, n={config.n}, d={config.dim}, "
            f"beta={config.beta:g}, T={config.final_time:g}, dt={dt:.6g}, "
            f"RK4 steps={number_of_steps}"
        )
        diagnostics = metadata["theory_diagnostics"]
        print(
            "[explicit_conditional] Assumption 3.1 | "
            f"verified={diagnostics['verified']}, "
            f"delta={diagnostics['delta_witness']:.6g}, "
            f"eta={diagnostics['eta_witness']:.6g}, "
            f"alpha={diagnostics['theorem_alpha']:.6g}, "
            f"Delta={diagnostics['Delta_witness']:.6g}"
        )
        print(
            "[explicit_conditional] dominance margins | "
            f"min I={diagnostics['minimum_inward_margin']:.6g}, "
            f"min H={diagnostics['minimum_curvature_margin']:.6g}"
        )

    for step in range(number_of_steps + 1):
        if step in steps_to_store:
            snapshots[step] = x.detach().cpu().numpy().copy()
        if step == number_of_steps:
            break
        x = rk4_step_on_sphere(x, rhs, dt)
        if not torch.isfinite(x).all():
            raise FloatingPointError(
                f"Non-finite state at step {step + 1}; reduce --dt or --beta."
            )
        if verbose and (step + 1) % progress_stride == 0:
            print(
                f"[explicit_conditional] t={(step + 1) * dt:7.3f}/{config.final_time:g}"
            )

    unit_norm_error = float(torch.abs(x.norm(dim=-1) - 1.0).max().item())
    elapsed = time.perf_counter() - start
    colors = token_colors(config.n, metadata["labels"].detach().cpu().numpy())
    result = {
        "model": "explicit",
        "config": config,
        "dt": dt,
        "number_of_steps": number_of_steps,
        "gif_steps": gif_steps,
        "montage_steps": montage_steps,
        "snapshots": snapshots,
        "x_initial": x_initial,
        "x_final": x.detach().cpu().numpy().copy(),
        "colors": colors,
        "metadata": metadata,
        "unit_norm_error": unit_norm_error,
        "wall_time": elapsed,
    }
    if verbose:
        print(
            f"[explicit_conditional] done in {elapsed:.2f}s | "
            f"max pairwise angle={maximum_pairwise_angle_deg(x.cpu()):.3f} deg | "
            f"max |norm-1|={unit_norm_error:.2e}"
        )
    return result
