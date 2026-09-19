from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from dynamics import (
    TensorRHS,
    explicit_attention_rhs,
    implicit_attention_rhs,
    maximum_pairwise_angle_deg,
    rk4_step_on_sphere,
    standard_attention_rhs,
)
from initial_conditions import (
    make_separated_directions,
    parse_float_list,
    parse_size_list,
    sample_grouped_caps,
    sample_spherical_cap,
    sample_uniform_sphere,
    sample_von_mises_fisher,
    von_mises_fisher_mean_resultant_length,
)
from radius_graph import RadiusGraph, build_radius_graph
from visualization import token_colors


@dataclass
class RunConfig:
    model: str
    n: int
    dim: int
    beta: float
    final_time: float
    maximum_dt: float
    seed: int
    leaders: int
    components: int
    standard_init: str
    standard_l2_kappa: float
    standard_cap_angle: float
    implicit_cap_angle: float
    implicit_component_sizes: Optional[str]
    leader_radii_spec: Optional[str]
    leader_weights_spec: Optional[str]
    interaction_radius: float
    coordinate_spacing: float
    component_gap: float
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
    model = config.model
    metadata: dict = {}

    if model == "standard":
        if config.standard_init == "uniform":
            x = sample_uniform_sphere(config.n, config.dim, device, config.seed)
        elif config.standard_init == "l2":
            center = torch.zeros(config.dim, device=device)
            center[0] = 1.0
            x = sample_von_mises_fisher(
                config.n,
                center,
                config.standard_l2_kappa,
                config.seed,
            )
            empirical_mean = x.mean(dim=0)
            theoretical_mean_norm = von_mises_fisher_mean_resultant_length(
                config.dim, config.standard_l2_kappa
            )
            metadata["standard_l2_center"] = center
            metadata["initialization_diagnostics"] = {
                "mode": "l2",
                "population_law": "von Mises-Fisher",
                "density_relative_to_uniform_measure": (
                    "exp(kappa * <center,x>) / Z_dim(kappa)"
                ),
                "kappa": float(config.standard_l2_kappa),
                "population_density_in_L2": True,
                "population_mean_nonzero": True,
                "theoretical_population_mean_norm": theoretical_mean_norm,
                "empirical_sample_mean_norm": float(empirical_mean.norm().item()),
                "empirical_sample_mean_projection_on_center": float(
                    torch.dot(empirical_mean, center).item()
                ),
                "empirical_measure_is_L2": False,
                "interpretation": (
                    "Finite-particle Monte Carlo approximation of an L2 "
                    "population law with nonzero mean."
                ),
                "small_beta_threshold_verified": False,
                "small_beta_note": (
                    "Theorem 2.2 asserts existence of a distribution-dependent "
                    "beta_0 but does not provide a threshold computable here."
                ),
            }
        elif config.standard_init == "cap":
            center = torch.zeros(config.dim, device=device)
            center[0] = 1.0
            x = sample_spherical_cap(
                config.n, center, config.standard_cap_angle, config.seed
            )
            condition_value = (
                10.0
                * (1.0 + math.sqrt(config.beta))
                * math.tan(config.standard_cap_angle)
            )
            metadata["standard_cap_condition_value"] = condition_value
            metadata["standard_cap_condition_satisfied"] = condition_value <= 1.0
        else:
            raise ValueError("--standard-init must be 'uniform', 'l2', or 'cap'.")
        rhs = lambda state: standard_attention_rhs(state, config.beta)

        # These labels affect only visualization colors, not the dynamics.
        color_reference_directions = make_separated_directions(
            config.leaders, config.dim, device
        )
        color_labels = torch.argmax(x @ color_reference_directions.T, dim=-1)
        metadata.update(
            {
                "labels": color_labels,
                "color_reference_directions": color_reference_directions,
                "color_note": (
                    "Visualization-only nearest-initial-direction labels; "
                    "they do not affect Standard Attention dynamics."
                ),
            }
        )

    elif model == "explicit":
        if config.leaders < 2:
            raise ValueError("The paper's explicit multi-cluster setting requires --leaders >= 2.")
        directions = make_separated_directions(config.leaders, config.dim, device)
        x = sample_uniform_sphere(config.n, config.dim, device, config.seed)
        labels = torch.argmax(x @ directions.T, dim=-1)
        sizes = torch.bincount(
            labels, minlength=config.leaders
        ).detach().cpu().tolist()
        initialization = "uniform sphere (no localization)"

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
        if radii_spec is not None and radii_spec.strip().lower() == "auto":
            raise ValueError("explicit requires numeric --leader-radii values.")
        radii_values = parse_float_list(
            radii_spec, config.leaders, "--leader-radii"
        ) or [1.0] * config.leaders
        radii = torch.tensor(radii_values, dtype=torch.float32, device=device)
        if torch.any(radii <= 0):
            raise ValueError("All representative radii must be positive.")
        if config.beta * float(radii.max().item()) >= 80.0:
            raise ValueError(
                "beta * max(leader radius) must be below 80 for stable float32 exponentials."
            )

        rhs = lambda state: explicit_attention_rhs(
            state, directions, radii, weights, config.beta
        )
        metadata.update(
            {
                "labels": labels,
                "leader_directions": directions,
                "leader_radii": radii,
                "leader_radii_source": "--leader-radii",
                "leader_weights": weights,
                "group_sizes": sizes,
                "initialization": initialization,
            }
        )

    elif model == "implicit":
        if config.components < 2:
            raise ValueError("The paper assumes --components >= 2 for implicit dynamics.")
        sizes = parse_size_list(
            config.implicit_component_sizes,
            config.n,
            config.components,
            "--implicit-component-sizes",
        )
        graph = build_radius_graph(
            sizes=sizes,
            interaction_radius=config.interaction_radius,
            coordinate_spacing=config.coordinate_spacing,
            component_gap=config.component_gap,
            device=device,
        )
        centers = make_separated_directions(config.components, config.dim, device)
        if not (0.0 <= config.implicit_cap_angle < math.pi / 4.0):
            raise ValueError("The paper requires --implicit-cap-angle in [0, pi/4).")
        x, sampled_labels = sample_grouped_caps(
            centers, sizes, config.implicit_cap_angle, config.seed
        )
        if not torch.equal(sampled_labels, graph.component_labels):
            raise RuntimeError("Token groups and detected graph components are misaligned.")
        rhs = lambda state: implicit_attention_rhs(
            state, graph.random_walk, config.beta
        )
        metadata.update(
            {
                "labels": graph.component_labels,
                "component_centers": centers,
                "component_sizes": sizes,
                "graph": graph,
            }
        )
    else:
        raise ValueError("model must be standard, explicit, or implicit.")

    initial_velocity = rhs(x)

    radial_error = torch.abs(
        (x * initial_velocity).sum(dim=-1)
    )
    speed = initial_velocity.norm(dim=-1)

    absolute_tangency_error = float(radial_error.max().item())

    active = speed > 1e-8
    if torch.any(active):
        relative_tangency_error = float(
            (radial_error[active] / speed[active]).max().item()
        )
    else:
        relative_tangency_error = 0.0

    if (
        absolute_tangency_error > 2e-4
        and relative_tangency_error > 1e-5
    ):
        raise RuntimeError(
            "Initial RHS is not numerically tangent to the sphere: "
            f"absolute error={absolute_tangency_error:.3e}, "
            f"relative error={relative_tangency_error:.3e}."
        )

    metadata["initial_tangency_error"] = absolute_tangency_error
    metadata["initial_relative_tangency_error"] = relative_tangency_error
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
            f"\n[{config.model}] device={device}, n={config.n}, d={config.dim}, "
            f"beta={config.beta:g}, T={config.final_time:g}, dt={dt:.6g}, "
            f"RK4 steps={number_of_steps}"
        )
        if config.model == "implicit":
            graph: RadiusGraph = metadata["graph"]
            rounded_gaps = [round(value, 6) for value in graph.spectral_gaps]
            print(
                f"[implicit] component sizes={metadata['component_sizes']}, "
                f"spectral gaps={rounded_gaps}"
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
                f"[{config.model}] t={(step + 1) * dt:7.3f}/{config.final_time:g}"
            )

    unit_norm_error = float(torch.abs(x.norm(dim=-1) - 1.0).max().item())
    elapsed = time.perf_counter() - start
    labels = metadata.get("labels")
    colors = token_colors(
        config.n,
        None if labels is None else labels.detach().cpu().numpy(),
    )

    result = {
        "model": config.model,
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
        final_cpu = x.detach().cpu()
        print(
            f"[{config.model}] done in {elapsed:.2f}s | "
            f"max pairwise angle={maximum_pairwise_angle_deg(final_cpu):.3f} deg | "
            f"max |norm-1|={unit_norm_error:.2e}"
        )
        if config.model == "standard" and config.standard_init == "cap":
            value = metadata["standard_cap_condition_value"]
            satisfied = metadata["standard_cap_condition_satisfied"]
            print(
                "[standard] localized-theorem condition "
                f"10(1+sqrt(beta))tan(alpha)={value:.6g}; satisfied={satisfied}"
            )
        if config.model == "standard" and config.standard_init == "l2":
            diagnostics = metadata["initialization_diagnostics"]
            print(
                "[standard] L2 population initialization | "
                f"law=vMF, kappa={diagnostics['kappa']:.6g}, "
                f"theoretical ||M0||={diagnostics['theoretical_population_mean_norm']:.6g}, "
                f"empirical ||M0^n||={diagnostics['empirical_sample_mean_norm']:.6g}"
            )
            print(
                "[standard] the vMF population law satisfies f0 in L2(sigma) "
                "and R0>0; the simulated empirical measure is its discrete "
                "Monte Carlo approximation."
            )
    return result
