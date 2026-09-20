from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

def make_separated_directions(
    count: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Well-separated directions for group centers or fixed representatives."""
    if count <= 0:
        raise ValueError("The number of directions must be positive.")
    if dim not in {2, 3}:
        raise ValueError("GIF visualization supports only --dim 2 or --dim 3.")

    if dim == 2:
        angles = 2.0 * math.pi * torch.arange(
            count, dtype=torch.float32, device=device
        ) / float(count)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    # Exact symmetric configurations where convenient.
    if count == 1:
        points = [[1.0, 0.0, 0.0]]
    elif count == 2:
        points = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    elif count == 3:
        root3 = math.sqrt(3.0)
        points = [
            [1.0, 0.0, 0.0],
            [-0.5, 0.5 * root3, 0.0],
            [-0.5, -0.5 * root3, 0.0],
        ]
    elif count == 4:
        points = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    elif count == 6:
        points = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    else:
        # Fibonacci sphere: deterministic, approximately uniform directions.
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        points = []
        for index in range(count):
            z = 1.0 - 2.0 * (index + 0.5) / count
            radius = math.sqrt(max(0.0, 1.0 - z * z))
            phi = index * golden_angle
            points.append([radius * math.cos(phi), radius * math.sin(phi), z])

    return normalize(torch.tensor(points, dtype=torch.float32, device=device))

def parse_float_list(spec: Optional[str], expected: int, name: str) -> Optional[list[float]]:
    if spec is None or not spec.strip():
        return None
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} requires {expected} comma-separated values; got {len(values)}.")
    return values

def parse_size_list(
    spec: Optional[str],
    total: int,
    groups: int,
    name: str,
) -> list[int]:
    """Parse group sizes, or split total as evenly as possible."""
    if groups <= 0:
        raise ValueError(f"{name}: the group count must be positive.")
    if spec is None or not spec.strip():
        base, remainder = divmod(total, groups)
        sizes = [base + (index < remainder) for index in range(groups)]
    else:
        sizes = [int(item.strip()) for item in spec.split(",") if item.strip()]
        if len(sizes) != groups:
            raise ValueError(f"{name} requires {groups} sizes; got {len(sizes)}.")
        if sum(sizes) != total:
            raise ValueError(f"{name} must sum to n={total}; got {sum(sizes)}.")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"All entries of {name} must be positive.")
    return [int(size) for size in sizes]

@torch.no_grad()
def sample_spherical_cap(
    count: int,
    center: torch.Tensor,
    alpha: float,
    seed: int,
) -> torch.Tensor:
    """Sample unit vectors at geodesic distance at most alpha from center."""
    if not (0.0 <= alpha < math.pi / 2.0):
        raise ValueError("A spherical-cap angle must lie in [0, pi/2).")
    torch.manual_seed(seed)
    center = normalize(center.reshape(-1))
    dim = center.numel()
    raw = torch.randn(count, dim, device=center.device)
    tangent = raw - (raw @ center).unsqueeze(-1) * center.unsqueeze(0)
    tangent = normalize(tangent)
    angles = alpha * torch.rand(count, device=center.device)
    points = (
        torch.cos(angles).unsqueeze(-1) * center.unsqueeze(0)
        + torch.sin(angles).unsqueeze(-1) * tangent
    )
    return normalize(points)

@torch.no_grad()
def sample_grouped_caps(
    centers: torch.Tensor,
    sizes: list[int],
    alpha: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one cap-supported group around every center."""
    if len(sizes) != centers.shape[0]:
        raise ValueError("The number of sizes must match the number of centers.")
    separation = minimum_pairwise_angle(centers)
    if centers.shape[0] > 1 and not (2.0 * alpha < separation):
        raise ValueError(
            "Group caps are not separated: require 2*cap_angle < minimum "
            f"center separation ({separation:.4f} rad)."
        )

    groups = []
    labels = []
    for group, (center, size) in enumerate(zip(centers, sizes)):
        groups.append(sample_spherical_cap(size, center, alpha, seed + 1009 * group))
        labels.append(torch.full((size,), group, dtype=torch.long, device=centers.device))
    return torch.cat(groups, dim=0), torch.cat(labels, dim=0)

@torch.no_grad()
def sample_spherical_cap_paired(
    count: int,
    center: torch.Tensor,
    alpha: float,
    seed: int,
) -> torch.Tensor:
    """Sample a cap whose empirical first moment is aligned with its center."""
    if count <= 0:
        raise ValueError("The number of paired cap samples must be positive.")
    if not (0.0 < alpha < math.pi / 2.0):
        raise ValueError("A paired spherical-cap angle must lie in (0, pi/2).")
    torch.manual_seed(seed)
    center = normalize(center.reshape(-1))
    pair_count = count // 2
    points: list[torch.Tensor] = []
    if pair_count > 0:
        raw = torch.randn(
            pair_count, center.numel(), dtype=center.dtype, device=center.device
        )
        tangent = normalize(
            raw - (raw @ center).unsqueeze(-1) * center.unsqueeze(0)
        )
        angles = alpha * torch.rand(
            pair_count, dtype=center.dtype, device=center.device
        )
        axial = torch.cos(angles).unsqueeze(-1) * center.unsqueeze(0)
        transverse = torch.sin(angles).unsqueeze(-1) * tangent
        points.extend([axial + transverse, axial - transverse])
    if count % 2 == 1:
        points.append(center.unsqueeze(0))
    samples = normalize(torch.cat(points, dim=0))
    return samples[torch.randperm(count, device=center.device)]

@torch.no_grad()
def sample_grouped_paired_caps(
    centers: torch.Tensor,
    sizes: list[int],
    alpha: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample antithetically paired cap populations around all centers."""
    if len(sizes) != centers.shape[0]:
        raise ValueError("The number of sizes must match the number of centers.")
    separation = minimum_pairwise_angle(centers)
    if centers.shape[0] > 1 and not (2.0 * alpha < separation):
        raise ValueError(
            "Group caps are not separated: require 2*cap_angle < minimum "
            f"center separation ({separation:.4f} rad)."
        )
    groups = []
    labels = []
    for group, (center, size) in enumerate(zip(centers, sizes)):
        groups.append(
            sample_spherical_cap_paired(size, center, alpha, seed + 1009 * group)
        )
        labels.append(
            torch.full((size,), group, dtype=torch.long, device=centers.device)
        )
    return torch.cat(groups, dim=0), torch.cat(labels, dim=0)

@torch.no_grad()
def explicit_theory_margins(
    directions: torch.Tensor,
    radii: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the inward and curvature margins in Assumption 3.1."""
    if not (0.0 < alpha < math.pi / 2.0):
        raise ValueError("--explicit-theorem-cap-angle must lie in (0, pi/2).")
    directions = normalize(directions.to(dtype=torch.float64))
    radii = radii.to(dtype=torch.float64)
    weights = weights.to(dtype=torch.float64)
    theta = torch.acos((directions @ directions.T).clamp(-1.0, 1.0))
    gamma = torch.cos(theta - alpha)
    cross = (
        (weights * radii).unsqueeze(0)
        * torch.exp(beta * radii.unsqueeze(0) * gamma)
    )
    off_diagonal = ~torch.eye(
        directions.shape[0], dtype=torch.bool, device=directions.device
    )
    own = weights * radii * torch.exp(beta * radii * math.cos(alpha))
    inward = own * math.sin(alpha) ** 2 - math.sin(alpha) * (
        cross * torch.sin(theta - alpha) * off_diagonal
    ).sum(dim=1)
    curvature = own * (
        math.cos(alpha) - beta * radii * math.sin(alpha) ** 2
    ) - (
        cross * (1.0 + beta * radii).unsqueeze(0) * off_diagonal
    ).sum(dim=1)
    return inward, curvature

@torch.no_grad()
def select_theory_compatible_common_radius(
    leader_directions: torch.Tensor,
    leader_weights: torch.Tensor,
    beta: float,
    alpha: float,
    target_relative_margin: float = 0.05,
) -> tuple[float, dict]:
    """Select a common radius with positive inward and curvature margins."""
    if beta <= 0.0:
        raise ValueError("beta must be positive before selecting leader radii.")
    if not (0.0 < target_relative_margin < 1.0):
        raise ValueError("target_relative_margin must lie in (0, 1).")
    sin_sq = math.sin(alpha) ** 2
    curvature_ceiling = math.cos(alpha) / max(sin_sq, 1e-12)
    scaled_upper = min(40.0, 0.98 * curvature_ceiling)
    scaled_lower = 0.02
    if scaled_upper <= scaled_lower:
        raise ValueError(
            "No positive-curvature common-radius search interval exists. "
            "Reduce --explicit-theorem-cap-angle."
        )
    directions = leader_directions.detach().to(device="cpu", dtype=torch.float64)
    weights = leader_weights.detach().to(device="cpu", dtype=torch.float64)
    best_positive: Optional[tuple[float, float, float, float]] = None
    for scaled_radius in np.linspace(scaled_lower, scaled_upper, 5000):
        radius = float(scaled_radius / beta)
        radii = torch.full_like(weights, radius)
        inward, curvature = explicit_theory_margins(
            directions, radii, weights, beta, alpha
        )
        own = weights * radii * torch.exp(beta * radii * math.cos(alpha))
        own_inward = own * sin_sq
        own_curvature = own * (math.cos(alpha) - beta * radii * sin_sq)
        if torch.any(own_curvature <= 0.0):
            continue
        min_inward = float(inward.min().item())
        min_curvature = float(curvature.min().item())
        if min_inward <= 0.0 or min_curvature <= 0.0:
            continue
        score = min(
            float((inward / own_inward).min().item()),
            float((curvature / own_curvature).min().item()),
        )
        candidate = (score, radius, min_inward, min_curvature)
        if best_positive is None or score > best_positive[0]:
            best_positive = candidate
        if score >= target_relative_margin:
            return radius, {
                "selection": "automatic common radius",
                "beta_times_radius": float(scaled_radius),
                "target_relative_margin": target_relative_margin,
                "achieved_relative_margin": score,
                "minimum_inward_margin": min_inward,
                "minimum_curvature_margin": min_curvature,
            }
    if best_positive is not None:
        score, radius, min_inward, min_curvature = best_positive
        return radius, {
            "selection": "automatic common radius (best positive candidate)",
            "beta_times_radius": float(beta * radius),
            "target_relative_margin": target_relative_margin,
            "achieved_relative_margin": score,
            "minimum_inward_margin": min_inward,
            "minimum_curvature_margin": min_curvature,
        }
    raise ValueError(
        "No common leader radius satisfies I_a(alpha)>0 and H_a(alpha)>0. "
        "Reduce the number of leaders or --explicit-theorem-cap-angle."
    )

@torch.no_grad()
def explicit_assumption_diagnostics(
    x: torch.Tensor,
    labels: torch.Tensor,
    directions: torch.Tensor,
    radii: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
    initialization_radius: float,
    theorem_alpha: float,
) -> dict:
    """Construct numerical witnesses for every condition in Assumption 3.1."""
    state = normalize(x.to(dtype=torch.float64))
    directions64 = normalize(directions.to(dtype=torch.float64))
    centers, moment_norms, group_deltas, leader_support_radii = [], [], [], []
    for group in range(directions.shape[0]):
        group_state = state[labels == group]
        if group_state.shape[0] == 0:
            raise ValueError(f"Explicit group {group + 1} is empty.")
        moment = group_state.mean(dim=0)
        moment_norm = float(moment.norm().item())
        if moment_norm <= 1e-12:
            raise ValueError(f"Explicit group {group + 1} has zero first moment.")
        center = normalize(moment)
        centers.append(center)
        moment_norms.append(moment_norm)
        group_deltas.append(float(torch.acos((group_state @ center).clamp(-1.0, 1.0)).max().item()))
        leader_support_radii.append(float(torch.acos((group_state @ directions64[group]).clamp(-1.0, 1.0)).max().item()))

    centers = torch.stack(centers)
    eta_actual = float(torch.acos((centers * directions64).sum(dim=1).clamp(-1.0, 1.0)).max().item())
    delta = max(max(group_deltas), 1e-12)
    Delta = minimum_pairwise_angle(centers)
    eta_upper = min(theorem_alpha - delta, 0.5 * Delta - theorem_alpha)
    geometry_has_witness = eta_upper > max(eta_actual, 0.0)
    eta_witness = (
        0.5 * (max(eta_actual, 0.0) + eta_upper)
        if geometry_has_witness else eta_actual
    )
    inward, curvature = explicit_theory_margins(
        directions64, radii, weights, beta, theorem_alpha
    )
    tolerance = 1e-10
    checks = {
        "m_at_least_two": directions.shape[0] >= 2,
        "all_group_moments_nonzero": min(moment_norms) > tolerance,
        "support_within_declared_initial_caps": max(leader_support_radii) <= initialization_radius + 2e-6,
        "positive_eta_witness_exists": geometry_has_witness,
        "delta_plus_eta_below_alpha": theorem_alpha - (delta + eta_witness) > tolerance,
        "separated_invariant_caps": Delta - 2.0 * (theorem_alpha + eta_witness) > tolerance,
        "all_inward_margins_positive": float(inward.min().item()) > tolerance,
        "all_curvature_margins_positive": float(curvature.min().item()) > tolerance,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "mode": "explicit Assumption 3.1 construction",
        "assumption": "Assumption 3.1 (Geometric separation and dominance)",
        "initialization_radius_bound": float(initialization_radius),
        "delta_witness": float(delta),
        "group_delta_values": group_deltas,
        "theorem_alpha": float(theorem_alpha),
        "eta_actual_max": float(eta_actual),
        "eta_witness": float(eta_witness),
        "Delta_witness": float(Delta),
        "localization_slack": float(theorem_alpha - (delta + eta_witness)),
        "separation_slack": float(Delta - 2.0 * (theorem_alpha + eta_witness)),
        "inward_margins": [float(value) for value in inward.cpu().tolist()],
        "curvature_margins": [float(value) for value in curvature.cpu().tolist()],
        "minimum_inward_margin": float(inward.min().item()),
        "minimum_curvature_margin": float(curvature.min().item()),
        "group_moment_norms": moment_norms,
        "leader_support_radii": leader_support_radii,
    }


TensorRHS = Callable[[torch.Tensor], torch.Tensor]


def normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vectors along the last axis."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def tangent_projection(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """P_x^perp(y) = y - <x,y>x for unit vectors x."""
    return y - (x * y).sum(dim=-1, keepdim=True) * x

def pairwise_angles(x: torch.Tensor) -> torch.Tensor:
    """Pairwise geodesic angles in radians."""
    return torch.acos((x @ x.T).clamp(-1.0, 1.0))

def minimum_pairwise_angle(x: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return math.inf
    angles = pairwise_angles(normalize(x)).clone()
    angles.fill_diagonal_(math.inf)
    return float(angles.min().item())

def maximum_pairwise_angle_deg(x: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(pairwise_angles(normalize(x)).max().item() * 180.0 / math.pi)

@torch.no_grad()
def explicit_attention_rhs(
    x: torch.Tensor,
    leader_directions: torch.Tensor,
    leader_radii: torch.Tensor,
    leader_weights: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Equation (1)/(17): the time-independent field of fixed representatives."""
    x = normalize(x)
    scores = beta * (x @ leader_directions.T) * leader_radii.unsqueeze(0)
    coefficients = torch.exp(scores) * (
        leader_weights * leader_radii
    ).unsqueeze(0)
    field = coefficients @ leader_directions
    return tangent_projection(x, field)

@torch.no_grad()
def rk4_step_on_sphere(x: torch.Tensor, rhs: TensorRHS, dt: float) -> torch.Tensor:
    """One fourth-order Runge-Kutta step followed by radial retraction."""
    k1 = rhs(x)
    k2 = rhs(normalize(x + 0.5 * dt * k1))
    k3 = rhs(normalize(x + 0.5 * dt * k2))
    k4 = rhs(normalize(x + dt * k3))
    return normalize(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
