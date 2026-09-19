from __future__ import annotations

import math
from typing import Callable

import torch

TensorRHS = Callable[[torch.Tensor], torch.Tensor]


def normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def tangent_projection(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return y - (x * y).sum(dim=-1, keepdim=True) * x


def pairwise_angles(x: torch.Tensor) -> torch.Tensor:
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
def standard_attention_rhs(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Equation (3): global Standard Attention with the exact 1/n factor."""
    x = normalize(x)
    weights = torch.exp(beta * (x @ x.T))
    field = (weights @ x) / float(x.shape[0])
    return tangent_projection(x, field)


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
def implicit_attention_rhs(
    x: torch.Tensor,
    random_walk_matrix: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Equation (2): radius-graph interactions normalized by each node degree."""
    x = normalize(x)
    kernel = torch.exp(beta * (x @ x.T))
    field = (random_walk_matrix * kernel) @ x
    return tangent_projection(x, field)


@torch.no_grad()
def rk4_step_on_sphere(x: torch.Tensor, rhs: TensorRHS, dt: float) -> torch.Tensor:
    """One fourth-order Runge-Kutta step followed by radial retraction."""
    k1 = rhs(x)
    k2 = rhs(normalize(x + 0.5 * dt * k1))
    k3 = rhs(normalize(x + 0.5 * dt * k2))
    k4 = rhs(normalize(x + dt * k3))
    return normalize(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
