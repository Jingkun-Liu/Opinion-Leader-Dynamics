from __future__ import annotations

import math
from typing import Optional

import torch

from dynamics import minimum_pairwise_angle, normalize


def make_separated_directions(
    count: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("The number of directions must be positive.")
    if dim not in {2, 3}:
        raise ValueError("GIF visualization supports only --dim 2 or --dim 3.")

    if dim == 2:
        angles = 2.0 * math.pi * torch.arange(
            count, dtype=torch.float32, device=device
        ) / float(count)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

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
def sample_uniform_sphere(
    count: int,
    dim: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return normalize(torch.randn(count, dim, device=device))


def von_mises_fisher_mean_resultant_length(dim: int, kappa: float) -> float:
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("The vMF concentration kappa must be finite and positive.")

    concentration = torch.tensor(kappa, dtype=torch.float64)
    if dim == 2:
        # Scaled Bessel functions avoid overflow for large kappa.
        return float(
            (
                torch.special.i1e(concentration)
                / torch.special.i0e(concentration)
            ).item()
        )
    if dim == 3:
        if kappa < 1e-3:
            # Stable expansion of coth(kappa) - 1/kappa near zero.
            return float(
                kappa / 3.0
                - kappa**3 / 45.0
                + 2.0 * kappa**5 / 945.0
            )
        return float(1.0 / math.tanh(kappa) - 1.0 / kappa)
    raise ValueError("The current visualization code supports only dim=2 or dim=3.")


@torch.no_grad()
def sample_von_mises_fisher(
    count: int,
    center: torch.Tensor,
    kappa: float,
    seed: int,
) -> torch.Tensor:
    """Sample a von Mises--Fisher distribution with Wood rejection sampling."""
    if count <= 0:
        raise ValueError("The number of vMF samples must be positive.")
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("--standard-l2-kappa must be finite and strictly positive.")

    torch.manual_seed(seed)
    center = normalize(center.reshape(-1))
    dim = int(center.numel())
    if dim < 2:
        raise ValueError("von Mises--Fisher sampling requires dimension at least two.")

    dtype = center.dtype
    device = center.device
    dim_minus_one = float(dim - 1)

    b = dim_minus_one / (
        math.sqrt(4.0 * kappa * kappa + dim_minus_one * dim_minus_one)
        + 2.0 * kappa
    )
    x0 = (1.0 - b) / (1.0 + b)
    log_normalizer = (
        kappa * x0 + dim_minus_one * math.log1p(-(x0 * x0))
    )
    beta_shape = torch.tensor(
        0.5 * dim_minus_one, dtype=dtype, device=device
    )
    beta_distribution = torch.distributions.Beta(beta_shape, beta_shape)
    tiny = torch.finfo(dtype).tiny

    accepted_local: list[torch.Tensor] = []
    remaining = count
    for _ in range(10000):
        if remaining == 0:
            break
        batch_size = max(64, 2 * remaining)
        z = beta_distribution.sample((batch_size,))
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        acceptance_score = (
            kappa * w
            + dim_minus_one
            * torch.log((1.0 - x0 * w).clamp_min(tiny))
            - log_normalizer
        )
        log_uniform = torch.log(
            torch.rand(batch_size, dtype=dtype, device=device).clamp_min(tiny)
        )
        accepted_w = w[acceptance_score >= log_uniform][:remaining]
        if accepted_w.numel() == 0:
            continue

        tangent = normalize(
            torch.randn(
                accepted_w.numel(), dim - 1, dtype=dtype, device=device
            )
        )
        local_points = torch.cat(
            (
                accepted_w.unsqueeze(-1),
                torch.sqrt((1.0 - accepted_w.square()).clamp_min(0.0)).unsqueeze(-1)
                * tangent,
            ),
            dim=-1,
        )
        accepted_local.append(local_points)
        remaining -= int(accepted_w.numel())

    if remaining != 0:
        raise RuntimeError(
            "The vMF rejection sampler did not finish; reduce "
            "--standard-l2-kappa and retry."
        )

    samples = torch.cat(accepted_local, dim=0)
    first_axis = torch.zeros(dim, dtype=dtype, device=device)
    first_axis[0] = 1.0
    householder_vector = first_axis - center
    householder_norm = float(householder_vector.norm().item())
    if householder_norm > 1e-7:
        householder_vector = householder_vector / householder_norm
        samples = samples - 2.0 * (
            samples @ householder_vector
        ).unsqueeze(-1) * householder_vector.unsqueeze(0)
    return normalize(samples)


@torch.no_grad()
def sample_spherical_cap(
    count: int,
    center: torch.Tensor,
    alpha: float,
    seed: int,
) -> torch.Tensor:
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
