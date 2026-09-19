from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RadiusGraph:
    coordinates: torch.Tensor
    adjacency: torch.Tensor
    degrees: torch.Tensor
    random_walk: torch.Tensor
    component_labels: torch.Tensor
    component_indices: list[torch.Tensor]
    spectral_gaps: list[float]


def connected_components(adjacency: torch.Tensor) -> list[torch.Tensor]:
    adjacency_cpu = adjacency.detach().cpu().bool()
    count = adjacency_cpu.shape[0]
    visited = torch.zeros(count, dtype=torch.bool)
    components: list[torch.Tensor] = []

    for start in range(count):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        members: list[int] = []
        while stack:
            node = stack.pop()
            members.append(node)
            neighbours = torch.where(adjacency_cpu[node])[0].tolist()
            for neighbour in neighbours:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        components.append(torch.tensor(sorted(members), dtype=torch.long))
    return components


@torch.no_grad()
def build_radius_graph(
    sizes: list[int],
    interaction_radius: float,
    coordinate_spacing: float,
    component_gap: float,
    device: torch.device,
) -> RadiusGraph:
    """Construct p_i in R, then A_ij = 1{|p_i-p_j| <= R}."""
    if interaction_radius <= 0.0:
        raise ValueError("--interaction-radius must be positive.")
    if coordinate_spacing <= 0.0:
        raise ValueError("--coordinate-spacing must be positive.")
    if component_gap <= 0.0:
        raise ValueError("--component-gap must be positive.")
    if any(size < 2 for size in sizes):
        raise ValueError("The paper assumes every implicit component has at least two nodes.")

    coordinate_blocks = []
    cursor = 0.0
    for size in sizes:
        block = cursor + coordinate_spacing * torch.arange(size, dtype=torch.float32)
        coordinate_blocks.append(block)
        cursor = float(block[-1].item()) + component_gap
    coordinates = torch.cat(coordinate_blocks).to(device)

    distances = torch.abs(coordinates[:, None] - coordinates[None, :])
    adjacency_bool = distances <= interaction_radius + 1e-7
    adjacency = adjacency_bool.to(torch.float32)
    degrees = adjacency.sum(dim=1)
    if torch.any(degrees <= 0):
        raise RuntimeError("Every radius-graph node must have its unit-diagonal self-loop.")

    components_cpu = connected_components(adjacency_bool)
    if len(components_cpu) != len(sizes):
        actual_sizes = [int(component.numel()) for component in components_cpu]
        raise ValueError(
            "The chosen radius/coordinate parameters do not create the requested "
            f"{len(sizes)} components. Actual component sizes: {actual_sizes}. "
            "Use radius >= spacing for within-component connectivity and "
            "radius < component-gap to prevent cross-component edges."
        )
    actual_sizes = [int(component.numel()) for component in components_cpu]
    if actual_sizes != sizes:
        raise ValueError(
            f"Detected radius-graph component sizes {actual_sizes}, expected {sizes}."
        )

    component_indices = [component.to(device) for component in components_cpu]
    labels = torch.empty(coordinates.shape[0], dtype=torch.long, device=device)
    spectral_gaps: list[float] = []
    for component_id, indices in enumerate(component_indices):
        labels[indices] = component_id
        sub_adjacency = adjacency[indices][:, indices]
        sub_degrees = sub_adjacency.sum(dim=1)
        inv_sqrt = torch.rsqrt(sub_degrees)
        normalized_adjacency = (
            inv_sqrt[:, None] * sub_adjacency * inv_sqrt[None, :]
        )
        laplacian = torch.eye(
            indices.numel(), dtype=sub_adjacency.dtype, device=device
        ) - normalized_adjacency
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        spectral_gaps.append(float(eigenvalues[1].item()))

    random_walk = adjacency / degrees.unsqueeze(1)
    return RadiusGraph(
        coordinates=coordinates,
        adjacency=adjacency,
        degrees=degrees,
        random_walk=random_walk,
        component_labels=labels,
        component_indices=component_indices,
        spectral_gaps=spectral_gaps,
    )
