from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from prompts import normalize_rows


@dataclass
class LayerHidden:
    layer_id: int
    hidden: torch.Tensor


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class HiddenStateRecorder:
    EMBEDDING_LAYER_ID = -1

    def __init__(
        self,
        model: torch.nn.Module,
        max_tokens: int = 1024,
        num_observation_layers: int = 12,
    ):
        self.model = model
        self.max_tokens = int(max_tokens)
        self.num_observation_layers = int(num_observation_layers)
        self.layers: Dict[int, LayerHidden] = {}
        self.observation_positions: List[int] = []
        self._handles: List[Any] = []

    def _store(self, layer_id: int, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        if tensor.ndim >= 3 and tensor.shape[0] == 1:
            token_features = tensor[0, : self.max_tokens]
        elif tensor.ndim == 2:
            token_features = tensor[: self.max_tokens]
        else:
            return
        if token_features.ndim < 2 or token_features.shape[0] < 2:
            return
        token_features = token_features.detach().reshape(token_features.shape[0], -1)
        if not torch.isfinite(token_features).all():
            return
        hidden = normalize_rows(token_features.float()).to(torch.float16).cpu()
        self.layers[int(layer_id)] = LayerHidden(int(layer_id), hidden)

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("HiddenStateRecorder is already installed")
        if not hasattr(self.model, "layers") or len(self.model.layers) == 0:
            raise RuntimeError("Model does not expose a nonempty .layers sequence")

        n_layers = len(self.model.layers)
        self.observation_positions = select_observation_layer_indices(
            n_layers, self.num_observation_layers
        )

        def embedding_hook(_module, inputs):
            self._store(self.EMBEDDING_LAYER_ID, inputs)

        self._handles.append(
            self.model.layers[0].register_forward_pre_hook(embedding_hook)
        )

        for fallback_id in self.observation_positions:
            layer = self.model.layers[fallback_id]
            layer_id = int(getattr(layer, "layer_id", fallback_id))

            def block_hook(_module, _inputs, output, *, _layer_id=layer_id):
                self._store(_layer_id, output)

            self._handles.append(layer.register_forward_hook(block_hook))

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def select_observation_layer_indices(n_layers: int, count: int) -> List[int]:
    if n_layers <= 0:
        return []
    if n_layers <= 3 or count <= 0 or count >= n_layers:
        return list(range(n_layers))
    target_count = min(n_layers, max(3, int(count)))
    required = {0, n_layers // 2, n_layers - 1}
    selected = set(required)
    while len(selected) < target_count:
        candidate = max(
            (index for index in range(n_layers) if index not in selected),
            key=lambda index: (
                min(abs(index - chosen) for chosen in selected),
                -index,
            ),
        )
        selected.add(candidate)
    return sorted(selected)


def _evenly_spaced_indices(n: int, count: int) -> torch.Tensor:
    if n <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long, device="cpu")
    count = min(n, count)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device="cpu")
    index = torch.arange(count, dtype=torch.long, device="cpu")
    return index * (n - 1) // (count - 1)


@torch.no_grad()
def cosine_distance_matrix(
    x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    left = normalize_rows(x.detach().to(device="cpu", dtype=torch.float32))
    right = left if y is None else normalize_rows(
        y.detach().to(device="cpu", dtype=torch.float32)
    )
    return (1.0 - left @ right.T).clamp_(0.0, 2.0)

def require_hdbscan():
    try:
        import hdbscan
    except ImportError as exc:
        raise RuntimeError("Install hdbscan: pip install hdbscan") from exc
    if not hasattr(hdbscan, "HDBSCAN"):
        raise RuntimeError("hdbscan package has no HDBSCAN class")
    try:
        version = importlib.metadata.version("hdbscan")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return hdbscan, version

def resolve_hdbscan_parameters(
    n_tokens: int,
    min_cluster_size: int,
    min_cluster_fraction: float,
    min_samples: int,
    min_samples_fraction: float,
) -> Dict[str, Any]:
    n = int(n_tokens)
    if n < 4:
        raise ValueError("HDBSCAN requires at least four aligned tokens")
    req_size, req_samples = int(min_cluster_size), int(min_samples)
    frac_size, frac_samples = float(min_cluster_fraction), float(min_samples_fraction)
    if req_size < 0 or req_samples < 0:
        raise ValueError("HDBSCAN count parameters must be nonnegative")
    if not 0.0 < frac_size <= 1.0 or not 0.0 < frac_samples <= 1.0:
        raise ValueError("HDBSCAN fraction parameters must be in (0, 1]")
    if req_size > 0:
        size, size_src = req_size, "absolute_argument"
    else:
        size = min(n, max(10, int(math.ceil(frac_size * n))))
        size_src = "fraction_with_lower_bound_10"
    if not 2 <= size <= n:
        raise ValueError(f"Resolved HDBSCAN min_cluster_size must be in [2, {n}]; got {size}")
    if req_samples > 0:
        samples, samples_src = req_samples, "absolute_argument"
    else:
        samples = min(size, max(5, int(math.ceil(frac_samples * n))))
        samples_src = "fraction_with_lower_bound_5"
    if not 1 <= samples <= n:
        raise ValueError(f"Resolved HDBSCAN min_samples must be in [1, {n}]; got {samples}")
    return {
        "min_cluster_size": int(size),
        "min_samples": int(samples),
        "min_cluster_size_source": size_src,
        "min_samples_source": samples_src,
        "requested_min_cluster_size": req_size,
        "requested_min_cluster_fraction": frac_size,
        "requested_min_samples": req_samples,
        "requested_min_samples_fraction": frac_samples,
    }

@torch.no_grad()
def assign_noise_to_nearest_cluster(
    coordinates: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x = normalize_rows(coordinates.detach().to(device="cpu", dtype=torch.float32))
    assigned_labels = labels.detach().to(device="cpu", dtype=torch.long).reshape(-1).clone()
    if x.ndim != 2 or x.shape[0] != assigned_labels.numel():
        raise ValueError("Coordinates and labels must contain the same tokens")
    noise_mask = assigned_labels < 0
    assigned_mask = ~noise_mask
    n_noise = int(noise_mask.sum())
    if n_noise == 0:
        return assigned_labels, {
            "reassigned_noise_count": 0,
            "fallback_single_cluster": False,
        }
    if not assigned_mask.any():
        return torch.zeros_like(assigned_labels), {
            "reassigned_noise_count": n_noise,
            "fallback_single_cluster": True,
        }

    unique = torch.unique(assigned_labels[assigned_mask], sorted=True)
    centers = []
    for cluster_id in unique.tolist():
        member = x[assigned_labels == int(cluster_id)]
        resultant = member.mean(dim=0)
        if float(resultant.norm()) <= 1e-12:
            centers.append(member[0].clone())
        else:
            centers.append(normalize_rows(resultant.unsqueeze(0)).squeeze(0))
    center_matrix = torch.stack(centers, dim=0)
    noise_points = x[noise_mask]
    nearest = torch.argmax(noise_points @ center_matrix.T, dim=1)
    assigned_labels[noise_mask] = unique[nearest]
    return assigned_labels, {
        "reassigned_noise_count": n_noise,
        "fallback_single_cluster": False,
    }

@torch.no_grad()
def fit_final_layer_hdbscan(
    coordinates: torch.Tensor,
    min_cluster_size: int = 0,
    min_cluster_fraction: float = 0.05,
    min_samples: int = 0,
    min_samples_fraction: float = 0.01,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
    allow_single_cluster: bool = False,
    assign_all_tokens: bool = True,
) -> Dict[str, Any]:
    x = normalize_rows(coordinates.detach().to(device="cpu", dtype=torch.float32))
    if x.ndim != 2 or x.shape[0] < 4 or not torch.isfinite(x).all():
        raise ValueError(f"Invalid final coordinates shape/values: {tuple(x.shape)}")
    method = str(cluster_selection_method).strip().lower()
    epsilon, alpha_value = float(cluster_selection_epsilon), float(alpha)
    if method not in {"eom", "leaf"} or epsilon < 0.0 or alpha_value <= 0.0:
        raise ValueError("Invalid HDBSCAN selection method / epsilon / alpha")
    resolved = resolve_hdbscan_parameters(
        int(x.shape[0]),
        min_cluster_size=min_cluster_size,
        min_cluster_fraction=min_cluster_fraction,
        min_samples=min_samples,
        min_samples_fraction=min_samples_fraction,
    )
    distance = cosine_distance_matrix(x).numpy().astype(np.float64, copy=False)
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    if not np.isfinite(distance).all():
        raise ValueError("Final-layer cosine distance matrix contains NaN or Inf")
    hdbscan, implementation_version = require_hdbscan()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=resolved["min_cluster_size"],
        min_samples=resolved["min_samples"],
        metric="precomputed",
        algorithm="generic",
        alpha=alpha_value,
        cluster_selection_method=method,
        cluster_selection_epsilon=epsilon,
        allow_single_cluster=bool(allow_single_cluster),
        prediction_data=False,
    )
    raw_labels = np.asarray(clusterer.fit_predict(distance), dtype=np.int64)
    if raw_labels.shape != (int(x.shape[0]),):
        raise RuntimeError(f"HDBSCAN returned unexpected label shape: {raw_labels.shape}")
    raw_ids = sorted(int(c) for c in np.unique(raw_labels) if int(c) >= 0)
    order = sorted(raw_ids, key=lambda c: int(np.flatnonzero(raw_labels == c)[0]))
    canonical = np.full(raw_labels.shape, -1, dtype=np.int64)
    for new_id, raw_id in enumerate(order):
        canonical[raw_labels == raw_id] = new_id
    labels = torch.from_numpy(canonical.copy()).to(dtype=torch.long)
    n_tok = int(x.shape[0])
    hdbscan_noise_count = int(np.sum(canonical < 0))
    reassignment = {"reassigned_noise_count": 0, "fallback_single_cluster": False}
    if assign_all_tokens:
        labels, reassignment = assign_noise_to_nearest_cluster(x, labels)
    n_clusters = int(torch.unique(labels[labels >= 0]).numel())
    counts = [int((labels == c).sum()) for c in range(n_clusters)]
    assigned_count = int((labels >= 0).sum())
    noise_count = n_tok - assigned_count
    probabilities = np.asarray(
        getattr(clusterer, "probabilities_", np.zeros_like(raw_labels, dtype=np.float64)),
        dtype=np.float64,
    )
    outlier_scores = np.asarray(
        getattr(clusterer, "outlier_scores_", np.full_like(raw_labels, np.nan, dtype=np.float64)),
        dtype=np.float64,
    )
    raw_persistence = np.asarray(
        getattr(clusterer, "cluster_persistence_", np.empty(0, dtype=np.float64)),
        dtype=np.float64,
    )
    persistence_map = {
        raw_id: float(raw_persistence[i])
        for i, raw_id in enumerate(raw_ids)
        if i < raw_persistence.size
    }
    cluster_persistence = [persistence_map.get(raw_id) for raw_id in order]
    assigned_prob = probabilities[labels.numpy() >= 0]
    valid_persist = [v for v in cluster_persistence if v is not None]
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "cluster_counts": counts,
        "cluster_fractions_all_tokens": [c / max(n_tok, 1) for c in counts],
        "cluster_fractions_assigned_tokens": [c / max(assigned_count, 1) for c in counts],
        "minimum_cluster_fraction_all_tokens": (min(counts) / n_tok if counts else None),
        "assigned_token_count": assigned_count,
        "assigned_token_fraction": assigned_count / n_tok,
        "noise_count": noise_count,
        "noise_fraction": noise_count / n_tok,
        "hdbscan_noise_count": hdbscan_noise_count,
        "hdbscan_noise_fraction": hdbscan_noise_count / n_tok,
        "assign_all_tokens": bool(assign_all_tokens),
        "reassigned_noise_count": int(reassignment["reassigned_noise_count"]),
        "fallback_single_cluster": bool(reassignment["fallback_single_cluster"]),
        "membership_probabilities": probabilities.tolist(),
        "mean_assigned_membership_probability": (
            float(assigned_prob.mean()) if assigned_prob.size else None
        ),
        "outlier_scores": outlier_scores.tolist(),
        "cluster_persistence": cluster_persistence,
        "mean_cluster_persistence": (
            float(np.mean(valid_persist)) if valid_persist else None
        ),
        "implementation": "hdbscan.HDBSCAN",
        "implementation_version": implementation_version,
        "distance_metric": "precomputed_cosine_distance",
        "cluster_selection_method": method,
        "cluster_selection_epsilon": epsilon,
        "alpha": alpha_value,
        "allow_single_cluster": bool(allow_single_cluster),
        **resolved,
    }

@torch.no_grad()
def silhouette_score(
    coordinates: torch.Tensor,
    labels: torch.Tensor,
    pairwise_distances: Optional[torch.Tensor] = None,
) -> float:
    x = coordinates.detach().to(device="cpu", dtype=torch.float32)
    labels = labels.detach().to(device="cpu", dtype=torch.long)
    unique = torch.unique(labels, sorted=True)
    if x.shape[0] < 3 or unique.numel() < 2:
        return 0.0
    distances = (
        pairwise_distances
        if pairwise_distances is not None
        else cosine_distance_matrix(x)
    )
    scores = torch.zeros(x.shape[0], dtype=torch.float32, device="cpu")

    for cluster_id in unique.tolist():
        own = labels == cluster_id
        own_count = int(own.sum())
        if own_count <= 1:
            continue
        within = distances[own][:, own].sum(dim=1) / (own_count - 1)
        nearest_other = torch.full_like(within, float("inf"))
        for other_id in unique.tolist():
            if other_id == cluster_id:
                continue
            other = labels == other_id
            nearest_other = torch.minimum(
                nearest_other, distances[own][:, other].mean(dim=1)
            )
        denominator = torch.maximum(within, nearest_other).clamp_min(1e-12)
        scores[own] = (nearest_other - within) / denominator
    return float(scores.mean())

def require_umap():
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "umap-learn is required for Riemann UMAP: pip install umap-learn"
        ) from exc
    version = getattr(umap, "__version__", "unknown")
    return umap, version

def _l2_normalize_numpy(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)

def _haversine_latlon_to_unit_xyz(latlon: np.ndarray) -> np.ndarray:
    """Map UMAP haversine (lat, lon) embeddings to Cartesian points on S^2."""
    if latlon.ndim != 2 or latlon.shape[1] < 2:
        raise ValueError(
            f"Haversine embedding must be [n, 2]; got {tuple(latlon.shape)}"
        )
    latitude = latlon[:, 0]
    longitude = latlon[:, 1]
    x = np.sin(latitude) * np.cos(longitude)
    y = np.sin(latitude) * np.sin(longitude)
    z = np.cos(latitude)
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

@torch.no_grad()
def fit_cluster_umap(
    layer_states: Sequence[torch.Tensor],
    n_components: int,
    fit_tokens_per_layer: int,
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    spherical_output: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Riemann UMAP: unit-sphere inputs + cosine metric; optional haversine S^2 output."""
    umap, _ = require_umap()
    sampled: List[torch.Tensor] = []
    for state in layer_states:
        matrix = state.detach().to(device="cpu", dtype=torch.float32)
        index = _evenly_spaced_indices(matrix.shape[0], fit_tokens_per_layer)
        sampled.append(matrix[index])
    fit = torch.cat(sampled, dim=0).contiguous()
    if fit.ndim != 2 or min(fit.shape) < 2:
        raise ValueError(f"Riemann UMAP requires a 2-D matrix; got {tuple(fit.shape)}")
    if not torch.isfinite(fit).all():
        raise ValueError("Riemann UMAP input contains NaN or Inf")

    n_rows, n_features = fit.shape
    if float(fit.var()) <= 1e-12:
        raise ValueError("Riemann UMAP input has zero variance")

    use_spherical_output = (
        bool(n_components <= 3) if spherical_output is None else bool(spherical_output)
    )
    if use_spherical_output:
        rank = 2
        output_metric = "haversine"
    else:
        rank = max(2, min(int(n_components), n_rows - 1, n_features))
        output_metric = "euclidean"
    neighbors = max(2, min(int(n_neighbors), n_rows - 1))
    fit_numpy = _l2_normalize_numpy(fit.numpy())
    reducer = umap.UMAP(
        n_components=rank,
        n_neighbors=neighbors,
        min_dist=float(min_dist),
        metric=str(metric),
        output_metric=output_metric,
        random_state=int(seed),
        n_jobs=-1,
    )
    reducer.fit(fit_numpy)
    reducer._riemann_spherical_output = use_spherical_output
    return reducer, {
        "method": "riemann_umap",
        "seed": int(seed),
        "n_fit_rows": int(n_rows),
        "hidden_dimension": int(n_features),
        "n_components": int(3 if use_spherical_output else rank),
        "umap_embedding_components": int(rank),
        "n_neighbors": int(neighbors),
        "min_dist": float(min_dist),
        "metric": str(metric),
        "output_metric": output_metric,
        "input_normalization": "row_l2",
        "spherical_output": use_spherical_output,
    }

@torch.no_grad()
def transform_cluster_umap(reducer: Any, hidden: torch.Tensor) -> torch.Tensor:
    matrix = _l2_normalize_numpy(
        hidden.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    coordinates = np.asarray(reducer.transform(matrix), dtype=np.float32)
    if bool(getattr(reducer, "_riemann_spherical_output", False)):
        coordinates = _haversine_latlon_to_unit_xyz(coordinates)
    return torch.as_tensor(coordinates, dtype=torch.float32, device="cpu")

def adjusted_rand_index(labels_a: torch.Tensor, labels_b: torch.Tensor) -> float:
    a = labels_a.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    b = labels_b.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("ARI label vectors must have the same length")
    n = int(a.numel())
    if n < 2:
        return 1.0
    _, a_inverse = torch.unique(a, sorted=True, return_inverse=True)
    _, b_inverse = torch.unique(b, sorted=True, return_inverse=True)
    n_b = int(b_inverse.max()) + 1
    contingency = torch.bincount(
        a_inverse * n_b + b_inverse,
        minlength=(int(a_inverse.max()) + 1) * n_b,
    ).to(torch.float64)
    row_sums = torch.bincount(a_inverse).to(torch.float64)
    column_sums = torch.bincount(b_inverse).to(torch.float64)

    def comb2(values: torch.Tensor) -> torch.Tensor:
        return (values * (values - 1.0) / 2.0).sum()

    index = float(comb2(contingency))
    row_pairs = float(comb2(row_sums))
    column_pairs = float(comb2(column_sums))
    total_pairs = n * (n - 1) / 2.0
    expected = row_pairs * column_pairs / max(total_pairs, 1e-12)
    maximum = 0.5 * (row_pairs + column_pairs)
    denominator = maximum - expected
    if abs(denominator) <= 1e-12:
        return 1.0
    return float((index - expected) / denominator)

def serializable_partition(partition: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in partition.items()
        if key not in {"token_labels", "centers", "labels"}
    }

def project_coordinates_to_unit_sphere(
    coordinates: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    coords = coordinates.detach().to(device="cpu", dtype=torch.float32)
    padded = torch.zeros((coords.shape[0], 3), dtype=torch.float32, device="cpu")
    padded[:, : min(3, coords.shape[1])] = coords[:, :3]
    norms = padded.norm(dim=1, keepdim=True)
    sphere = padded / norms.clamp_min(eps)
    zero_rows = norms[:, 0] <= eps
    if zero_rows.any():
        sphere[zero_rows] = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device="cpu"
        )
    return sphere

def prepare_compatible_layers(
    captures: Dict[int, LayerHidden],
) -> Tuple[List[Tuple[int, torch.Tensor]], List[Dict[str, Any]]]:
    candidates: List[Tuple[int, torch.Tensor]] = []
    skipped: List[Dict[str, Any]] = []
    for layer_id in sorted(captures, key=lambda value: (int(value) != -1, int(value))):
        hidden = captures[layer_id].hidden
        if hidden.ndim != 2 or hidden.shape[0] < 4:
            skipped.append(
                {"layer": int(layer_id), "reason": "hidden state is not [tokens, features] with >=4 tokens"}
            )
        else:
            candidates.append((int(layer_id), hidden))
    if not candidates:
        return [], skipped
    dimensions: Dict[int, int] = {}
    for _, hidden in candidates:
        dimensions[int(hidden.shape[1])] = dimensions.get(int(hidden.shape[1]), 0) + 1
    feature_dimension = max(dimensions, key=lambda key: (dimensions[key], key))
    compatible: List[Tuple[int, torch.Tensor]] = []
    for layer_id, hidden in candidates:
        if int(hidden.shape[1]) == feature_dimension:
            compatible.append((layer_id, hidden))
        else:
            skipped.append(
                {
                    "layer": int(layer_id),
                    "reason": f"feature dimension {hidden.shape[1]} differs from modal dimension {feature_dimension}",
                }
            )
    return compatible, skipped

def select_first_middle_last_positions(
    layers: Sequence[Tuple[int, torch.Tensor]],
) -> List[Tuple[int, str]]:
    transformer_positions = [
        position
        for position, (layer_id, _) in enumerate(layers)
        if int(layer_id) != HiddenStateRecorder.EMBEDDING_LAYER_ID
    ]
    if not transformer_positions:
        return []
    first = transformer_positions[0]
    last = transformer_positions[-1]
    target_layer_id = (
        int(layers[first][0]) + int(layers[last][0])
    ) // 2
    middle = min(
        transformer_positions,
        key=lambda position: (
            abs(int(layers[position][0]) - target_layer_id),
            int(layers[position][0]),
        ),
    )
    selected: List[Tuple[int, str]] = []
    for position, role in ((first, "first"), (middle, "middle"), (last, "final")):
        if all(existing_position != position for existing_position, _ in selected):
            selected.append((position, role))
    return selected

@torch.no_grad()
def fixed_label_directional_metrics(
    coordinates: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Any]:
    all_x = normalize_rows(coordinates.detach().to(device="cpu", dtype=torch.float32))
    fixed_labels = labels.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if all_x.ndim != 2 or all_x.shape[0] != fixed_labels.numel():
        raise ValueError("Coordinates and fixed labels must contain the same tokens")
    assigned = fixed_labels >= 0
    x = all_x[assigned]
    assigned_labels = fixed_labels[assigned]
    unique = torch.unique(assigned_labels, sorted=True)
    n_total = int(all_x.shape[0])
    n_assigned = int(assigned.sum())
    n_noise = n_total - n_assigned
    if n_assigned == 0:
        return {
            "n_clusters": 0,
            "metric_token_policy": "exclude_final_hdbscan_noise",
            "assigned_token_count": 0,
            "assigned_token_fraction": 0.0,
            "noise_count": n_noise,
            "noise_fraction": 1.0 if n_total else 0.0,
            "fixed_label_cosine_silhouette": None,
            "cluster_counts": [],
            "cluster_fractions_all_tokens": [],
            "cluster_fractions_assigned_tokens": [],
        }

    cosine_silhouette = (
        silhouette_score(x, assigned_labels, cosine_distance_matrix(x))
        if unique.numel() >= 2
        else None
    )
    counts = [
        int((assigned_labels == int(cluster_id)).sum())
        for cluster_id in unique.tolist()
    ]
    return {
        "n_clusters": int(unique.numel()),
        "metric_token_policy": "exclude_final_hdbscan_noise",
        "assigned_token_count": n_assigned,
        "assigned_token_fraction": n_assigned / max(n_total, 1),
        "noise_count": n_noise,
        "noise_fraction": n_noise / max(n_total, 1),
        "fixed_label_cosine_silhouette": (
            None if cosine_silhouette is None else float(cosine_silhouette)
        ),
        "cluster_counts": counts,
        "cluster_fractions_all_tokens": [c / max(n_total, 1) for c in counts],
        "cluster_fractions_assigned_tokens": [c / max(n_assigned, 1) for c in counts],
    }

@torch.no_grad()
def analyze_hidden_state_umap(
    captures: Dict[int, LayerHidden],
    cluster_umap_components: int = 128,
    cluster_fit_tokens_per_layer: int = 256,
    spherical_cluster_space: str = "umap",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    hdbscan_min_cluster_size: int = 0,
    hdbscan_min_cluster_fraction: float = 0.05,
    hdbscan_min_samples: int = 0,
    hdbscan_min_samples_fraction: float = 0.01,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_alpha: float = 1.0,
    hdbscan_allow_single_cluster: bool = False,
    assign_all_tokens: bool = True,
    plot_tokens: int = 512,
    seed: int = 0,
) -> Dict[str, Any]:
    layers, skipped_layers = prepare_compatible_layers(captures)
    if len(layers) < 2:
        return {
            "available": False,
            "reason": "Cluster evolution requires compatible states from at least two observations",
            "skipped_layers": skipped_layers,
        }
    common_tokens = min(int(hidden.shape[0]) for _, hidden in layers)
    if common_tokens < 4:
        return {
            "available": False,
            "reason": "Cluster evolution requires at least four aligned tokens",
            "skipped_layers": skipped_layers,
        }
    layers = [(layer_id, hidden[:common_tokens]) for layer_id, hidden in layers]
    layer_hiddens = [hidden for _, hidden in layers]

    try:
        visual_reducer, visual_umap_fit = fit_cluster_umap(
            layer_hiddens,
            n_components=3,
            fit_tokens_per_layer=cluster_fit_tokens_per_layer,
            seed=seed,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            spherical_output=True,
        )
    except (RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "reason": f"Shared Riemann UMAP sphere visualization failed: {exc}",
        }

    visual_umap_coordinates = [
        transform_cluster_umap(visual_reducer, hidden) for hidden in layer_hiddens
    ]

    cluster_umap_fit: Optional[Dict[str, Any]] = None
    if spherical_cluster_space == "hidden":
        clustering_coordinates = [
            normalize_rows(hidden.to(dtype=torch.float32)) for hidden in layer_hiddens
        ]
        clustering_space_name = "original_row_l2_normalized_hidden_state"
    elif spherical_cluster_space == "umap":
        cluster_components = max(2, int(cluster_umap_components))
        if cluster_components <= 3:
            cluster_coordinates_raw = visual_umap_coordinates
            cluster_umap_fit = {**visual_umap_fit, "reused_visual_sphere_umap": True}
        else:
            try:
                cluster_reducer, cluster_umap_fit = fit_cluster_umap(
                    layer_hiddens,
                    n_components=cluster_components,
                    fit_tokens_per_layer=cluster_fit_tokens_per_layer,
                    seed=int(seed) + 1,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                    spherical_output=False,
                )
            except (RuntimeError, ValueError) as exc:
                return {
                    "available": False,
                    "reason": f"Shared Riemann UMAP clustering failed: {exc}",
                }
            cluster_coordinates_raw = [
                transform_cluster_umap(cluster_reducer, hidden)
                for hidden in layer_hiddens
            ]
            cluster_umap_fit = {
                **cluster_umap_fit,
                "reused_visual_sphere_umap": False,
            }
        clustering_coordinates = [
            normalize_rows(coordinates) for coordinates in cluster_coordinates_raw
        ]
        clustering_space_name = "row_l2_normalized_shared_riemann_umap_coordinates"
    else:
        raise ValueError("spherical_cluster_space must be 'umap' or 'hidden'")

    transformer_positions = [
        position
        for position, (layer_id, _) in enumerate(layers)
        if int(layer_id) != HiddenStateRecorder.EMBEDDING_LAYER_ID
    ]
    if len(transformer_positions) < 3:
        return {
            "available": False,
            "reason": (
                "Final-layer tracking requires at least three captured "
                "Transformer layers for first/middle/final snapshots"
            ),
            "skipped_layers": skipped_layers,
        }
    final_position = transformer_positions[-1]
    final_partition_internal = fit_final_layer_hdbscan(
        clustering_coordinates[final_position],
        min_cluster_size=hdbscan_min_cluster_size,
        min_cluster_fraction=hdbscan_min_cluster_fraction,
        min_samples=hdbscan_min_samples,
        min_samples_fraction=hdbscan_min_samples_fraction,
        cluster_selection_method=hdbscan_cluster_selection_method,
        cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        alpha=hdbscan_alpha,
        allow_single_cluster=hdbscan_allow_single_cluster,
        assign_all_tokens=assign_all_tokens,
    )
    final_labels = final_partition_internal["labels"]
    inferred_final_k = int(final_partition_internal["n_clusters"])

    layer_trace_internal: List[Dict[str, Any]] = []
    for position, ((layer_id, _), directional_coordinates) in enumerate(
        zip(layers, clustering_coordinates)
    ):
        metrics = fixed_label_directional_metrics(
            directional_coordinates, final_labels
        )
        layer_trace_internal.append(
            {
                "trace_index": int(position),
                "layer": int(layer_id),
                "stage": (
                    "embedding" if int(layer_id) == -1 else f"layer_{layer_id}"
                ),
                "n_clusters": inferred_final_k,
                "label_source_trace_index": int(final_position),
                "label_source_layer": int(layers[final_position][0]),
                "labels_reused_without_reclustering": True,
                **metrics,
            }
        )

    visual_token_index = torch.arange(
        common_tokens, dtype=torch.long, device="cpu"
    )

    snapshot_positions = select_first_middle_last_positions(layers)
    if not snapshot_positions:
        return {
            "available": False,
            "reason": "No Transformer layers are available for sphere snapshots",
            "final_layer_clustering_completed": True,
            "layer_trace": layer_trace_internal,
        }
    plot_index = _evenly_spaced_indices(len(visual_token_index), plot_tokens)
    snapshots: List[Dict[str, Any]] = []
    for position, snapshot_role in snapshot_positions:
        coordinates = visual_umap_coordinates[position][plot_index]
        sphere = project_coordinates_to_unit_sphere(coordinates)
        original_token_index = visual_token_index[plot_index]
        trace_row = layer_trace_internal[position]
        snapshots.append(
            {
                "trace_index": int(position),
                "layer": int(layers[position][0]),
                "stage": trace_row["stage"],
                "snapshot_role": snapshot_role,
                "n_clusters": inferred_final_k,
                "fixed_label_cosine_silhouette": trace_row[
                    "fixed_label_cosine_silhouette"
                ],
                "token_index": original_token_index.tolist(),
                "final_cluster_label": final_labels[
                    original_token_index
                ].tolist(),
                "axis1": coordinates[:, 0].tolist(),
                "axis2": (
                    coordinates[:, 1].tolist()
                    if coordinates.shape[1] > 1
                    else [0.0] * coordinates.shape[0]
                ),
                "axis3": (
                    coordinates[:, 2].tolist()
                    if coordinates.shape[1] > 2
                    else [0.0] * coordinates.shape[0]
                ),
                "sphere_x": sphere[:, 0].tolist(),
                "sphere_y": sphere[:, 1].tolist(),
                "sphere_z": sphere[:, 2].tolist(),
            }
        )

    first_transformer_trace = layer_trace_internal[transformer_positions[0]]
    final_trace = layer_trace_internal[final_position]
    middle_position = next(
        position
        for position, role in snapshot_positions
        if role == "middle"
    )
    middle_trace = layer_trace_internal[middle_position]

    def optional_difference(
        later_value: Optional[float], earlier_value: Optional[float]
    ) -> Optional[float]:
        if later_value is None or earlier_value is None:
            return None
        return float(later_value) - float(earlier_value)

    tracking_summary = {
        "first_transformer_layer": int(layers[transformer_positions[0]][0]),
        "middle_transformer_layer": int(layers[middle_position][0]),
        "final_transformer_layer": int(layers[final_position][0]),
        "inferred_final_n_clusters": inferred_final_k,
        "final_noise_fraction": float(final_partition_internal["noise_fraction"]),
        "first_fixed_label_cosine_silhouette": first_transformer_trace[
            "fixed_label_cosine_silhouette"
        ],
        "middle_fixed_label_cosine_silhouette": middle_trace[
            "fixed_label_cosine_silhouette"
        ],
        "final_fixed_label_cosine_silhouette": final_trace[
            "fixed_label_cosine_silhouette"
        ],
        "final_minus_first_cosine_silhouette": optional_difference(
            final_trace["fixed_label_cosine_silhouette"],
            first_transformer_trace["fixed_label_cosine_silhouette"],
        ),
    }
    final_layer_clustering = {
        "reference_trace_index": int(final_position),
        "reference_layer": int(layers[final_position][0]),
        "reference_stage": final_trace["stage"],
        "n_clusters": inferred_final_k,
        "status": (
            "multi_cluster"
            if inferred_final_k > 1
            else "single_cluster"
            if inferred_final_k == 1
            else "no_density_cluster"
        ),
        "is_density_supported_multicluster": bool(inferred_final_k > 1),
        "noise_label": None if bool(assign_all_tokens) else -1,
        "token_labels": final_labels.tolist(),
        **serializable_partition(final_partition_internal),
        "fixed_label_cosine_silhouette": final_trace.get(
            "fixed_label_cosine_silhouette"
        ),
    }
    warnings = [
        f"HDBSCAN on {final_trace['stage']} ({clustering_space_name}); earlier layers reuse labels.",
        (
            "Noise reassigned to nearest cluster."
            if assign_all_tokens
            else "Label -1 retained as noise and excluded from cosine silhouette."
        ),
    ]
    return {
        "available": True,
        "analysis": "final_layer_anchored_hdbscan_tracking",
        "interpretation_scope": "density_inferred_final_partition_with_retrospective_tracking",
        "n_layers": len(layers),
        "n_aligned_tokens": common_tokens,
        "captured_layers": [int(layer_id) for layer_id, _ in layers],
        "embedding_baseline_available": any(layer_id == -1 for layer_id, _ in layers),
        "clustering_space": clustering_space_name,
        "cluster_space_mode": spherical_cluster_space,
        "cluster_algorithm": "hdbscan",
        "distance_metric": "precomputed_cosine_distance",
        "cluster_umap_fit": cluster_umap_fit or visual_umap_fit,
        "visualization_umap_fit": visual_umap_fit,
        "visualization_method": "riemann_umap_sphere",
        "tracking_config": {
            "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
            "hdbscan_min_cluster_fraction": float(hdbscan_min_cluster_fraction),
            "hdbscan_min_samples": int(hdbscan_min_samples),
            "hdbscan_min_samples_fraction": float(hdbscan_min_samples_fraction),
            "hdbscan_cluster_selection_method": str(hdbscan_cluster_selection_method),
            "hdbscan_cluster_selection_epsilon": float(hdbscan_cluster_selection_epsilon),
            "hdbscan_alpha": float(hdbscan_alpha),
            "hdbscan_allow_single_cluster": bool(hdbscan_allow_single_cluster),
            "assign_all_tokens": bool(assign_all_tokens),
            "cluster_space": spherical_cluster_space,
        },
        "layer_trace": layer_trace_internal,
        "tracking_summary": tracking_summary,
        "final_layer_clustering": final_layer_clustering,
        "snapshots_sphere": snapshots,
        "skipped_layers": skipped_layers,
        "warnings": warnings,
    }
