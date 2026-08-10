from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch


ProjectionMethod = Literal["umap", "pca"]
UmapBackendPreference = Literal["cpu", "gpu", "auto"]


@dataclass(frozen=True)
class ProjectionResult:
    method: str
    backend: str
    coordinates: np.ndarray
    metadata: dict[str, Any]


def _normalize_coordinate_array(values: Any, *, n_components: int) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    coordinates = np.asarray(values, dtype=np.float32)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)
    if coordinates.shape[1] < n_components:
        padding = np.zeros((coordinates.shape[0], n_components - coordinates.shape[1]), dtype=np.float32)
        coordinates = np.concatenate([coordinates, padding], axis=1)
    return coordinates[:, :n_components]


def _pca_projection(
    matrix: torch.Tensor,
    *,
    n_components: int,
) -> np.ndarray:
    row_count, hidden_size = matrix.shape
    if row_count <= 1 or hidden_size == 0:
        return np.zeros((row_count, n_components), dtype=np.float32)

    centered = matrix - matrix.mean(dim=0, keepdim=True)
    component_rank = min(max(n_components + 1, 2), row_count, hidden_size)
    try:
        _, _, right_vectors = torch.pca_lowrank(centered, q=component_rank, center=False)
        coordinates = centered @ right_vectors[:, : min(n_components, right_vectors.shape[1])]
    except Exception:
        coordinates = centered[:, : min(n_components, hidden_size)]
    return _normalize_coordinate_array(coordinates, n_components=n_components)


def project_embeddings(
    values: Any,
    *,
    method: ProjectionMethod = "umap",
    n_components: int = 2,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 17,
    umap_init: str = "random",
    umap_backend_preference: UmapBackendPreference = "cpu",
) -> ProjectionResult:
    matrix = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 2:
        raise ValueError(f"project_embeddings expects a rank-2 matrix, received shape {tuple(matrix.shape)}")

    row_count = int(matrix.shape[0])
    if row_count == 0:
        raise ValueError("project_embeddings requires at least one row")

    if method == "pca":
        coordinates = _pca_projection(matrix, n_components=n_components)
        return ProjectionResult(
            method="pca",
            backend="torch_pca",
            coordinates=coordinates,
            metadata={
                "row_count": row_count,
                "n_components": int(n_components),
            },
        )

    if method != "umap":
        raise ValueError(f"Unsupported projection method: {method}")

    if row_count <= 2:
        coordinates = _pca_projection(matrix, n_components=n_components)
        return ProjectionResult(
            method="umap",
            backend="torch_pca_fallback",
            coordinates=coordinates,
            metadata={
                "row_count": row_count,
                "n_components": int(n_components),
                "fallback_reason": "row_count_too_small_for_umap",
            },
        )

    estimator_specs_by_preference: dict[UmapBackendPreference, tuple[tuple[str, str, str], ...]] = {
        "cpu": (
            ("umap", "UMAP", "umap_learn"),
            ("cuml.manifold", "UMAP", "rapids_cuml.manifold"),
            ("cuml", "UMAP", "rapids_cuml"),
        ),
        "gpu": (
            ("cuml.manifold", "UMAP", "rapids_cuml.manifold"),
            ("cuml", "UMAP", "rapids_cuml"),
            ("umap", "UMAP", "umap_learn"),
        ),
        "auto": (
            ("umap", "UMAP", "umap_learn"),
            ("cuml.manifold", "UMAP", "rapids_cuml.manifold"),
            ("cuml", "UMAP", "rapids_cuml"),
        ),
    }
    if umap_backend_preference not in estimator_specs_by_preference:
        raise ValueError(f"Unsupported umap backend preference: {umap_backend_preference}")

    estimator_specs = estimator_specs_by_preference[umap_backend_preference]
    fit_errors: list[str] = []
    effective_neighbors = max(2, min(int(n_neighbors), row_count - 1))
    matrix_array = matrix.numpy()

    for module_name, attr_name, backend_name in estimator_specs:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Tensorflow not installed; ParametricUMAP will be unavailable",
                )
                module = importlib.import_module(module_name)
            estimator_cls = getattr(module, attr_name)
        except Exception:
            continue

        try:
            estimator = estimator_cls(
                n_components=n_components,
                n_neighbors=effective_neighbors,
                min_dist=float(min_dist),
                metric=metric,
                random_state=int(random_state),
                init=umap_init,
            )
            coordinates = estimator.fit_transform(matrix_array)
            return ProjectionResult(
                method="umap",
                backend=backend_name,
                coordinates=_normalize_coordinate_array(coordinates, n_components=n_components),
                metadata={
                    "row_count": row_count,
                    "n_components": int(n_components),
                    "n_neighbors": int(effective_neighbors),
                    "min_dist": float(min_dist),
                    "metric": metric,
                    "random_state": int(random_state),
                    "init": umap_init,
                    "umap_backend_preference": umap_backend_preference,
                },
            )
        except Exception as exc:
            fit_errors.append(f"{backend_name}: {exc}")

    coordinates = _pca_projection(matrix, n_components=n_components)
    return ProjectionResult(
        method="umap",
        backend="torch_pca_fallback",
        coordinates=coordinates,
        metadata={
            "row_count": row_count,
            "n_components": int(n_components),
            "n_neighbors": int(effective_neighbors),
            "min_dist": float(min_dist),
            "metric": metric,
            "random_state": int(random_state),
            "init": umap_init,
            "umap_backend_preference": umap_backend_preference,
            "fallback_errors": fit_errors,
        },
    )


__all__ = ["ProjectionMethod", "ProjectionResult", "UmapBackendPreference", "project_embeddings"]
