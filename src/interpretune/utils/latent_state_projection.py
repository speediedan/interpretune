"""Low-dimensional projections of embedding matrices for latent-state visualization."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch


ProjectionMethod = Literal["umap", "pca"]
UmapBackendPreference = Literal["cpu", "gpu", "auto"]
ProjectionFallback = Literal["refuse", "pca"]


class UmapUnavailableError(RuntimeError):
    """A UMAP projection was requested and no UMAP backend could produce one.

    A PCA projection is never returned in its place under the UMAP name: the plot would look plausible and say
    something different. A caller who accepts PCA asks for ``method="pca"`` or passes ``fallback="pca"``, and the
    result is then labelled ``pca``.
    """


@dataclass(frozen=True)
class ProjectionResult:
    """Outcome of projecting an embedding matrix to a low-dimensional coordinate array."""

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
    fallback: ProjectionFallback = "refuse",
) -> ProjectionResult:
    """Project an embedding matrix to ``n_components`` dimensions for visualization.

    ``method="umap"`` tries the backends ``umap_backend_preference`` orders (``umap-learn``, from the ``examples``
    extra, and RAPIDS ``cuml``). When none can fit, or there are too few rows for UMAP, it raises
    :class:`UmapUnavailableError` naming what was tried, unless ``fallback="pca"``: then the result is a PCA
    projection whose ``method`` is ``"pca"`` and whose metadata records the UMAP request and why it was not met.
    """
    if fallback not in ("refuse", "pca"):
        raise ValueError(f"Unsupported projection fallback: {fallback!r} (expected 'refuse' or 'pca')")
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
        return _umap_not_met(
            matrix,
            n_components=n_components,
            fallback=fallback,
            reason=f"UMAP needs at least 3 rows, received {row_count}",
            errors=[],
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
        except Exception as exc:
            fit_errors.append(f"{backend_name}: not importable ({type(exc).__name__}: {exc})")
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

    return _umap_not_met(
        matrix,
        n_components=n_components,
        fallback=fallback,
        reason="no UMAP backend could fit",
        errors=fit_errors,
    )


def _umap_not_met(
    matrix: torch.Tensor,
    *,
    n_components: int,
    fallback: ProjectionFallback,
    reason: str,
    errors: list[str],
) -> ProjectionResult:
    if fallback != "pca":
        tried = "; ".join(errors) if errors else "none"
        raise UmapUnavailableError(
            f"method='umap' could not run: {reason} (backends tried: {tried}). Install umap-learn "
            "(`interpretune[examples]`), ask for method='pca', or pass fallback='pca' to accept a PCA projection "
            "labelled as PCA."
        )
    return ProjectionResult(
        method="pca",
        backend="torch_pca",
        coordinates=_pca_projection(matrix, n_components=n_components),
        metadata={
            "row_count": int(matrix.shape[0]),
            "n_components": int(n_components),
            "requested_method": "umap",
            "fallback_reason": reason,
            "fallback_errors": errors,
        },
    )


__all__ = [
    "ProjectionFallback",
    "ProjectionMethod",
    "ProjectionResult",
    "UmapBackendPreference",
    "UmapUnavailableError",
    "project_embeddings",
]
