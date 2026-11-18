"""A UMAP projection is UMAP, or a refusal naming why not; a PCA stand-in is labelled PCA."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
import torch

from interpretune.utils import latent_state_projection
from interpretune.utils.latent_state_projection import UmapUnavailableError, project_embeddings

_UMAP_MODULES = ("umap", "cuml", "cuml.manifold")


@pytest.fixture
def no_umap_backend(monkeypatch):
    """Every UMAP backend fails to import, whatever this environment has installed."""
    real_import = importlib.import_module

    def _import(name, *args, **kwargs):
        if name in _UMAP_MODULES:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(latent_state_projection.importlib, "import_module", _import)


@pytest.fixture
def fake_umap_learn(monkeypatch):
    """A stand-in ``umap`` module whose estimator returns a known projection."""

    class _UMAP:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components

        def fit_transform(self, values):
            return np.full((values.shape[0], self.n_components), 7.0, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=_UMAP))


def _rows(n: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, 5)


class TestUmapRequested:
    def test_no_backend_is_refused_by_name(self, no_umap_backend) -> None:
        with pytest.raises(UmapUnavailableError) as excinfo:
            project_embeddings(_rows(), method="umap")
        message = str(excinfo.value)
        assert "no UMAP backend could fit" in message
        assert "umap_learn: not importable" in message
        assert "rapids_cuml: not importable" in message
        assert "fallback='pca'" in message

    def test_opted_in_fallback_is_labelled_pca(self, no_umap_backend) -> None:
        result = project_embeddings(_rows(), method="umap", fallback="pca")
        assert result.method == "pca"
        assert result.backend == "torch_pca"
        assert result.coordinates.shape == (6, 2)
        assert result.metadata["requested_method"] == "umap"
        assert result.metadata["fallback_reason"] == "no UMAP backend could fit"
        assert any(err.startswith("umap_learn: not importable") for err in result.metadata["fallback_errors"])

    def test_too_few_rows_follows_the_same_rule(self, fake_umap_learn) -> None:
        with pytest.raises(UmapUnavailableError, match="at least 3 rows, received 2"):
            project_embeddings(_rows(2), method="umap")
        result = project_embeddings(_rows(2), method="umap", fallback="pca")
        assert result.method == "pca"
        assert result.metadata["fallback_reason"] == "UMAP needs at least 3 rows, received 2"

    def test_an_importable_backend_is_used(self, fake_umap_learn) -> None:
        # positive control: the refusal above is about the missing backend, not about UMAP always failing
        result = project_embeddings(_rows(), method="umap")
        assert (result.method, result.backend) == ("umap", "umap_learn")
        assert np.all(result.coordinates == 7.0)

    def test_an_unknown_fallback_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unsupported projection fallback: 'umap'"):
            project_embeddings(_rows(), method="umap", fallback="umap")  # type: ignore[arg-type]


def test_real_umap_learn_projects_when_installed() -> None:
    pytest.importorskip("umap")
    result = project_embeddings(_rows(12), method="umap", n_neighbors=4)
    assert (result.method, result.backend) == ("umap", "umap_learn")
    assert result.coordinates.shape == (12, 2)
