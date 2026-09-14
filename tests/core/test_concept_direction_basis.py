"""The `concept_basis` selector on `concept_direction` (#420).

Four named values, no default: `embed` builds from token-group embedding rows, `store` aggregates
latent-example rows, and `jlens_paper` / `jlens_norm_aware` build per-token J-lens direction rows
through the read path. An absent or unknown basis is refused by name, and a basis whose inputs are
not on the batch is refused naming what is missing -- the previous silent fallback from missing
store rows to embeddings is removed, because it returned a plausible direction for a basis nobody
asked for.

The J-lens cases run on CPU against a synthetic anisotropic lens (monkeypatched seam, the same
pattern as `test_jlens_read_ops.py`). A real lens is unreachable in offline CI, so the
non-collinearity case uses an anisotropic `J` plus a non-uniform norm scale: the exact property
the test exists to pin is that the two bases differ by more than a scalar multiple.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.bundled.concept import concept_ops
from interpretune.analysis.optools import JLensArtifact

D, VOCAB = 8, 24
LAYERS = (0, 4, 8)


class _Tok:
    def __init__(self):
        self._vocab = {f"tok{i}": i for i in range(VOCAB)}

    def get_vocab(self):
        return dict(self._vocab)


class _RMSNorm(torch.nn.Module):  # class name drives kind detection
    def __init__(self, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(scale)

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


def _module(scale=None):
    torch.manual_seed(0)
    inner = type("Inner", (), {})()
    inner.norm = _RMSNorm(torch.ones(D) if scale is None else scale)
    model = type("Model", (), {})()
    model.config = type("Cfg", (), {"model_type": "llama"})()
    model.lm_head = type("Head", (), {})()
    model.lm_head.weight = torch.randn(VOCAB, D)
    model.model = inner
    model.tokenizer = _Tok()
    model.embed = type("Embed", (), {"W_E": torch.randn(VOCAB, D)})()
    module = type("Module", (), {})()
    module.model = model
    return module


@pytest.fixture
def anisotropic_lens(monkeypatch):
    """A lens whose J is anisotropic, so the folded and unfolded bases cannot coincide."""

    def _resolve(module, analysis_batch, kwargs, *, default_percentile=0.85):
        del module, default_percentile
        layer = int(kwargs.get("jlens_layer", analysis_batch.get("jlens_layer") or 4))
        if layer not in LAYERS:
            # The seam's own refusal, propagated rather than swallowed: the fitted-set rule is
            # pinned in test_jlens_read_ops.py, so this fake only carries the branch shape.
            raise ValueError(f"which does not include {layer}")
        return (
            torch.diag(torch.arange(1.0, D + 1)),
            layer,
            JLensArtifact(
                j_by_layer={ell: torch.diag(torch.arange(1.0, D + 1)) for ell in LAYERS},
                source_layers=list(LAYERS),
                d_model=D,
                repo_id="synthetic",
                path="synthetic/jlens/c/synthetic_jacobian_lens.pt",
                hf_model_name="org/synthetic",
                provenance={},
            ),
        )

    monkeypatch.setattr(concept_ops, "resolve_jlens_layer", _resolve)


def _groups():
    return {"concept_group_a": ["tok3"], "concept_group_b": ["tok7"]}


class TestBasisIsRequired:
    def test_absent_basis_is_refused_naming_the_values(self):
        with pytest.raises(ValueError, match="concept_basis.*embed.*store.*jlens_paper.*jlens_norm_aware"):
            concept_ops.concept_direction_impl(_module(), AnalysisBatch(**_groups()), batch=None, batch_idx=0)

    def test_unknown_basis_is_refused(self):
        with pytest.raises(ValueError, match="not a basis"):
            concept_ops.concept_direction_impl(
                _module(), AnalysisBatch(**_groups(), concept_basis="logit"), batch=None, batch_idx=0
            )


class TestNoSilentFallback:
    def test_store_without_rows_is_refused_rather_than_embedded(self):
        """The removed behavior, pinned as refused: rows missing + token groups present."""
        with pytest.raises(ValueError, match="silent fallback"):
            concept_ops.concept_direction_impl(
                _module(), AnalysisBatch(**_groups(), concept_basis="store"), batch=None, batch_idx=0
            )


class TestBasesAreRecorded:
    def test_embed_records_its_basis(self):
        out = concept_ops.concept_direction_impl(
            _module(), AnalysisBatch(**_groups(), concept_basis="embed"), batch=None, batch_idx=0
        )
        assert out.concept_basis == "embed"
        assert out.concept_group_a_token_ids == [3]
        assert out.concept_group_b_token_ids == [7]

    def test_store_records_its_basis(self):
        out = concept_ops.concept_direction_impl(
            _module(),
            AnalysisBatch(
                concept_latent_state=torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
                concept_group_id=torch.tensor([0, 1]),
                concept_basis="store",
            ),
            batch=None,
            batch_idx=0,
        )
        assert out.concept_basis == "store"

    def test_jlens_paths_record_their_bases(self, anisotropic_lens):
        for basis in ("jlens_paper", "jlens_norm_aware"):
            out = concept_ops.concept_direction_impl(
                _module(),
                AnalysisBatch(**_groups(), concept_basis=basis, jlens_layer=4),
                batch=None,
                batch_idx=0,
            )
            assert out.concept_basis == basis
            assert out.jlens_layer == 4


class TestJLensBasesDiffer:
    def test_folded_and_unfolded_directions_are_not_scalar_multiples(self, anisotropic_lens):
        """The executable form of the alignment document's item 5: a refactor collapsing the two bases into one
        value with a coefficient fails loudly here."""
        scale = 1.0 + 0.1 * torch.arange(D)  # non-uniform scale: the premise the test needs
        assert not torch.allclose(scale, scale.mean() * torch.ones(D), atol=1e-6)
        paper = concept_ops.concept_direction_impl(
            _module(scale),
            AnalysisBatch(**_groups(), concept_basis="jlens_paper", jlens_layer=4),
            batch=None,
            batch_idx=0,
        ).concept_direction
        aware = concept_ops.concept_direction_impl(
            _module(scale),
            AnalysisBatch(**_groups(), concept_basis="jlens_norm_aware", jlens_layer=4),
            batch=None,
            batch_idx=0,
        ).concept_direction
        cosine = torch.dot(paper, aware) / torch.linalg.vector_norm(paper) / torch.linalg.vector_norm(aware)
        assert cosine < 1 - 1e-6, "bases coincide: the fold choice is not being honored"

    def test_jlens_direction_matches_the_independent_construction(self, anisotropic_lens):
        """`rows @ J` recomputed in-test, so the op cannot agree with itself by construction."""
        module = _module()
        out = concept_ops.concept_direction_impl(
            module, AnalysisBatch(**_groups(), concept_basis="jlens_paper", jlens_layer=4), batch=None, batch_idx=0
        )
        w_u = module.model.lm_head.weight.float()
        j = torch.diag(torch.arange(1.0, D + 1))
        expected = (w_u[3] @ j - w_u[7] @ j).float()
        expected = expected / torch.linalg.vector_norm(expected)
        torch.testing.assert_close(out.concept_direction, expected, atol=1e-5, rtol=1e-5)


class TestJLensRefusals:
    def test_unfitted_layer_refusal_propagates(self, anisotropic_lens):
        with pytest.raises(ValueError, match="does not include 5"):
            concept_ops.concept_direction_impl(
                _module(),
                AnalysisBatch(**_groups(), concept_basis="jlens_paper", jlens_layer=5),
                batch=None,
                batch_idx=0,
            )
