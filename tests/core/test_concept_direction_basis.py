"""The `concept_basis` selector on `concept_direction` (#420).

Four named values, no default: `embed` builds from token-group embedding rows, `store` aggregates
latent-example rows, and `jlens_paper` / `jlens_norm_aware` build per-token J-lens direction rows
through the read path. An absent or unknown basis is refused by name, and a basis whose inputs are
not on the batch is refused naming what is missing -- the previous silent fallback from missing
store rows to embeddings is removed, because it returned a plausible direction for a basis nobody
asked for.

The J-lens guard cases run on CPU against synthetic lenses (monkeypatched seam, the same
pattern as `test_jlens_read_ops.py`): the guard tests the construction, so its verdict must not
depend on an artifact's contents. A second test measures non-collinearity on the real gpt2-small
lens, warmed into the offline cache by tests/hf_warm_manifest.yaml.
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


class _LayerNorm(torch.nn.Module):  # class name drives kind detection
    def __init__(self, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(scale)

    def forward(self, x):
        x = x.float()
        return (
            (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-6) * self.weight
        )


def _module(scale=None, norm_cls=_RMSNorm):
    torch.manual_seed(0)
    inner = type("Inner", (), {})()
    inner.norm = norm_cls(torch.ones(D) if scale is None else scale)
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
def seam_lens(monkeypatch):
    """The seam's layer rule with a swappable J: tests set ``holder['J']`` to the matrix the case needs.

    The fitted-set refusal branch is the seam's own, propagated rather than swallowed (the rule is pinned in
    test_jlens_read_ops.py, so this fake only carries the branch shape).
    """
    holder = {}

    def _resolve(module, analysis_batch, kwargs, *, default_percentile=0.85):
        del module, default_percentile
        layer = int(kwargs.get("jlens_layer", analysis_batch.get("jlens_layer") or 4))
        if layer not in LAYERS:
            raise ValueError(f"which does not include {layer}")
        return (
            holder["J"],
            layer,
            JLensArtifact(
                j_by_layer={ell: holder["J"] for ell in LAYERS},
                source_layers=list(LAYERS),
                d_model=D,
                repo_id="synthetic",
                path="synthetic/jlens/c/synthetic_jacobian_lens.pt",
                hf_model_name="org/synthetic",
                provenance={},
            ),
        )

    monkeypatch.setattr(concept_ops, "resolve_jlens_layer", _resolve)
    return holder


def _j_mats():
    """The guard's two lenses: degenerate identity plus a random full-rank J (premise-guarded)."""
    torch.manual_seed(7)
    full_rank = torch.randn(D, D)
    assert torch.linalg.matrix_rank(full_rank) == D, "guard fixture is rank-deficient"
    return {"identity": torch.eye(D), "full_rank": full_rank}


def _concept_vectors(module_fn, **batch_extra):
    """Both bases' concept vectors plus their recorded names, bases-checked-first by the caller."""
    outs = {}
    direction_mode = batch_extra.pop("concept_direction_mode", "mean_difference")
    for basis in ("jlens_paper", "jlens_norm_aware"):
        outs[basis] = concept_ops.concept_direction_impl(
            module_fn(),
            AnalysisBatch(
                **_groups(),
                concept_basis=basis,
                jlens_layer=4,
                concept_direction_mode=direction_mode,
                **batch_extra,
            ),
            batch=None,
            batch_idx=0,
        )
    assert outs["jlens_paper"].concept_basis == "jlens_paper"
    assert outs["jlens_norm_aware"].concept_basis == "jlens_norm_aware"
    return outs["jlens_paper"].concept_direction, outs["jlens_norm_aware"].concept_direction


def _one_minus_abs_cos(a, b):
    return 1 - abs(float(torch.dot(a, b) / torch.linalg.vector_norm(a) / torch.linalg.vector_norm(b)))


def _zero_meaned_module(scale):
    """The seamed module with the concept rows centered: centering then has nothing to move."""
    module = _module(scale, norm_cls=_LayerNorm)
    rows = module.model.lm_head.weight.data[[3, 7]]
    module.model.lm_head.weight.data[[3, 7]] = rows - rows.mean(dim=-1, keepdim=True)
    return module


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

    def test_jlens_paths_record_their_bases(self, seam_lens):
        seam_lens["J"] = torch.eye(D)
        for basis in ("jlens_paper", "jlens_norm_aware"):
            out = concept_ops.concept_direction_impl(
                _module(),
                AnalysisBatch(**_groups(), concept_basis=basis, jlens_layer=4),
                batch=None,
                batch_idx=0,
            )
            assert out.concept_basis == basis
            assert out.jlens_layer == 4


class TestSyntheticGuard:
    """The construction guard: collapsing the two bases into one value with a coefficient fails here.

    Each case goes through the op (not around it) so the recorded-basis check runs first: a result
    that cannot name its basis is not comparable, and the comparison below would be meaningless.
    """

    def test_rmsnorm_nonuniform_scale_is_not_collinear(self, seam_lens):
        """Anisotropic scale separates the bases, with the lens applied and without it.

        Margin 1e-3 against measured gaps of 6.8e-02 (J=I) and 1.5e-01 (full-rank): the margin is set by the fixture's
        own anisotropy, roughly fifty times below what it produces.
        """
        scale = 0.5 + 1.5 * torch.rand(D)  # non-uniform: the premise this case needs
        assert not torch.allclose(scale, scale.mean() * torch.ones(D), atol=1e-6)
        for name, j in _j_mats().items():
            seam_lens["J"] = j
            paper, aware = _concept_vectors(lambda: _module(scale))
            gap = _one_minus_abs_cos(paper, aware)
            assert gap > 1e-3, f"RMSNorm non-uniform scale coincides at J={name}: gap {gap:.2e}"

    def test_rmsnorm_uniform_scale_is_collinear(self, seam_lens):
        """Positive control: with a uniform scale the two constructions agree to float tolerance.

        Without it, the case above is satisfied by any bug that perturbs one vector.
        """
        for name, j in _j_mats().items():
            seam_lens["J"] = j
            paper, aware = _concept_vectors(lambda: _module(torch.full((D,), 1.7)))
            assert _one_minus_abs_cos(paper, aware) < 1e-6, f"uniform scale diverges at J={name}"

    def test_layernorm_uniform_scale_is_not_collinear(self, seam_lens):
        """Centering moves the direction even when the scale does not: the axis a uniform-scale
        RMSNorm fixture cannot reach, and the one that matters for gpt2.

        Single-group mode, deliberately: centering moves a single row by its mean, while the
        difference of two random rows is already near-zero-mean and would exercise the control
        below instead of this case. The direction mode is orthogonal to the fold construction.
        """
        seam_lens["J"] = torch.eye(D)
        paper, aware = _concept_vectors(
            lambda: _module(torch.full((D,), 1.7), norm_cls=_LayerNorm), concept_direction_mode="single_group"
        )
        assert _one_minus_abs_cos(paper, aware) > 1e-3

    def test_layernorm_zero_mean_row_coincides(self, seam_lens):
        """Control: a concept row that is already zero-mean is untouched by centering, so the two
        bases coincide again. Proves the previous case is about centering and nothing else."""
        seam_lens["J"] = torch.eye(D)

        def module_fn():
            return _zero_meaned_module(torch.full((D,), 1.7))

        paper, aware = _concept_vectors(module_fn)
        assert _one_minus_abs_cos(paper, aware) < 1e-6

    def test_jlens_direction_matches_the_independent_construction(self, seam_lens):
        """`rows @ J` recomputed in-test, so the op cannot agree with itself by construction."""
        seam_lens["J"] = torch.diag(torch.arange(1.0, D + 1))
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
    def test_unfitted_layer_refusal_propagates(self, seam_lens):
        with pytest.raises(ValueError, match="does not include 5"):
            concept_ops.concept_direction_impl(
                _module(),
                AnalysisBatch(**_groups(), concept_basis="jlens_paper", jlens_layer=5),
                batch=None,
                batch_idx=0,
            )


#: Pinned revision of the public lens repo carrying the gpt2-small artifact below. Mirrors the
#: tests/hf_warm_manifest.yaml entry so the test cannot float on another organisation's `main`.
GPT2_LENS_REVISION = "0731326edff4ae730ffc5356fe1a4728c748b3a6"
GPT2_LENS_PATH = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"


class TestRealLensNonCollinearity:
    """Measurement on the shipped gpt2-small artifact, not a guard: asserts non-collinearity only.

    Warmed into the offline cache by tests/hf_warm_manifest.yaml; `path=` plus `revision=` are explicit so no Hub
    listing is needed (offline-safe, hence not hf_live: reaching the Hub is not what this exercises). gpt2's final norm
    is a LayerNorm, so the real instance exercises centering as well as scale. No gemma: gigabytes of unembedding have
    no place in the CPU phase. The measured cosine is recorded on the alignment page, not asserted: the number belongs
    to the artifact revision.
    """

    def test_bases_differ_on_gpt2_small(self):
        from types import SimpleNamespace

        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        hf.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        module = SimpleNamespace(model=hf)
        for token in (" Paris", " London"):
            ids = hf.tokenizer.encode(token, add_special_tokens=False)
            assert len(ids) == 1, f"{token!r} is not a single token: {ids}"
        fields = {
            "concept_group_a": [" Paris"],
            "concept_group_b": [" London"],
            "jlens_repo_id": "neuronpedia/jacobian-lens",
            "jlens_lens_path": GPT2_LENS_PATH,
            "jlens_revision": GPT2_LENS_REVISION,
            "jlens_layer": 8,
        }
        outs = {}
        for basis in ("jlens_paper", "jlens_norm_aware"):
            outs[basis] = concept_ops.concept_direction_impl(
                module, AnalysisBatch(**fields, concept_basis=basis), batch=None, batch_idx=0
            )
        assert outs["jlens_paper"].concept_basis == "jlens_paper"
        assert outs["jlens_norm_aware"].concept_basis == "jlens_norm_aware"
        paper, aware = outs["jlens_paper"].concept_direction, outs["jlens_norm_aware"].concept_direction
        gap = 1 - abs(
            float(torch.dot(paper, aware) / torch.linalg.vector_norm(paper) / torch.linalg.vector_norm(aware))
        )
        assert gap > 1e-3, f"bases coincide on the real lens: gap {gap:.2e}"
        print(f"\ngpt2-small layer 8 Paris-vs-London 1-|cos| = {gap:.4f}")
