"""The steering demos' shared maths, checked on synthetic tensors so no model or kernel is needed."""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.optools import UnembedNormInfo
from it_examples.utils import steering_demo_helpers as helpers


class _Tokenizer:
    """Maps each word to its character codes modulo a tiny vocabulary: multi-token words and empty groups."""

    def __call__(self, word: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(c) % 8 for c in word]}

    def decode(self, ids: list[int]) -> str:
        return f"t{ids[0]}"


class _Backend:
    def __init__(self, embed: torch.Tensor):
        self.embed = embed

    def get_embedding_weight(self, module):
        return self.embed


def test_embed_pole_pair_means_every_token_of_every_word(monkeypatch):
    embed = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    monkeypatch.setattr(helpers, "require_analysis_backend", lambda module: _Backend(embed))
    poles = helpers.embed_pole_pair(object(), _Tokenizer(), ["ab"], ["c", "d"])
    ids_a = [ord("a") % 8, ord("b") % 8]
    ids_b = [ord("c") % 8, ord("d") % 8]
    torch.testing.assert_close(poles[0], embed[ids_a].mean(dim=0))
    torch.testing.assert_close(poles[1], embed[ids_b].mean(dim=0))


def test_embed_pole_pair_refuses_an_empty_group_by_name(monkeypatch):
    monkeypatch.setattr(helpers, "require_analysis_backend", lambda module: _Backend(torch.zeros(8, 3)))
    with pytest.raises(ValueError, match=r"concept group b \(\[\]\) tokenized to no ids"):
        helpers.embed_pole_pair(object(), _Tokenizer(), ["a"], [])


def test_outside_span_share_uses_the_pseudoinverse_projector_for_non_orthonormal_poles():
    # Non-orthogonal poles spanning the x-y plane: a direction in the plane has nothing outside it even though
    # projecting onto each pole separately would miscount the overlap.
    poles = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert helpers.outside_span_share(torch.tensor([0.6, 0.8, 0.0]), poles) == pytest.approx(0.0, abs=1e-6)
    assert helpers.outside_span_share(torch.tensor([0.0, 0.6, 0.8]), poles) == pytest.approx(0.8, abs=1e-6)


def test_central_difference_slopes_recover_a_linear_gap(monkeypatch):
    gradient = torch.tensor([2.0, -3.0])

    def _gap(module, prompt, batch, site, tensor, scale, a, b):
        return float(1.0 + scale * (tensor @ gradient))

    monkeypatch.setattr(helpers, "gap_after_add", _gap)
    directions = torch.eye(2)
    slopes = helpers.central_difference_gap_slopes(None, "p", {}, "site", directions, 0, 1, eps=0.5)
    assert slopes == pytest.approx([2.0, -3.0])


def test_readout_top_tokens_matches_the_jlens_read_readout():
    torch.manual_seed(0)
    d, vocab = 4, 6
    scale = torch.rand(d) + 0.5
    info = UnembedNormInfo(w_u=torch.randn(vocab, d), norm_scale=scale, norm_kind="rmsnorm")
    lens = torch.randn(d, d)
    h = torch.randn(d)
    expected = (((h @ lens.T) * scale) @ info.w_u.T).topk(3)
    top = helpers.jlens_readout_top_tokens(h, lens, info, _Tokenizer(), k=3)
    assert [label for label, _ in top] == [f"t{int(i)}" for i in expected.indices]
    assert [score for _, score in top] == pytest.approx(expected.values.tolist(), rel=1e-5)
