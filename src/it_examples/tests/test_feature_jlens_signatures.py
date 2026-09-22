"""Tests for per-feature J-space signatures and the decoupling comparison (#421).

The signature reads a feature's decoder vector through the folded J-lens; the decoupling score compares that disposition
against the input-side concept share. CPU-synthetic throughout (stub graph, stub transcoders, tiny random lens): the
consultation with real artifacts belongs to the GPU lanes, and nothing here needs it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from it_examples.utils.example_helpers import feature_io_profiles


class _StubGraph:
    def __init__(self) -> None:
        self.active_features = torch.tensor([[10, 1, 7], [10, 2, 7], [12, 3, 9]])
        self.activation_values = torch.tensor([3.0, 1.0, 2.0])


class _StubTranscoders:
    def _get_decoder_vectors(self, layer, feats):
        torch.manual_seed(1000 + int(layer))
        return torch.randn(len(feats), 8)


class _FixedTranscoders:
    def __init__(self, vec: torch.Tensor) -> None:
        self._vec = vec

    def _get_decoder_vectors(self, layer, feats):
        return self._vec.unsqueeze(0).repeat(len(feats), 1)


def _lens_parts(vocab: int = 32, d: int = 8):
    torch.manual_seed(11)
    j = {10: torch.randn(d, d), 12: torch.randn(d, d)}
    w_u = torch.randn(vocab, d)
    scale = torch.ones(d) * 1.5
    info = SimpleNamespace(w_u=w_u, norm_scale=scale, norm_kind="rmsnorm")
    artifact = SimpleNamespace(j_by_layer=j)
    return artifact, info


class _StubTokenizer:
    def __init__(self) -> None:
        self._tokens = {i: f"tok{i}" for i in range(32)}
        self._tokens[5] = "Fruit"
        self._tokens[9] = "Color"

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._tokens.get(int(t), f"<{t}>") for t in token_ids)


def _profiles(**kwargs):
    target = torch.zeros(8)
    target[0] = 1.0
    return feature_io_profiles(
        _StubGraph(),
        [(10, 7), (12, 9), (14, 3)],
        target,
        _StubTranscoders(),
        {1, 2},
        **kwargs,
    )


class TestFeatureIOSignatures:
    def test_without_lens_the_new_fields_stay_none(self) -> None:
        for prof in _profiles():
            assert prof.jlens_signature is None
            assert prof.jlens_concept_mass is None
            assert isinstance(prof.output_projection, float)

    def test_signature_is_ranked_and_folded(self) -> None:
        artifact, info = _lens_parts()
        profiles = _profiles(
            jlens_artifact=artifact,
            unembed_info=info,
            tokenizer=_StubTokenizer(),
            concept_token_ids=[5, 9],
        )
        prof = profiles[0]
        assert prof.jlens_signature is not None and len(prof.jlens_signature) == 10
        scores = [s for _, s in prof.jlens_signature]
        assert scores == sorted(scores, reverse=True)
        assert prof.jlens_concept_mass is not None and 0.0 <= prof.jlens_concept_mass <= 1.0

    def test_concept_mass_counts_only_concept_tokens(self) -> None:
        d = 8
        j = {10: torch.eye(d), 12: torch.eye(d)}
        w_u = torch.zeros(32, d)
        w_u[5] = torch.ones(d) * 10.0
        w_u[9] = torch.ones(d) * 5.0
        info = SimpleNamespace(w_u=w_u, norm_scale=torch.ones(d), norm_kind="rmsnorm")
        artifact = SimpleNamespace(j_by_layer=j)
        target = torch.ones(d) / (d**0.5)
        graph = _StubGraph()
        transcoders = _FixedTranscoders(torch.ones(d))
        lens_kwargs: dict[str, object] = dict(
            jlens_artifact=artifact,
            unembed_info=info,
            tokenizer=_StubTokenizer(),
        )
        with_concepts = feature_io_profiles(
            graph,
            [(10, 7)],
            target,
            transcoders,
            {1, 2},
            concept_token_ids=[5, 9],
            **lens_kwargs,  # type: ignore[arg-type]
        )[0]
        without_concepts = feature_io_profiles(
            graph,
            [(10, 7)],
            target,
            transcoders,
            {1, 2},
            concept_token_ids=[30, 31],
            **lens_kwargs,  # type: ignore[arg-type]
        )[0]
        assert with_concepts.jlens_signature == without_concepts.jlens_signature
        assert with_concepts.jlens_concept_mass == pytest.approx(1.0)
        assert without_concepts.jlens_concept_mass == pytest.approx(0.0)

    def test_unfitted_layer_refuses_rather_than_interpolates(self) -> None:
        artifact, info = _lens_parts()
        profiles = _profiles(
            jlens_artifact=artifact,
            unembed_info=info,
            tokenizer=_StubTokenizer(),
            concept_token_ids=[5],
        )
        assert profiles[2].jlens_signature is None
        assert profiles[2].jlens_concept_mass is None

    def test_old_metric_survives_beside_the_new_one(self) -> None:
        artifact, info = _lens_parts()
        plain = _profiles()[0]
        with_sig = _profiles(
            jlens_artifact=artifact,
            unembed_info=info,
            tokenizer=_StubTokenizer(),
            concept_token_ids=[5],
        )[0]
        assert plain.output_projection == pytest.approx(with_sig.output_projection)
        assert plain.input_concept_share == pytest.approx(with_sig.input_concept_share)


class TestSignatureDisplay:
    def test_table_renders_signature_column_when_present(self) -> None:
        from it_examples.utils import nb_ui_utils

        artifact, info = _lens_parts()
        profiles = _profiles(
            jlens_artifact=artifact,
            unembed_info=info,
            tokenizer=_StubTokenizer(),
            concept_token_ids=[5],
        )
        nb_ui_utils.display_feature_decoupling_table(profiles)

    def test_table_renders_without_signatures(self) -> None:
        from it_examples.utils import nb_ui_utils

        nb_ui_utils.display_feature_decoupling_table(_profiles())

    def test_projection_map_refuses_unknown_coloring(self) -> None:
        from it_examples.utils import nb_ui_utils

        with pytest.raises(ValueError, match="not a coloring"):
            nb_ui_utils.plot_decoder_projection_map(
                _profiles(),
                torch.randn(3, 8),
                torch.randn(20, 8),
                color_by="bogus",
            )
