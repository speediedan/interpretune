"""Tests for per-feature J-space signatures and the decoupling comparison.

The signature reads a feature's decoder vector through the folded J-lens; the decoupling score compares that disposition
against the input-side concept share. CPU-synthetic throughout (stub graph, stub transcoders, tiny random lens): the
consultation with real artifacts belongs to the GPU lanes, and nothing here needs it.

The environment helpers (``validate_env_vars``, ``required_os_env``) and ``concept_token_positions`` are tested at the
end of the file: environment variables are set per test through ``monkeypatch``, ``.env`` files live in ``tmp_path``,
and auto-discovery is replaced by a recorder so no test reads a real ``.env``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from interpretune.harness import notebook_inputs
from interpretune.harness.notebook_inputs import (
    concept_token_positions,
    feature_io_profiles,
    required_os_env,
    validate_env_vars,
)


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
            **lens_kwargs,  # pyright: ignore[reportArgumentType]
        )[0]
        without_concepts = feature_io_profiles(
            graph,
            [(10, 7)],
            target,
            transcoders,
            {1, 2},
            concept_token_ids=[30, 31],
            **lens_kwargs,  # pyright: ignore[reportArgumentType]
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
        from interpretune.harness import notebook_ui

        artifact, info = _lens_parts()
        profiles = _profiles(
            jlens_artifact=artifact,
            unembed_info=info,
            tokenizer=_StubTokenizer(),
            concept_token_ids=[5],
        )
        notebook_ui.display_feature_decoupling_table(profiles)

    def test_table_renders_without_signatures(self) -> None:
        from interpretune.harness import notebook_ui

        notebook_ui.display_feature_decoupling_table(_profiles())

    def test_projection_map_refuses_unknown_coloring(self) -> None:
        from interpretune.harness import notebook_ui

        with pytest.raises(ValueError, match="not a coloring"):
            notebook_ui.plot_decoder_projection_map(
                _profiles(),
                torch.randn(3, 8),
                torch.randn(20, 8),
                color_by="bogus",
            )


_PRESENT, _EMPTY, _ABSENT = "IT_TEST_NB_PRESENT", "IT_TEST_NB_EMPTY", "IT_TEST_NB_ABSENT"


@pytest.fixture
def env(monkeypatch) -> None:
    monkeypatch.setenv(_PRESENT, "hf_abc123")
    monkeypatch.setenv(_EMPTY, "")
    monkeypatch.delenv(_ABSENT, raising=False)


class TestValidateEnvVars:
    def test_present_accepts_empty_but_not_absent(self, env) -> None:
        assert validate_env_vars([(_PRESENT, "present"), (_EMPTY, "present")]) is True
        assert validate_env_vars([(_PRESENT, "present"), (_ABSENT, "present")]) is False

    def test_non_empty_rejects_empty_and_absent(self, env) -> None:
        assert validate_env_vars([(_PRESENT, "non-empty")]) is True
        assert validate_env_vars([(_EMPTY, "non-empty")]) is False
        assert validate_env_vars([(_ABSENT, "non-empty")]) is False

    def test_callable_receives_the_value_and_is_skipped_when_absent(self, env) -> None:
        seen: list[str] = []

        def check(value: str) -> bool:
            seen.append(value)
            return value.startswith("hf_")

        assert validate_env_vars([(_PRESENT, check)]) is True
        assert validate_env_vars([(_EMPTY, check)]) is False
        assert validate_env_vars([(_ABSENT, check)]) is False
        assert seen == ["hf_abc123", ""]

    def test_every_spec_must_pass(self, env) -> None:
        assert validate_env_vars([]) is True
        assert validate_env_vars([(_PRESENT, "non-empty"), (_EMPTY, "non-empty")]) is False

    def test_unknown_mode_is_refused_by_name(self, env) -> None:
        with pytest.raises(ValueError, match=f"Invalid mode for {_PRESENT}: required"):
            validate_env_vars([(_PRESENT, "required")])


class TestRequiredOsEnv:
    _FROM_FILE = "IT_TEST_NB_FROM_DOTENV"

    @pytest.fixture
    def clean_var(self, monkeypatch) -> None:
        # set then delete, so monkeypatch also removes whatever load_dotenv writes into os.environ
        monkeypatch.setenv(self._FROM_FILE, "placeholder")
        monkeypatch.delenv(self._FROM_FILE)

    @pytest.fixture
    def recorded_loads(self, monkeypatch) -> list[tuple]:
        calls: list[tuple] = []
        monkeypatch.setattr(notebook_inputs, "load_dotenv", lambda *args: calls.append(args) or False)
        return calls

    def test_loads_the_named_file_then_validates(self, tmp_path, clean_var) -> None:
        pytest.importorskip("dotenv")
        env_file = tmp_path / "nb.env"
        env_file.write_text(f"{self._FROM_FILE}=from_file\n", encoding="utf-8")
        assert required_os_env(env_file, [(self._FROM_FILE, "non-empty")]) is True
        assert os.environ[self._FROM_FILE] == "from_file"

    def test_failed_requirement_is_reported_after_loading(self, tmp_path, clean_var, env) -> None:
        pytest.importorskip("dotenv")
        env_file = tmp_path / "nb.env"
        env_file.write_text(f"{self._FROM_FILE}=from_file\n", encoding="utf-8")
        assert required_os_env(env_file, [(self._FROM_FILE, "present"), (_ABSENT, "present")]) is False

    def test_missing_file_falls_back_to_auto_discovery(self, tmp_path, recorded_loads) -> None:
        assert required_os_env(tmp_path / "missing.env") is True
        assert recorded_loads == [()]

    def test_no_path_auto_discovers(self, recorded_loads) -> None:
        required_os_env()
        assert recorded_loads == [()]

    def test_existing_file_is_passed_as_a_string(self, tmp_path, recorded_loads) -> None:
        env_file = tmp_path / "nb.env"
        env_file.write_text("", encoding="utf-8")
        required_os_env(env_file)
        assert recorded_loads == [(str(env_file),)]

    def test_without_dotenv_still_validates(self, monkeypatch, env) -> None:
        monkeypatch.setattr(notebook_inputs, "load_dotenv", None)
        assert required_os_env(None, [(_PRESENT, "non-empty")]) is True
        assert required_os_env(None, [(_ABSENT, "present")]) is False
        assert required_os_env(None, None) is True


class _DecodeTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.decoded: list[list[int]] = []

    def decode(self, token_ids: list[int]) -> str:
        self.decoded.append(token_ids)
        return self.pieces[token_ids[0]]


class TestConceptTokenPositions:
    def test_matches_case_space_and_sentencepiece_insensitively(self) -> None:
        tok = _DecodeTokenizer([" Fruit", "▁color", "FRUIT", "fruity", "the", "Color "])
        positions = concept_token_positions(tok, [0, 1, 2, 3, 4, 5], [" fruit ", "Color"])
        assert positions == {0, 1, 2, 5}

    def test_positions_index_the_prompt_not_the_vocab(self) -> None:
        tok = _DecodeTokenizer(["the", "Fruit"])
        assert concept_token_positions(tok, torch.tensor([0, 0, 1, 0, 1]), ["fruit"]) == {2, 4}
        # each id is decoded alone, as a plain int
        assert all(len(ids) == 1 and type(ids[0]) is int for ids in tok.decoded)

    def test_no_concept_words_matches_nothing(self) -> None:
        assert concept_token_positions(_DecodeTokenizer(["fruit"]), [0, 0], []) == set()
