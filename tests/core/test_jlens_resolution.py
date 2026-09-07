"""Lens artifact resolution: does it place a real repository's every model, and refuse when it cannot?

Offline throughout. The listing in `fixtures/jlens_repo_listing.json` is a RECORDING of the published
repository rather than a hand-built sample, because a sample would only contain the shapes we already
thought of, and it is the shapes we did not that break a formatted path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from interpretune.analysis import optools
from interpretune.analysis.optools import (
    JLensArtifact,
    jlens_layer_for_percentile,
    resolve_jlens,
)

LISTING = json.loads((Path(__file__).parent / "fixtures" / "jlens_repo_listing.json").read_text())
ARTIFACTS: dict[str, list[str]] = LISTING["artifacts"]
HF_NAMES: dict[str, str] = LISTING["hf_model_names"]
CANONICAL = "{d}/jlens/Salesforce-wikitext/{d}_jacobian_lens.pt"


class _Cfg:
    def __init__(self, name):
        self._name_or_path = name


class _Model:
    def __init__(self, name):
        self.config = _Cfg(name)


class _Module:
    def __init__(self, name):
        self.model = _Model(name)


@pytest.fixture
def offline_repo(monkeypatch):
    """Serve the recorded listing and sidecars; any real network call is a test bug."""
    monkeypatch.setattr(optools, "_jlens_repo_artifacts", lambda *a, **k: ARTIFACTS)

    def _sidecar(repo_id, artifact_path, revision, token):
        model_dir = artifact_path.split("/")[0]
        name = HF_NAMES.get(model_dir)
        return {"hf_model_name": name, "results": {"prompts_fitted": 277}} if name else {}

    monkeypatch.setattr(optools, "_read_jlens_sidecar", _sidecar)


def _discover(module):
    return optools._discover_jlens_path(module, "repo", ARTIFACTS, None, None)


class TestTheLayoutThisResolverExistsFor:
    def test_the_canonical_path_is_the_minority_case(self):
        """If this ever flips, a formatted path becomes defensible again and this design is overkill."""
        canonical = sum(1 for d, fs in ARTIFACTS.items() for f in fs if f == CANONICAL.format(d=d))
        total = sum(len(v) for v in ARTIFACTS.values())
        assert canonical < total / 2, f"{canonical}/{total} canonical: revisit whether discovery is needed"

    def test_every_directory_with_a_sidecar_resolves_from_its_own_model_name(self, offline_repo):
        unresolved = {}
        for model_dir, hf_name in HF_NAMES.items():
            try:
                got = _discover(_Module(hf_name))
            except Exception as exc:  # - the failure detail is the point of the report
                unresolved[model_dir] = f"{type(exc).__name__}: {exc}"
                continue
            if got.split("/")[0] != model_dir:
                unresolved[model_dir] = f"resolved to {got.split('/')[0]}"
        assert not unresolved, unresolved

    @pytest.mark.parametrize(
        "hf_name, expected",
        [
            # the stem is the HF name, not the directory
            ("openai-community/gpt2", "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"),
            # a base-model suffix the directory drops
            ("google/gemma-3-1b-pt", "gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt"),
            # two artifacts present: the unsuffixed one wins
            ("Qwen/Qwen3.5-4B", "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt"),
            # the ONLY artifact is suffixed, so requiring an unsuffixed name would fail here
            ("Qwen/Qwen3.6-27B", "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"),
        ],
    )
    def test_the_awkward_shapes_resolve(self, offline_repo, hf_name, expected):
        assert _discover(_Module(hf_name)) == expected

    def test_a_non_wikitext_fitting_corpus_is_not_assumed_away(self, offline_repo):
        """One model was fit on a different corpus; a hard-coded corpus segment would miss it."""
        path = optools._select_jlens_artifact(ARTIFACTS["deepseek-v4-flash"], "deepseek-v4-flash")
        assert "/NeelNanda-pile-10k/" in path


class TestRefusingRatherThanGuessing:
    def test_several_artifacts_and_no_default_raises_naming_them(self):
        """Never fires on today's repository, so it is asserted here or it is not asserted at all.

        The repair someone reaches for is sort-order selection, which would make this pass silently.
        """
        candidates = [
            "m/jlens/c/m_jacobian_lens_n1000.pt",
            "m/jlens/c/m_jacobian_lens_cblank.pt",
        ]
        with pytest.raises(ValueError, match="none is unambiguously the default"):
            optools._select_jlens_artifact(candidates, "m")

    def test_an_unknown_model_is_named_rather_than_silently_mismatched(self, offline_repo):
        with pytest.raises(ValueError, match="publishes no lens whose `hf_model_name`"):
            _discover(_Module("some-org/not-a-published-model"))

    def test_a_model_name_that_cannot_be_determined_says_how_to_proceed(self, offline_repo):
        with pytest.raises(ValueError, match="pass `model_id=`"):
            _discover(object())

    def test_an_unknown_model_id_lists_what_is_available(self, offline_repo, monkeypatch):
        with pytest.raises(ValueError, match="has no lens directory"):
            resolve_jlens(_Module("openai-community/gpt2"), repo_id="repo", model_id="no-such-dir")


class TestArtifactSurface:
    @staticmethod
    def _artifact(layers=(0, 6, 12, 18, 24)):
        return JLensArtifact(
            j_by_layer={i: torch.eye(4) for i in layers},
            source_layers=list(layers),
            d_model=4,
            repo_id="repo",
            path="m/jlens/c/m_jacobian_lens.pt",
            hf_model_name="org/m",
            provenance={"results": {"prompts_fitted": 277}},
        )

    @pytest.mark.parametrize("pct, expected", [(0.0, 0), (0.5, 12), (0.85, 18), (1.0, 24)])
    def test_percentile_indexes_the_fitted_set_not_model_depth(self, pct, expected):
        """Pins the distinction rather than the intuition.

        With layers fit at 0/6/12/18/24, `percentile=0.85` selects index 3, which is layer 18 and so 75% of the way down
        the model. That is the definition the published steering recipes were tuned against, so a change to true depth
        would move which layer a validated demo patches. Asserting 18 here is what makes such a change fail rather than
        pass quietly.
        """
        assert jlens_layer_for_percentile(self._artifact(), pct) == expected

    def test_every_selected_layer_was_actually_fit(self):
        """The property the function exists for, independent of the indexing convention above."""
        artifact = self._artifact()
        for pct in (0.0, 0.17, 0.33, 0.5, 0.66, 0.85, 1.0):
            assert jlens_layer_for_percentile(artifact, pct) in artifact.j_by_layer

    @pytest.mark.parametrize("pct", [-0.01, 1.01])
    def test_percentile_out_of_range_raises(self, pct):
        with pytest.raises(ValueError, match=r"percentile must be in \[0, 1\]"):
            jlens_layer_for_percentile(self._artifact(), pct)

    def test_an_unverified_resolution_is_visible_in_the_return_value(self):
        """Not every directory publishes a sidecar, and a caller must be able to tell."""
        assert self._artifact()._replace(hf_model_name=None).hf_model_name is None
