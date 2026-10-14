"""Lens artifact resolution: does it place a real repository's every model, and refuse when it cannot?

Offline throughout. The listing in `fixtures/jlens_repo_listing.json` is a RECORDING of the published
repository rather than a hand-built sample, because a sample would only contain the shapes we already
thought of, and it is the shapes we did not that break a formatted path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from interpretune.analysis import optools
from interpretune.analysis.optools import (
    JLENS_KINDS,
    JLensArtifact,
    jlens_layer_for_percentile,
    resolve_jlens,
)

LISTING = json.loads((Path(__file__).parent / "fixtures" / "jlens_repo_listing.json").read_text())
SIDECARS: dict[str, dict] = LISTING["sidecars"]
CANONICAL = "{d}/jlens/Salesforce-wikitext/{d}_jacobian_lens.pt"


def _recorded_artifacts(kind: str) -> dict[str, list[str]]:
    """The resolver's own grouping of the recorded listing, so the kind filter is exercised rather than
    replaced."""
    with mock.patch("huggingface_hub.HfApi.list_repo_files", return_value=LISTING["files"]):
        return optools._jlens_repo_artifacts("repo", None, None, kind)


ARTIFACTS: dict[str, list[str]] = _recorded_artifacts("jlens")
JPP_ARTIFACTS: dict[str, list[str]] = _recorded_artifacts("jpp")


def _hf_names(kind: str) -> dict[str, str]:
    """``{model_dir: hf_model_name}`` for the directories of ``kind`` whose sidecar names a model."""
    kind_dir = JLENS_KINDS[kind][0]
    return {
        d.split("/")[0]: facts["hf_model_name"]
        for d, facts in SIDECARS.items()
        if d.split("/")[1] == kind_dir and facts.get("hf_model_name")
    }


HF_NAMES: dict[str, str] = _hf_names("jlens")
JPP_HF_NAMES: dict[str, str] = _hf_names("jpp")


class _Cfg:
    def __init__(self, name):
        self._name_or_path = name


class _Model:
    def __init__(self, name):
        self.config = _Cfg(name)


class _Module:
    def __init__(self, name):
        self.model = _Model(name)


def _recorded_sidecar(repo_id, artifact_path, revision, token):
    """The recorded sidecar facts for an artifact's directory, shaped as the published sidecars are."""
    facts = SIDECARS.get(artifact_path.rsplit("/", 1)[0])
    if facts is None:
        return {}
    fit = {"n_prompts": 96} if facts["fit_method"] is None else {"method": facts["fit_method"], "n_prompts": 96}
    return {"hf_model_name": facts["hf_model_name"], "fit": fit, "results": {"prompts_fitted": 277}}


@pytest.fixture
def offline_repo(monkeypatch):
    """Serve the recorded listing and sidecars; any real network call is a test bug."""
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", lambda *a, **k: LISTING["files"])
    monkeypatch.setattr(optools, "_read_jlens_sidecar", _recorded_sidecar)


def _discover(module, kind="jlens"):
    artifacts = ARTIFACTS if kind == "jlens" else JPP_ARTIFACTS
    return optools._discover_jlens_path(module, "repo", artifacts, None, None, kind)


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

    @pytest.mark.parametrize("short_name", ["gpt2", "gemma-2-2b", "Qwen3.5-4B"])
    def test_a_model_loaded_by_its_SHORT_name_still_resolves(self, offline_repo, short_name):
        """Regression: sidecars record `org/name` while a short-loaded model self-reports `name`.

        This escaped the cases above by construction rather than by oversight. Every other test here
        feeds a sidecar's own `hf_model_name` back in, so the two strings always matched and the
        comparison could never be too strict. The real repository found it on the first pass because
        `AutoModelForCausalLM.from_pretrained("gpt2")` reports `gpt2`, not `openai-community/gpt2`.
        """
        assert _discover(_Module(short_name)).split("/")[0] in ARTIFACTS

    def test_a_qualified_name_from_the_WRONG_organization_is_still_refused(self, offline_repo):
        """The positive control on the fix above: relaxing the comparison must not disarm it.

        A basename-only comparison for every name would make this pass, which is the failure mode a
        loosened check invites: it resolves everything, including the thing it exists to catch.
        """
        with pytest.raises(ValueError, match="publishes no 'jlens' lens whose `hf_model_name`"):
            _discover(_Module("not-the-real-org/gpt2"))

    def test_a_non_wikitext_fitting_corpus_is_not_assumed_away(self, offline_repo):
        """One model was fit on a different corpus; a hard-coded corpus segment would miss it."""
        path = optools._select_jlens_artifact(ARTIFACTS["deepseek-v4-flash"], "deepseek-v4-flash", "jlens")
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
            optools._select_jlens_artifact(candidates, "m", "jlens")

    def test_an_unknown_model_is_named_rather_than_silently_mismatched(self, offline_repo):
        with pytest.raises(ValueError, match="publishes no 'jlens' lens whose `hf_model_name`"):
            _discover(_Module("some-org/not-a-published-model"))

    def test_a_model_name_that_cannot_be_determined_says_how_to_proceed(self, offline_repo):
        with pytest.raises(ValueError, match="pass `model_id=`"):
            _discover(object())

    def test_an_unknown_model_id_lists_what_is_available(self, offline_repo, monkeypatch):
        with pytest.raises(ValueError, match="has no 'jlens' lens directory"):
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
            kind="jlens",
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


class TestLensKindsInTheRecordedRepository:
    def test_the_kind_filter_partitions_the_lens_files_and_drops_everything_else(self):
        """Every `.pt` under a kind directory lands in exactly that kind; nothing else (e.g. `oracle/`) does."""
        grouped = {kind: _recorded_artifacts(kind) for kind in JLENS_KINDS}
        for kind, artifacts in grouped.items():
            kind_dir = JLENS_KINDS[kind][0]
            assert artifacts, f"the recording has no {kind!r} lenses; re-record or revisit the kind table"
            assert all(p.split("/")[1] == kind_dir for paths in artifacts.values() for p in paths)
        selected = sum(len(paths) for artifacts in grouped.values() for paths in artifacts.values())
        lens_files = [f for f in LISTING["files"] if len(f.split("/")) == 4]
        assert selected == len(lens_files), "a lens file was dropped by, or shared between, kinds"

    def test_every_jpp_directory_resolves_from_its_own_model_name(self, offline_repo):
        unresolved = {
            model_dir: got
            for model_dir, hf_name in JPP_HF_NAMES.items()
            if not (got := _discover(_Module(hf_name), kind="jpp")).startswith(f"{model_dir}/jpp/")
        }
        assert JPP_HF_NAMES and not unresolved, unresolved

    def test_a_model_with_both_kinds_resolves_each_by_the_kind_asked_for(self, offline_repo):
        module = _Module("Qwen/Qwen3-4B")
        assert _discover(module) == "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"
        assert _discover(module, kind="jpp") == "qwen3-4b/jpp/Salesforce-wikitext/Qwen3-4B_jpp_lens.pt"

    def test_a_model_without_a_jpp_lens_is_refused_naming_the_kind(self, offline_repo):
        with pytest.raises(ValueError, match="has no 'jpp' lens directory 'gemma-3-1b-it'"):
            resolve_jlens(_Module("google/gemma-3-1b-it"), repo_id="repo", model_id="gemma-3-1b-it", kind="jpp")
        with pytest.raises(ValueError, match="publishes no 'jpp' lens whose `hf_model_name`"):
            _discover(_Module("google/gemma-3-1b-it"), kind="jpp")

    @pytest.mark.parametrize("kind", ["rlens", "JPP", "", 1])
    def test_an_unknown_kind_is_refused_by_name(self, offline_repo, kind):
        with pytest.raises(ValueError, match="is not a lens kind: expected one of"):
            resolve_jlens(_Module("Qwen/Qwen3-4B"), repo_id="repo", kind=kind)


def _write_lens(tmp_path, *, provenance=None, author_format=False, d=4):
    """A tiny lens checkpoint in the published layout (or the fitting library's native one), on local disk."""
    j = {layer: torch.eye(d) * (layer + 1) for layer in (0, 1, 2)}
    if author_format:
        ckpt = {"parameters": {"jacobians": j}, "source_layers": [0, 1, 2], "config": {"lrp_mode": "rlens"}}
    else:
        ckpt = {"J": j, "source_layers": [0, 1, 2], "d_model": d, "n_prompts": 96}
        if provenance is not None:
            ckpt["provenance"] = provenance
    path = tmp_path / "lens.pt"
    torch.save(ckpt, path)
    return path


JPP_SIDECAR = {
    "hf_model_name": "org/m",
    "fit": {"method": "jpp", "n_prompts": 96, "lrp_mode": "rlens"},
    "lens": {"capture_point": "block_output", "dtype": "float16"},
    "extra_metadata": {"num_clusters": 0},
}
JLENS_SIDECAR = {"hf_model_name": "org/m", "fit": {"n_prompts": 460}, "results": {"prompts_fitted": 460}}
JPP_PATH = "m/jpp/Salesforce-wikitext/m_jpp_lens.pt"
JLENS_PATH = "m/jlens/Salesforce-wikitext/m_jacobian_lens.pt"


@pytest.fixture
def local_lens(tmp_path, monkeypatch):
    """Serve one local checkpoint for any repo path, with a chosen sidecar; returns a configurator."""

    def configure(sidecar, **ckpt_kwargs):
        local = _write_lens(tmp_path, **ckpt_kwargs)
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(local))
        monkeypatch.setattr(optools, "_read_jlens_sidecar", lambda *a, **k: dict(sidecar))

    return configure


class TestLoadingByKind:
    def test_a_jpp_lens_records_its_kind_lens_block_and_checkpoint_provenance(self, local_lens):
        local_lens(JPP_SIDECAR, provenance={"method": "jpp", "lrp_mode": "rlens"})
        artifact = resolve_jlens(_Module("org/m"), repo_id="repo", path=JPP_PATH)
        assert artifact.kind == "jpp"
        assert (artifact.provenance["kind"], artifact.provenance["kind_source"]) == ("jpp", "sidecar")
        assert artifact.provenance["lens"]["capture_point"] == "block_output"
        assert artifact.provenance["extra_metadata"] == {"num_clusters": 0}
        assert "results" not in artifact.provenance  # J++ sidecars publish none, and that is accepted
        expected_checkpoint = {"n_prompts": 96, "provenance": {"method": "jpp", "lrp_mode": "rlens"}}
        assert artifact.provenance["checkpoint"] == expected_checkpoint

    def test_a_jlens_lens_reads_as_before_plus_its_kind(self, local_lens):
        """Positive control: a J-lens resolves as a J-lens and keeps the provenance it always carried."""
        local_lens(JLENS_SIDECAR)
        artifact = resolve_jlens(_Module("org/m"), repo_id="repo", path=JLENS_PATH)
        assert artifact.kind == "jlens"
        assert artifact.provenance["results"] == {"prompts_fitted": 460}
        assert artifact.provenance["fit"] == {"n_prompts": 460}

    @pytest.mark.parametrize(
        "sidecar, provenance, path, expected_source",
        [
            ({}, {"method": "jpp"}, JPP_PATH, "checkpoint"),  # no sidecar: the checkpoint's own provenance
            ({}, None, JPP_PATH, "path"),  # neither: the published kind directory
            (JPP_SIDECAR, None, "local/m_jpp_lens.pt", "sidecar"),  # outside the layout, with a sidecar
        ],
    )
    def test_an_unrequested_kind_comes_from_the_artifacts_own_declarations(
        self, local_lens, sidecar, provenance, path, expected_source
    ):
        local_lens(sidecar, provenance=provenance)
        artifact = resolve_jlens(_Module("org/m"), repo_id="repo", path=path)
        assert (artifact.kind, artifact.provenance["kind_source"]) == ("jpp", expected_source)

    def test_a_requested_kind_the_sidecar_contradicts_is_refused(self, local_lens):
        local_lens(JPP_SIDECAR)
        with pytest.raises(ValueError, match="'jpp' lens according to its sidecar, but the resolved kind is 'jlens'"):
            resolve_jlens(_Module("org/m"), repo_id="repo", path=JPP_PATH, kind="jlens")

    def test_declarations_that_disagree_with_each_other_are_refused(self, local_lens):
        """A J++ sidecar beside a file under `jlens/`: nothing was requested, and still nothing is guessed."""
        local_lens(JPP_SIDECAR)
        with pytest.raises(ValueError, match="is a 'jlens' lens according to its path"):
            resolve_jlens(_Module("org/m"), repo_id="repo", path=JLENS_PATH)

    def test_a_path_nothing_declares_a_kind_for_is_refused(self, local_lens):
        local_lens({})
        with pytest.raises(ValueError, match="cannot tell which lens kind"):
            resolve_jlens(_Module("org/m"), repo_id="repo", path="local/lens.pt")

    def test_an_unknown_fit_method_is_refused_by_name(self, local_lens):
        local_lens({**JPP_SIDECAR, "fit": {"method": "rlens"}})
        with pytest.raises(ValueError, match="declares fit method 'rlens' in its sidecar"):
            resolve_jlens(_Module("org/m"), repo_id="repo", path=JPP_PATH)

    def test_the_fitting_librarys_native_layout_is_refused_naming_the_conversion(self, local_lens):
        local_lens(JPP_SIDECAR, author_format=True)
        with pytest.raises(ValueError, match=r"native layout \(`parameters.jacobians`\).*`fit.method` is `jpp`"):
            resolve_jlens(_Module("org/m"), repo_id="repo", path=JPP_PATH)


class TestBatchLevelKind:
    """An op collection resolves through ``resolve_jlens_layer(module, batch, {})``; the batch must carry the
    kind."""

    def test_a_batch_level_kind_reaches_a_caller_that_passes_no_kwargs(self, local_lens):
        local_lens(JPP_SIDECAR)
        batch = {"jlens_kind": "jpp", "jlens_lens_path": JPP_PATH, "jlens_layer": 1}
        j, layer, artifact = optools.resolve_jlens_layer(_Module("org/m"), batch, {})
        assert (layer, artifact.kind, artifact.provenance["kind_source"]) == (1, "jpp", "requested")
        assert torch.equal(j, torch.eye(4) * 2)

    def test_the_batch_level_kind_is_really_read(self, local_lens):
        """Control: the same batch asking for the other kind is refused, so the batch value took effect."""
        local_lens(JPP_SIDECAR)
        batch = {"jlens_kind": "jlens", "jlens_lens_path": JPP_PATH, "jlens_layer": 1}
        with pytest.raises(ValueError, match="resolved kind is 'jlens'"):
            optools.resolve_jlens_layer(_Module("org/m"), batch, {})
