"""The docs-artifact renderer (``interpretune.utils.notebook_docs``) and interpretune's hooks for it.

The engine is repository-neutral and configured from ``[tool.notebook-publishing]``; most cases here drive it
against a throwaway repository, so the case of a vendoring repository with no hooks at all is exercised. The
op-surface cases test ``scripts/notebook_docs_hooks.py``, interpretune's implementation of the hook interface.

`--check-stale` can see output-only drift, not just source drift: the engine's comparator builds its signature from
``(cell_type, source)``, so output drift is unrepresentable there, and the op-surface hooks exist to close that.
Confirmed in the wild: one rename touched two artifacts, and the gate flagged the one whose cell SOURCE mentioned the
symbol while staying silent on the one that mentioned it only in captured OUTPUT.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from interpretune.utils import notebook_docs as engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOKS_PATH = PROJECT_ROOT / "scripts" / "notebook_docs_hooks.py"
SHIM = PROJECT_ROOT / "scripts" / "render_notebook_docs_artifacts.py"


@pytest.fixture(scope="module")
def hooks():
    """Interpretune's hooks file, loaded by path the way the engine loads it."""
    spec = importlib.util.spec_from_file_location("_it_notebook_docs_hooks_under_test", HOOKS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(recorded: list[str] | None, output_text: str = "") -> dict:
    notebook: dict = {
        "cells": [{"cell_type": "code", "source": ["x = 1\n"], "outputs": [{"text": [output_text]}]}],
        "metadata": {},
    }
    if recorded is not None:
        notebook["metadata"]["interpretune_op_surface"] = sorted(recorded)
    return notebook


def _text(notebook: dict) -> str:
    return engine.captured_output_text(notebook)


def _notebook(output_lines: list[str] | None = None, source: str = "x = 1\n") -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": [source],
                "outputs": [{"output_type": "stream", "name": "stdout", "text": output_lines or []}],
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _repo(tmp_path: Path, *, extra: str = "", hooks_path: Path | None = None) -> Path:
    """A throwaway consuming repository: a publish lane, an artifact lane, and the table naming them."""
    root = tmp_path / "consumer"
    (root / "publish" / "lane").mkdir(parents=True)
    (root / "artifacts").mkdir()
    table = (
        '[tool.notebook-publishing]\ndev_dir = "dev"\npublish_dir = "publish"\ncolab_repo = "org/consumer"\n'
        'install_cell = ["# %pip install consumer"]\ndocs_artifact_dir = "artifacts"\n'
        'docs_render_command = "python scripts/notebook_docs.py"\n'
    )
    if hooks_path is not None:
        table += f"docs_hooks = {json.dumps(str(hooks_path))}\n"
    (root / "pyproject.toml").write_text(table + extra)
    return root


def _write(path: Path, notebook: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook), encoding="utf-8")


class TestOutputOnlyDrift:
    def test_dead_op_name_in_output_is_detected(self, hooks):
        """The case that slipped through in the wild: renamed op present ONLY in captured output."""
        artifact = _artifact(recorded=["labels_to_ids", "sae_correct_acts"], output_text="ran sae_correct_acts ok\n")
        assert hooks.stale_output_references(artifact, {"labels_to_ids", "latent_correct_acts"}, _text(artifact)) == [
            "sae_correct_acts"
        ]

    def test_live_op_names_in_output_are_not_flagged(self, hooks):
        artifact = _artifact(recorded=["labels_to_ids"], output_text="ran labels_to_ids ok\n")
        assert hooks.stale_output_references(artifact, {"labels_to_ids"}, _text(artifact)) == []

    def test_a_dead_name_absent_from_output_is_not_flagged(self, hooks):
        """Only fires when the dead name is actually present -- a removed op nobody printed is not drift."""
        artifact = _artifact(recorded=["labels_to_ids", "sae_correct_acts"], output_text="nothing relevant\n")
        assert hooks.stale_output_references(artifact, {"labels_to_ids"}, _text(artifact)) == []

    def test_unstamped_artifact_reports_nothing_rather_than_guessing(self, hooks):
        """Absence of a stamp is not evidence of freshness; the check reports those separately."""
        artifact = _artifact(recorded=None, output_text="sae_correct_acts\n")
        assert hooks.stale_output_references(artifact, {"labels_to_ids"}, _text(artifact)) == []

    def test_output_text_is_collected_from_data_mimetypes_too(self, hooks):
        """Rich outputs carry text under `data`, not `text`; missing those would blind the check."""
        artifact = {
            "cells": [{"cell_type": "code", "outputs": [{"data": {"text/plain": ["sae_correct_acts\n"]}}]}],
            "metadata": {"interpretune_op_surface": ["sae_correct_acts"]},
        }
        assert hooks.stale_output_references(artifact, {"labels_to_ids"}, _text(artifact)) == ["sae_correct_acts"]


class TestOpSurfaceSource:
    def test_op_names_come_from_yaml_not_the_dispatcher(self, hooks):
        """Must be deterministic across sessions: a pulled hub collection must not change the answer."""
        names = hooks.bundled_op_names()
        assert "labels_to_ids" in names and "logit_diffs_base" in names
        assert all("." not in n for n in names), "bundled names are bare; a dotted name implies a hub source"

    def test_stamping_is_idempotent_and_sorted(self, hooks):
        notebook: dict = {"cells": [], "metadata": {}}
        hooks.stamp_op_surface(notebook, {"b", "a"})
        first = notebook["metadata"]["interpretune_op_surface"]
        hooks.stamp_op_surface(notebook, {"b", "a"})
        assert first == notebook["metadata"]["interpretune_op_surface"] == ["a", "b"]


def test_bundled_op_names_degrades_when_yaml_is_absent(hooks, monkeypatch):
    """The op surface must degrade to unavailable, not raise, when pyyaml is missing.

    `docs-build.yml` runs `--check-stale` TWICE, and the first run happens deliberately BEFORE the requirements install
    so it fails fast. A third-party import there is not a missing dependency to add, it breaks that contract -- which
    is exactly how this surfaced, as a `ModuleNotFoundError` the workflow then reported as "an artifact has drifted".
    `sys.modules["yaml"] = None` is the documented way to force `import yaml` to raise ImportError.
    """
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert hooks.bundled_op_names() == set()
    assert hooks.HOOKS["notes"]() == [
        "note: pyyaml unavailable, output-drift check skipped (source drift still checked)"
    ]


def test_yaml_absence_degrades_output_drift_but_keeps_source_drift(hooks, monkeypatch):
    """Degrading must cost ONLY the output-drift half, never the source-drift half.

    Losing source drift too would turn the fail-fast pre-install step into one that passes unconditionally -- worse than
    removing it, because it would still report success.
    """
    monkeypatch.setitem(sys.modules, "yaml", None)
    artifact = {
        "metadata": {hooks.OP_SURFACE_KEY: ["zzz_removed_op"]},
        "cells": [{"cell_type": "code", "source": ["x = 1\n"], "outputs": [{"text": ["ran zzz_removed_op\n"]}]}],
    }
    # With no surface the comparison cannot be made, so it reports nothing rather than guess...
    assert hooks.stale_output_references(artifact, set(), _text(artifact)) == []
    assert hooks.HOOKS["check"](artifact, rel="a.ipynb", output_text=_text(artifact)) == []
    # ...while the engine's source comparator is pure stdlib and stays fully functional.
    assert engine.artifact_matches_source(artifact, artifact)
    assert not engine.artifact_matches_source(
        artifact, {"cells": [{"cell_type": "code", "source": ["x = 2\n"]}], "metadata": {}}
    )


class TestDriftExitCodeContract:
    """`docs-build` must be able to tell "an artifact drifted" from "the check failed to run".

    Both exit non-zero, so the previous `|| { echo "...drifted..."; }` wrapper could only ever assert the first. It got
    that wrong in production: a `ModuleNotFoundError` was reported as drift. Only a distinct exit code lets the caller
    say WHICH happened, so the code IS the contract, and it spans two files that must agree.
    """

    def test_drift_code_avoids_the_codes_it_must_be_distinguishable_from(self):
        """0/1/2 are already spoken for, and 2 (argparse's usage error, which runs before `main()`) is easy to
        miss."""
        assert engine.DRIFT_EXIT_CODE not in (0, 1, 2)

    def test_argparse_usage_error_is_not_reported_as_drift(self):
        """The collision above, measured rather than assumed, through the shim docs-build actually runs."""
        result = subprocess.run([sys.executable, str(SHIM), "--bogus-flag"], capture_output=True)
        assert result.returncode == 2, "argparse's usage convention changed; re-check DRIFT_EXIT_CODE"
        assert result.returncode != engine.DRIFT_EXIT_CODE

    def test_the_shim_runs_without_importing_interpretune(self):
        """Docs-build runs the shim before installing anything; importing the package would pull in torch."""
        probe = (
            "import runpy, sys; sys.argv = ['x', '--list']\n"
            "try:\n    runpy.run_path(sys.argv_path, run_name='__main__')\nexcept SystemExit:\n    pass\n"
            "assert 'interpretune' not in sys.modules, sorted(m for m in sys.modules if m.startswith('interp'))\n"
        ).replace("sys.argv_path", repr(str(SHIM)))
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert result.returncode == 0, result.stderr

    def test_workflow_branches_on_the_code_the_script_actually_returns(self):
        """The engine and `docs-build.yml` must agree, and nothing else checks that they do."""
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "docs-build.yml").read_text()
        assert f"\n            {engine.DRIFT_EXIT_CODE}) echo " in workflow, (
            f"docs-build.yml has no branch for DRIFT_EXIT_CODE={engine.DRIFT_EXIT_CODE}"
        )
        # The wildcard arm must say plainly that it is NOT a drift report; that sentence is the fix.
        assert "NOT a drift report" in workflow


class TestMissingArtifactIsDrift:
    """A published notebook with no artifact renders code-only on the site with no error anywhere, which is the
    state the artifact mandate exists to prevent.

    Measured before this landed: `--check-stale` exited 0 with one notebook reported MISSING by `--list`.
    """

    def test_a_published_notebook_with_no_artifact_fails_the_check_by_name(self, tmp_path, capsys):
        """Positive control: the check can fire on exactly the planted absence, naming the configured command."""
        root = _repo(tmp_path)
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        assert engine.main(["--check-stale"], root=root) == engine.DRIFT_EXIT_CODE
        err = capsys.readouterr().err
        # The path is printed in the platform's own spelling (a backslash on Windows), so compare that form.
        assert "MISSING artifact" in err and str(Path("lane") / "nb.ipynb") in err
        assert "python scripts/notebook_docs.py --notebook nb" in err

    def test_a_present_matching_artifact_passes(self, tmp_path, capsys):
        """Negative control on the same fixture: presence is the only thing that changed."""
        root = _repo(tmp_path)
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        _write(root / "artifacts" / "lane" / "nb.ipynb", _notebook())
        assert engine.main(["--check-stale"], root=root) == 0
        assert "MISSING artifact" not in capsys.readouterr().err


class TestSkippedSectionIsDrift:
    """An enabled section that skipped at render time ships as a `[SKIPPED]` line where the docs show its output,
    with no error anywhere.

    Measured on the local-Neuronpedia steering demo, whose J-space section rendered as a skip line (the render host's
    token could not see the private collection) and stayed that way until someone read the page.
    """

    SKIP_LINE = "[SKIPPED] J-space section: could not pull owner/collection: 404"
    # The SOURCE carries the print that produces the marker in every case, so the check can only be keyed on outputs.
    SOURCE = 'print("[SKIPPED] J-space section: ...")\n'

    def _plant(self, tmp_path: Path, artifact_output: list[str]) -> Path:
        root = _repo(tmp_path)
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook([], self.SOURCE))
        _write(root / "artifacts" / "lane" / "nb.ipynb", _notebook(artifact_output, self.SOURCE))
        return root

    def test_a_skipped_section_in_the_outputs_fails_the_check_by_name(self, tmp_path, capsys):
        """Positive control: the check fires on exactly the planted skip line and names the section."""
        root = self._plant(tmp_path, ["pulled nothing\n", self.SKIP_LINE + "\n"])
        assert engine.main(["--check-stale"], root=root) == engine.DRIFT_EXIT_CODE
        err = capsys.readouterr().err
        assert "SKIPPED section in artifact" in err and self.SKIP_LINE in err and str(Path("lane") / "nb.ipynb") in err

    def test_the_marker_in_source_alone_passes(self, tmp_path, capsys):
        """Negative control on the same fixture: the source's print is not evidence the skip fired."""
        root = self._plant(tmp_path, ["pulled owner/collection @ 872760c7\n"])
        assert engine.main(["--check-stale"], root=root) == 0
        assert "SKIPPED section" not in capsys.readouterr().err

    def test_the_marker_inside_a_line_does_not_count(self, tmp_path):
        """Only a line that STARTS with the marker is a skip announcement; prose quoting it is not."""
        root = self._plant(tmp_path, ["the cell prints [SKIPPED] when the pull fails\n"])
        assert engine.main(["--check-stale"], root=root) == 0


class TestStampUnderRecording:
    """An artifact re-rendered while the stamping machinery changed under it carries CURRENT outputs with a STALE
    stamp.

    `stale_output_references` stays silent -- the outputs are genuinely current -- but every op the stamp fails to
    record is invisible to the next rename's drift check. Two correct PRs composed into this on main (one
    re-rendered, one stamped; merged in either order the result is the same), so neither PR's CI could have caught it.
    """

    def test_current_op_in_output_missing_from_stamp_is_detected(self, hooks):
        """The defect observed on main: outputs mention `logit_diffs_latent`, the stamp predates it."""
        artifact = _artifact(recorded=["labels_to_ids", "logit_diffs_sae"], output_text="ran logit_diffs_latent\n")
        current = {"labels_to_ids", "logit_diffs_latent"}
        assert hooks.unstamped_output_references(artifact, current, _text(artifact)) == ["logit_diffs_latent"]

    def test_word_boundary_blocks_substring_accusations(self, hooks):
        """An output mentioning only `logit_diffs_latent` must not read as mentioning `logit_diffs`.

        This direction accuses a STAMP, not the page, so it wants precision where the dead-name direction wants
        sensitivity.
        """
        artifact = _artifact(recorded=["labels_to_ids"], output_text="ran logit_diffs_latent\n")
        assert hooks.unstamped_output_references(artifact, {"labels_to_ids", "logit_diffs"}, _text(artifact)) == []
        artifact = _artifact(recorded=["labels_to_ids"], output_text="ran logit_diffs alone\n")
        assert hooks.unstamped_output_references(artifact, {"labels_to_ids", "logit_diffs"}, _text(artifact)) == [
            "logit_diffs"
        ]

    def test_missing_stamp_is_not_a_stamp_gap(self, hooks):
        """No stamp at all is the check's note, not a stamp gap; accusing an absent stamp would double-count."""
        artifact = _artifact(recorded=None, output_text="ran logit_diffs_latent\n")
        assert hooks.unstamped_output_references(artifact, {"logit_diffs_latent"}, _text(artifact)) == []

    def test_unreadable_surface_reports_nothing(self, hooks):
        """Same contract as the dead-name direction: an unreadable surface must not read as `no ops`."""
        artifact = _artifact(recorded=["labels_to_ids"], output_text="ran labels_to_ids\n")
        assert hooks.unstamped_output_references(artifact, set(), _text(artifact)) == []

    def test_check_hook_routes_each_finding_through_the_engine(self, tmp_path, hooks, monkeypatch, capsys):
        """Through `main`: a dead name and a stamp gap fail the check, an unstamped artifact is only a note."""
        root = _repo(tmp_path, hooks_path=HOOKS_PATH)
        monkeypatch.setattr(engine, "load_hooks", lambda path: hooks.HOOKS)
        monkeypatch.setattr(hooks, "bundled_op_names", lambda: {"live_op", "new_op"})
        cases = {
            "dead": _artifact(recorded=["dead_op", "live_op", "new_op"], output_text="ran dead_op\n"),
            "gap": _artifact(recorded=["live_op"], output_text="ran new_op\n"),
            "unstamped": _artifact(recorded=None, output_text="ran live_op\n"),
        }
        for name, nb in cases.items():
            _write(root / "publish" / "lane" / f"{name}.ipynb", nb)
            _write(root / "artifacts" / "lane" / f"{name}.ipynb", nb)
        assert engine.main(["--check-stale"], root=root) == engine.DRIFT_EXIT_CODE
        captured = capsys.readouterr()
        assert "STALE artifact OUTPUT (re-execute it)" in captured.err and "dead_op" in captured.err
        assert "STALE artifact STAMP" in captured.err and "new_op" in captured.err
        assert "note: no recorded op surface" in captured.err
        assert "2 stale or missing artifact(s); 1 note(s)" in captured.out

    def test_restamp_refuses_an_artifact_with_dead_output_references(self, tmp_path, hooks, monkeypatch, capsys):
        """The refusal is MECHANICAL, not advisory: restamping sets recorded == current, after which `recorded -
        current` is empty forever and the dead-name check can never fire again for that artifact."""
        root = _repo(tmp_path, hooks_path=HOOKS_PATH)
        monkeypatch.setattr(engine, "load_hooks", lambda path: hooks.HOOKS)
        monkeypatch.setattr(hooks, "bundled_op_names", lambda: {"live_op"})
        dead = _artifact(recorded=["dead_op", "live_op"], output_text="ran dead_op\n")
        fixable = _artifact(recorded=["dead_op", "live_op"], output_text="ran live_op\n")
        for name, nb in (("dead.ipynb", dead), ("fixable.ipynb", fixable)):
            _write(root / "publish" / "demo" / name, nb)
            _write(root / "artifacts" / "demo" / name, nb)

        assert engine.main(["--restamp"], root=root) == 1
        err = capsys.readouterr().err
        # `rel` is a native Path, so the separator differs by OS -- assert the native rendering.
        assert f"REFUSED {Path('demo') / 'dead.ipynb'}" in err and "re-execute" in err
        restamped = json.loads((root / "artifacts" / "demo" / "fixable.ipynb").read_text())
        kept = json.loads((root / "artifacts" / "demo" / "dead.ipynb").read_text())
        assert restamped["metadata"][hooks.OP_SURFACE_KEY] == ["live_op"]
        assert "dead_op" in kept["metadata"][hooks.OP_SURFACE_KEY]

    def test_restamp_refuses_to_stamp_an_unreadable_surface(self, tmp_path, hooks, monkeypatch, capsys):
        """An empty stamp would turn every later drift check into a no-op for that file."""
        root = _repo(tmp_path, hooks_path=HOOKS_PATH)
        monkeypatch.setattr(engine, "load_hooks", lambda path: hooks.HOOKS)
        monkeypatch.setattr(hooks, "bundled_op_names", lambda: set())
        nb = _artifact(recorded=["live_op"], output_text="ran live_op\n")
        _write(root / "publish" / "lane" / "nb.ipynb", nb)
        _write(root / "artifacts" / "lane" / "nb.ipynb", nb)
        assert engine.main(["--restamp"], root=root) == 1
        assert "op surface unreadable" in capsys.readouterr().err
        assert json.loads((root / "artifacts" / "lane" / "nb.ipynb").read_text())["metadata"][hooks.OP_SURFACE_KEY] == [
            "live_op"
        ]


class TestHostPathNormalization:
    """An artifact must not publish the identity of whoever rendered it.

    The renderer bakes absolute paths into published docs twice over: papermill records its own input/output paths in
    metadata, and captured warnings carry the file they fired from. Both name the working copy of whoever ran the
    render host that day, which for this project has included scratch worktrees (`.../worktrees/it-stale/...`).
    """

    MARKERS = ("src/it_examples/notebooks/publish/", "docs/notebook_artifacts/")

    def test_rewrites_the_users_root_wherever_it_lives(self):
        """A `/home`-only rule misses most of them, which is how the first version of this passed while leaving 22
        paths in place."""
        rx = engine._user_root_pattern("someuser")
        for original in (
            "/home/someuser/repos/interpretune/src/interpretune/config/shared.py:222: UserWarning",
            "/mnt/cache/someuser/worktrees/it-opdocs/src/interpretune/analysis/core.py",
            "/mnt/cache_extended/someuser/.cache/huggingface/hub",
            "/mnt/cache/someuser/.venvs/it_latest/lib/python3.13/site-packages/x.py",
            '  File "/home/someuser/repos/x/y.py", line 3',
        ):
            assert rx.sub("~/", original) != original, original
            assert "someuser" not in rx.sub("~/", original)

    def test_leaves_hub_and_git_references_alone(self):
        """`owner/repo` names real published repositories the examples depend on; rewriting them would break
        working references in order to fix a cosmetic one."""
        rx = engine._user_root_pattern("someuser")
        for untouched in (
            "Downloading from huggingface.co/someuser/rte to cache",
            "https://huggingface.co/someuser/jlens_steering_ops",
            "see https://github.com/someuser/interpretune/pull/1",
            "resolves to someuser.concept_direction_ops.concept",
            "models--someuser--concept_direction_ops/116271f2",
            "repo_id='someuser/trivial_op_repo'",
        ):
            assert rx.sub("~/", untouched) == untouched, untouched

    def test_a_url_is_not_a_path_even_after_a_double_slash(self):
        """REGRESSION: excluding a preceding word character and colon but not another SLASH let
        `https://github.com/<user>/interpretune` match from the second slash and become `https:/~/interpretune`."""
        rx = engine._user_root_pattern("someuser")
        url = "see https://github.com/someuser/interpretune/pull/1"
        assert rx.sub("~/", url) == url

    def test_papermill_metadata_becomes_repo_relative(self):
        notebook = {
            "cells": [],
            "metadata": {
                "papermill": {
                    "input_path": "/mnt/cache/someuser/worktrees/it-stale/src/it_examples/notebooks/publish/a.ipynb",
                    "output_path": "/home/someuser/repos/interpretune/docs/notebook_artifacts/a.ipynb",
                }
            },
        }
        assert engine.normalize_host_paths(notebook, markers=self.MARKERS, user="someuser") == 2
        pm = notebook["metadata"]["papermill"]
        assert pm["input_path"] == "src/it_examples/notebooks/publish/a.ipynb"
        assert pm["output_path"] == "docs/notebook_artifacts/a.ipynb"

    def test_normalizes_output_text_and_is_idempotent(self):
        """Idempotence is what lets this run on EVERY write, including `--no-execute`, without churn."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "outputs": [{"text": ["/home/someuser/repos/interpretune/src/x.py:9: UserWarning\n"]}],
                }
            ],
            "metadata": {},
        }
        assert engine.normalize_host_paths(notebook, markers=self.MARKERS, user="someuser") == 1
        assert notebook["cells"][0]["outputs"][0]["text"] == ["~/repos/interpretune/src/x.py:9: UserWarning\n"]
        assert engine.normalize_host_paths(notebook, markers=self.MARKERS, user="someuser") == 0

    def test_interpretunes_markers_come_from_its_table(self):
        """The engine holds no repository paths: interpretune's markers are derived from its configured lanes."""
        cfg = engine.DocsConfig.from_pyproject(PROJECT_ROOT)
        assert cfg.path_markers == self.MARKERS


class TestGenericConsumer:
    """A vendoring repository with no hooks gets the generic behavior and nothing of interpretune's."""

    def test_strip_only_render_drops_excluded_cells_and_adds_no_repository_metadata(self, tmp_path):
        root = _repo(tmp_path)
        source = _notebook()
        source["cells"][:0] = [
            {"cell_type": "code", "metadata": {"id": "install-deps"}, "source": ["# %pip install x"], "outputs": []},
            {"cell_type": "code", "metadata": {"tags": ["injected-parameters"]}, "source": ["n = 2"], "outputs": []},
        ]
        _write(root / "publish" / "lane" / "nb.ipynb", source)
        (root / "publish" / "lane" / "asset.png").write_bytes(b"\x89PNG")
        assert engine.main(["--no-execute"], root=root) == 0
        written = json.loads((root / "artifacts" / "lane" / "nb.ipynb").read_text())
        assert [c["source"] for c in written["cells"]] == [["x = 1\n"]]
        assert written["metadata"] == {}, "no hooks, so no stamp"
        assert (root / "artifacts" / "lane" / "asset.png").read_bytes() == b"\x89PNG"
        assert engine.main(["--check-stale"], root=root) == 0

    def test_restamp_without_a_restamp_hook_is_refused_by_name(self, tmp_path, capsys):
        root = _repo(tmp_path)
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        assert engine.main(["--restamp"], root=root) == 1
        assert "needs a docs_hooks file whose HOOKS define restamp" in capsys.readouterr().err

    def test_a_missing_artifact_dir_key_is_refused_by_name(self, tmp_path, capsys):
        root = tmp_path / "bare"
        root.mkdir()
        (root / "pyproject.toml").write_text('[tool.notebook-publishing]\npublish_dir = "publish"\n')
        assert engine.main(["--check-stale"], root=root) == 1
        assert "docs_artifact_dir" in capsys.readouterr().err


class TestHookContract:
    """A hook the engine would never call is refused, or its check would pass forever."""

    @staticmethod
    def _hooks_file(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "hooks.py"
        path.write_text(body)
        return path

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("HOOKS = {'chek': lambda *a, **k: []}\n", "unknown hook(s) ['chek']"),
            ("def check(*a, **k):\n    return []\n", "defines no HOOKS dict"),
            ("HOOKS = {'check': 'not a function'}\n", "are not callable"),
        ],
    )
    def test_malformed_hooks_are_refused_by_name(self, tmp_path, capsys, body, message):
        root = _repo(tmp_path, hooks_path=self._hooks_file(tmp_path, body))
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        assert engine.main(["--check-stale"], root=root) == 1
        assert message in capsys.readouterr().err

    def test_a_hooks_path_that_is_not_a_file_is_refused(self, tmp_path, capsys):
        root = _repo(tmp_path, hooks_path=tmp_path / "absent.py")
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        assert engine.main(["--check-stale"], root=root) == 1
        assert "which is not a file" in capsys.readouterr().err

    def test_hook_findings_fail_or_note_by_kind_and_stamp_runs_on_write(self, tmp_path, capsys):
        body = (
            "def check(artifact, *, rel, output_text):\n"
            "    return [('drift', f'CUSTOM drift: {rel}')] if 'bad' in output_text else [('note', f'note: {rel}')]\n"
            "def stamp(notebook):\n"
            "    notebook.setdefault('metadata', {})['consumer_stamp'] = 1\n"
            "HOOKS = {'check': check, 'stamp': stamp}\n"
        )
        root = _repo(tmp_path, hooks_path=self._hooks_file(tmp_path, body))
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        assert engine.main(["--no-execute"], root=root) == 0
        assert json.loads((root / "artifacts" / "lane" / "nb.ipynb").read_text())["metadata"] == {"consumer_stamp": 1}
        assert engine.main(["--check-stale"], root=root) == 0, "a note alone must not fail the check"
        assert "note: " in capsys.readouterr().err
        _write(root / "artifacts" / "lane" / "nb.ipynb", _notebook(["bad\n"]))
        assert engine.main(["--check-stale"], root=root) == engine.DRIFT_EXIT_CODE
        assert "CUSTOM drift" in capsys.readouterr().err

    def test_a_check_finding_of_unknown_kind_is_refused(self, tmp_path):
        body = "HOOKS = {'check': lambda artifact, *, rel, output_text: [('warning', 'x')]}\n"
        root = _repo(tmp_path, hooks_path=self._hooks_file(tmp_path, body))
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        _write(root / "artifacts" / "lane" / "nb.ipynb", _notebook())
        with pytest.raises(engine.NotebookConfigError, match="kind 'warning'"):
            engine.main(["--check-stale"], root=root)

    def test_interpretunes_hooks_load_through_its_table(self):
        cfg = engine.DocsConfig.from_pyproject(PROJECT_ROOT)
        assert set(cfg.hooks) == set(engine.HOOK_NAMES)


class TestDocsParameters:
    """A consumer supplies papermill parameters per notebook from its table, and the CLI overrides them."""

    def test_table_parameters_reach_papermill_and_the_cli_overrides_them(self, tmp_path, monkeypatch):
        root = _repo(tmp_path, extra='[tool.notebook-publishing.docs_parameters."lane/nb"]\nepochs = 1\nname = "m"\n')
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        _write(root / "publish" / "lane" / "other.ipynb", _notebook())
        seen: dict[str, dict] = {}

        def fake_execute(source, target, timeout, parameters):
            seen[source.stem] = parameters
            _write(target, _notebook(["ran\n"]))

        monkeypatch.setattr(engine, "execute", fake_execute)
        assert engine.main(["--parameters", '{"epochs": 3}'], root=root) == 0
        assert seen == {"nb": {"epochs": 3, "name": "m"}, "other": {"epochs": 3}}

    def test_a_parameter_key_naming_no_notebook_is_refused(self, tmp_path, capsys):
        """A typo would otherwise render the docs with the notebook's defaults and no error anywhere."""
        root = _repo(tmp_path, extra='[tool.notebook-publishing.docs_parameters."lane/nbb"]\nepochs = 1\n')
        _write(root / "publish" / "lane" / "nb.ipynb", _notebook())
        assert engine.main(["--check-stale"], root=root) == 1
        assert "['lane/nbb']" in capsys.readouterr().err
