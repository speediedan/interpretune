"""Interpretune's hooks for nbpress's docs-artifact renderer, and interpretune's use of the renderer.

The renderer is nbpress's (a git-deps pin) and is tested there; these cases cover ``scripts/notebook_docs_hooks.py``,
interpretune's implementation of its hook interface, the parts of interpretune's configuration it reads, and the
docs workflow that branches on its exit code.

`--check-stale` can see output-only drift, not just source drift: the engine's comparator builds its signature from
``(cell_type, source)``, so output drift is unrepresentable there, and the op-surface hooks exist to close that.
Confirmed in the wild: one rename touched two artifacts, and the gate flagged the one whose cell SOURCE mentioned the
symbol while staying silent on the one that mentioned it only in captured OUTPUT.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from nbpress import render as engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOKS_PATH = PROJECT_ROOT / "scripts" / "notebook_docs_hooks.py"


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
        '[tool.nbpress]\ndev_dir = "dev"\npublish_dir = "publish"\ncolab_repo = "org/consumer"\n'
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

    def test_workflow_branches_on_the_code_the_script_actually_returns(self):
        """The engine and `docs-build.yml` must agree, and nothing else checks that they do."""
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "docs-build.yml").read_text()
        assert f"\n            {engine.DRIFT_EXIT_CODE}) echo " in workflow, (
            f"docs-build.yml has no branch for DRIFT_EXIT_CODE={engine.DRIFT_EXIT_CODE}"
        )
        # The wildcard arm must say plainly that it is NOT a drift report; that sentence is the fix.
        assert "NOT a drift report" in workflow


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


class TestInterpretuneConfiguration:
    """What the renderer reads from interpretune's table, and what it finds in interpretune's artifacts."""

    def test_interpretunes_markers_come_from_its_table(self):
        """The renderer holds no repository paths: interpretune's markers are derived from its configured lanes."""
        cfg = engine.DocsConfig.from_pyproject(PROJECT_ROOT)
        assert cfg.path_markers == ("src/it_examples/notebooks/publish/", "docs/notebook_artifacts/")

    def test_no_artifact_names_a_checkout_path(self):
        """Warning locations once carried whichever working copy rendered an artifact (scratch and agent
        worktrees).

        nbpress writes them repository-relative and its stale check refuses a survivor; this runs that detector over
        interpretune's own artifacts, so a hand-committed or old-renderer artifact fails here as well as in docs-build.
        """
        cfg = engine.DocsConfig.from_pyproject(PROJECT_ROOT)
        found = {
            str(path.relative_to(cfg.artifact_dir)): paths
            for path in sorted(cfg.artifact_dir.rglob("*.ipynb"))
            if (paths := engine.artifact_checkout_paths(engine.load_notebook(path), cfg.root))
        }
        assert not found, f"checkout paths in artifact outputs (run nbpress-render --no-execute): {found}"

    def test_interpretunes_hooks_load_through_its_table(self):
        cfg = engine.DocsConfig.from_pyproject(PROJECT_ROOT)
        assert set(cfg.hooks) == set(engine.HOOK_NAMES)
