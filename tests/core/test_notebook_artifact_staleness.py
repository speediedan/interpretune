"""#300: `--check-stale` can see output-only drift, not just source drift.

The comparator behind the original check builds its signature from ``(cell_type, source)``, so output drift is not
merely unchecked -- it is unrepresentable in the comparison. Confirmed in the wild: one rename touched two artifacts,
and the gate flagged the one whose cell SOURCE mentioned the symbol while staying silent on the one that mentioned it
only in captured OUTPUT.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def renderer():
    """Import ``scripts/render_notebook_docs_artifacts.py`` as a module (it guards its CLI entrypoint)."""
    spec = importlib.util.spec_from_file_location(
        "_it_render_notebook_docs_artifacts", PROJECT_ROOT / "scripts" / "render_notebook_docs_artifacts.py"
    )
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


class TestOutputOnlyDrift:
    def test_dead_op_name_in_output_is_detected(self, renderer):
        """The case that slipped through in the wild: renamed op present ONLY in captured output."""
        artifact = _artifact(recorded=["labels_to_ids", "sae_correct_acts"], output_text="ran sae_correct_acts ok\n")
        assert renderer.stale_output_references(artifact, {"labels_to_ids", "latent_correct_acts"}) == [
            "sae_correct_acts"
        ]

    def test_live_op_names_in_output_are_not_flagged(self, renderer):
        artifact = _artifact(recorded=["labels_to_ids"], output_text="ran labels_to_ids ok\n")
        assert renderer.stale_output_references(artifact, {"labels_to_ids"}) == []

    def test_a_dead_name_absent_from_output_is_not_flagged(self, renderer):
        """Only fires when the dead name is actually present -- a removed op nobody printed is not drift."""
        artifact = _artifact(recorded=["labels_to_ids", "sae_correct_acts"], output_text="nothing relevant\n")
        assert renderer.stale_output_references(artifact, {"labels_to_ids"}) == []

    def test_unstamped_artifact_reports_nothing_rather_than_guessing(self, renderer):
        """Absence of a stamp is not evidence of freshness; the caller reports these separately."""
        artifact = _artifact(recorded=None, output_text="sae_correct_acts\n")
        assert renderer.stale_output_references(artifact, {"labels_to_ids"}) == []

    def test_output_text_is_collected_from_data_mimetypes_too(self, renderer):
        """Rich outputs carry text under `data`, not `text`; missing those would blind the check."""
        artifact = {
            "cells": [{"cell_type": "code", "outputs": [{"data": {"text/plain": ["sae_correct_acts\n"]}}]}],
            "metadata": {"interpretune_op_surface": ["sae_correct_acts"]},
        }
        assert renderer.stale_output_references(artifact, {"labels_to_ids"}) == ["sae_correct_acts"]


class TestOpSurfaceSource:
    def test_op_names_come_from_yaml_not_the_dispatcher(self, renderer):
        """Must be deterministic across sessions: a pulled hub collection must not change the answer."""
        names = renderer.bundled_op_names()
        assert "labels_to_ids" in names and "logit_diffs_base" in names
        assert all("." not in n for n in names), "bundled names are bare; a dotted name implies a hub source"

    def test_stamping_is_idempotent_and_sorted(self, renderer):
        notebook: dict = {"cells": [], "metadata": {}}
        renderer.stamp_op_surface(notebook, {"b", "a"})
        first = notebook["metadata"]["interpretune_op_surface"]
        renderer.stamp_op_surface(notebook, {"b", "a"})
        assert first == notebook["metadata"]["interpretune_op_surface"] == ["a", "b"]


def test_bundled_op_names_degrades_when_yaml_is_absent(renderer, monkeypatch):
    """The op surface must degrade to unavailable, not raise, when pyyaml is missing.

    `docs-build.yml` runs `--check-stale` TWICE, and the first run happens deliberately BEFORE the
    requirements install so it fails fast; the step is commented "cheap, stdlib-only". A third-party
    import there is not a missing dependency to add, it breaks that contract -- which is exactly how
    this surfaced, as a `ModuleNotFoundError` the workflow then reported as "an artifact has drifted".

    `sys.modules["yaml"] = None` is the documented way to force `import yaml` to raise ImportError
    without touching the filesystem or the real module.
    """
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert renderer.bundled_op_names() == set()


def test_yaml_absence_degrades_output_drift_but_keeps_source_drift(renderer, monkeypatch):
    """Degrading must cost ONLY the output-drift half, never the source-drift half.

    Losing source drift too would turn the fail-fast pre-install step into one that passes
    unconditionally -- worse than removing it, because it would still report success.
    """
    monkeypatch.setitem(sys.modules, "yaml", None)
    # An artifact whose OUTPUT names a since-removed op: detectable only via the recorded op surface.
    artifact = {
        "metadata": {renderer.OP_SURFACE_KEY: ["zzz_removed_op"]},
        "cells": [{"cell_type": "code", "source": ["x = 1\n"], "outputs": [{"text": ["ran zzz_removed_op\n"]}]}],
    }
    # With no surface available the comparison cannot be made, so it must report nothing rather than
    # guess. `current_ops` empty is precisely the "unavailable" signal bundled_op_names returns.
    assert renderer.stale_output_references(artifact, set()) == []
    # ...while the source comparator is pure stdlib and stays fully functional.
    assert renderer.artifact_matches_source(artifact, artifact)
    assert not renderer.artifact_matches_source(
        artifact, {"cells": [{"cell_type": "code", "source": ["x = 2\n"]}], "metadata": {}}
    )


class TestDriftExitCodeContract:
    """#311: `docs-build` must be able to tell "an artifact drifted" from "the check failed to run".

    Both exit non-zero, so the previous `|| { echo "...drifted..."; }` wrapper could only ever assert the
    first. It got that wrong in production: a `ModuleNotFoundError` was reported as drift. Rewording cannot
    fix it -- only a distinct exit code lets the caller say WHICH happened -- so the code IS the contract,
    and it spans two files that must agree.
    """

    def test_drift_code_avoids_the_codes_it_must_be_distinguishable_from(self, renderer):
        """0/1/2 are already spoken for, and 2 is the one that is easy to miss.

        1 is an uncaught exception. 2 is argparse's usage-error convention, which this script does not choose and cannot
        override -- argparse exits before `main()` runs. Using 2 for drift would report a renamed or mistyped flag as
        "an artifact has drifted", which is the original defect in a new hat.
        """
        assert renderer.DRIFT_EXIT_CODE not in (0, 1, 2)

    def test_argparse_usage_error_is_not_reported_as_drift(self):
        """The collision above, measured rather than assumed -- it is why the code is 3 and not 2."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "render_notebook_docs_artifacts.py"), "--bogus-flag"],
            capture_output=True,
        )
        assert result.returncode == 2, "argparse's usage convention changed; re-check DRIFT_EXIT_CODE"
        spec = importlib.util.spec_from_file_location(
            "_it_rnda_probe", PROJECT_ROOT / "scripts" / "render_notebook_docs_artifacts.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert result.returncode != mod.DRIFT_EXIT_CODE

    def test_workflow_branches_on_the_code_the_script_actually_returns(self, renderer):
        """The script and `docs-build.yml` must agree, and nothing else checks that they do.

        They are edited independently, so a change to one silently returns the gate to reporting a confident wrong cause
        -- the failure is invisible until a real drift or a real crash arrives, at which point CI names the wrong one.
        Cheap to pin, and it is the only thing tying the two together.
        """
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "docs-build.yml").read_text()
        assert f"\n            {renderer.DRIFT_EXIT_CODE}) echo " in workflow, (
            f"docs-build.yml has no branch for DRIFT_EXIT_CODE={renderer.DRIFT_EXIT_CODE}"
        )
        # The wildcard arm must say plainly that it is NOT a drift report; that sentence is the fix.
        assert "NOT a drift report" in workflow
