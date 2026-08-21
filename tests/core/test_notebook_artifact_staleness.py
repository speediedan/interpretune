"""#300: `--check-stale` can see output-only drift, not just source drift.

The comparator behind the original check builds its signature from ``(cell_type, source)``, so output drift is not
merely unchecked -- it is unrepresentable in the comparison. Confirmed in the wild: one rename touched two artifacts,
and the gate flagged the one whose cell SOURCE mentioned the symbol while staying silent on the one that mentioned it
only in captured OUTPUT.
"""

from __future__ import annotations

import importlib.util
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
        assert renderer.stale_output_references(artifact, set()) == ["sae_correct_acts"]


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
