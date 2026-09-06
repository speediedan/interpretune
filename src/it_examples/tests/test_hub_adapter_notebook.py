"""Static invariants for the hub-delivered adapter notebook (N6).

These run without a kernel, a GPU, a Hub token or a local Neuronpedia. That is deliberate: the notebook's *execution* is
covered by the opt-in contract lane, but the properties asserted here are the ones a reader relies on before running
anything, and they are the ones a well-meaning edit silently breaks.

**Every check locates its anchor first and fails when the anchor is missing.** A structural assertion that searches for
a pattern and asserts something about what it finds passes trivially once the notebook is restructured and it finds
nothing -- so "the notebook was reorganised" and "the invariant holds" would otherwise be indistinguishable. That
failure mode is the reason this module exists in the shape it does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_REL = "interp_engine_example/interp_engine_hub_adapter.ipynb"
PUBLISH_DIR = Path(__file__).parent.parent / "notebooks" / "publish"
DEV_DIR = Path(__file__).parent.parent / "notebooks" / "dev"


def _cells(notebook: Path) -> list[dict]:
    assert notebook.exists(), f"notebook not found: {notebook} — was it published?"
    return json.loads(notebook.read_text(encoding="utf-8"))["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _find_cell(cells: list[dict], needle: str, cell_type: str = "code") -> tuple[int, str]:
    """Return (index, source) of the first ``cell_type`` cell containing ``needle``, failing if there is none.

    The failure is the point: it converts "the notebook no longer has this section" from a silent pass
    into a named failure.

    ``cell_type`` defaults to code and is not incidental. The prose cells describe the same identifiers the
    code cells use, so an unfiltered search matches the markdown that *explains* a call rather than the call
    itself -- which is how the first version of this module asserted an ordering over a paragraph.
    """
    for i, cell in enumerate(cells):
        if cell["cell_type"] != cell_type:
            continue
        src = _source(cell)
        if needle in src:
            return i, src
    raise AssertionError(f"no {cell_type} cell contains {needle!r} — the notebook's structure changed under this test")


@pytest.fixture(scope="module")
def dev_cells() -> list[dict]:
    return _cells(DEV_DIR / NOTEBOOK_REL)


class TestHubAdapterNotebookIsPublished:
    def test_the_dev_notebook_has_a_published_counterpart(self):
        """A dev-only notebook is invisible to every notebook test, which run against ``publish/``."""
        assert (DEV_DIR / NOTEBOOK_REL).exists(), "dev notebook missing"
        assert (PUBLISH_DIR / NOTEBOOK_REL).exists(), (
            f"{NOTEBOOK_REL} is not published; run `interpretune-publish-notebooks` — notebook tests read "
            "the published copies, so an unpublished notebook is silently untested rather than failing"
        )


class TestTheDeliveryContractIsVisibleToAReader:
    """What the notebook promises about fetching and executing third-party code."""

    def test_the_revision_is_pinned_rather_than_floating(self, dev_cells):
        """A floating revision defeats the argument the notebook makes for pinning one."""
        _, params = _find_cell(dev_cells, "ADAPTER_REVISION")
        line = next(ln for ln in params.splitlines() if ln.strip().startswith("ADAPTER_REVISION"))
        value = line.split("=", 1)[1].split("#")[0].strip()
        assert value not in ("None", '""', "''"), (
            f"ADAPTER_REVISION is {value} — the notebook tells the reader that pinning is what makes the "
            "code they read the code they run, so shipping it unpinned contradicts its own text"
        )
        assert value.startswith(('"', "'")) and len(value.strip("\"'")) >= 8, (
            f"ADAPTER_REVISION={value} is not a concrete revision"
        )

    def test_trust_is_not_granted_on_the_readers_behalf(self, dev_cells):
        """Defaulting the opt-in to True would make the trust step invisible, which is the one step that should not
        be."""
        _, params = _find_cell(dev_cells, "TRUST_HUB_ADAPTER_CODE")
        line = next(ln for ln in params.splitlines() if ln.strip().startswith("TRUST_HUB_ADAPTER_CODE"))
        assert line.split("=", 1)[1].split("#")[0].strip() == "False", (
            "TRUST_HUB_ADAPTER_CODE must default False: the notebook executes code published by a third-party "
            "repository, and opting in for the reader removes the decision it exists to present"
        )

    def test_the_trust_gate_is_reachable_before_any_code_executes(self, dev_cells):
        """Fetch must precede trust, so a reader can read the entrypoint before deciding.

        Checked against the parsed cell rather than its text. A substring search for the flag name passes even when the
        gate has been disabled, because the name survives in the message the gate raises -- measured, on the first
        version of this test.
        """
        import ast

        _, hub = _find_cell(dev_cells, "load_hub_adapter")
        tree = ast.parse(hub)

        def _guards_on_flag(node: ast.If) -> bool:
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            exits = any(isinstance(s, (ast.Raise,)) for s in ast.walk(node))
            return "TRUST_HUB_ADAPTER_CODE" in names and exits

        guards = [n for n in ast.walk(tree) if isinstance(n, ast.If) and _guards_on_flag(n)]
        assert guards, (
            "no conditional on TRUST_HUB_ADAPTER_CODE raises before the component loads — the notebook "
            "would execute third-party code without the reader having opted in"
        )
        assert hub.index("it.hub.pull") < hub.index("load_hub_adapter"), (
            "the hub cell must fetch before it registers, so the reader can inspect the cached entrypoint "
            "before deciding to execute it"
        )


class TestTheCompositionDependsOnDelivery:
    def test_delivery_precedes_session_construction(self, dev_cells):
        """``it.Adapter.interp_engine`` does not exist until the component registers."""
        hub_idx, _ = _find_cell(dev_cells, "load_hub_adapter")
        session_idx, _ = _find_cell(dev_cells, "ITSession(session_cfg)")
        assert hub_idx < session_idx, (
            "session construction precedes hub delivery; the adapter enum member it composes with is added "
            "by the delivery cell, so this ordering cannot work"
        )

    def test_the_composition_names_the_hub_delivered_adapter(self, dev_cells):
        _, session = _find_cell(dev_cells, "adapter_ctx = ")
        assert "it.Adapter.interp_engine" in session, (
            "the composition no longer names interp_engine — this notebook's whole subject is a hub-delivered "
            "adapter composing like a bundled one"
        )
        assert "it.Adapter.circuit_tracer" in session, (
            "the (core, interp_engine, circuit_tracer) composition is the subject"
        )


class TestDeferredWorkSaysWhy:
    def test_the_jspace_section_is_off_and_names_its_reason(self, dev_cells):
        """Section 4b is deferred for a reason specific to this notebook, not the sibling's reachability one.

        Asserted because a bare ``False`` reads as an oversight, and the next person to see it will either
        turn it on without knowing what is unsettled or delete the section as dead weight.
        """
        _, params = _find_cell(dev_cells, "RUN_JSPACE_SECTION")
        line = next(ln for ln in params.splitlines() if ln.strip().startswith("RUN_JSPACE_SECTION"))
        assert line.split("=", 1)[1].split("#")[0].strip() == "False"
        # The comment immediately preceding the flag, not the whole cell: "interp-engine" also occurs in the
        # component repo id a few lines up, so a cell-wide search passes even once the reason is deleted --
        # measured, on the first version of this test.
        import itertools

        preamble = params[: params.index("RUN_JSPACE_SECTION")]
        # takewhile, not a filter: filtering keeps every comment line in the cell, so the assertion below
        # was satisfied by an unrelated comment mentioning the same words. Only the contiguous block
        # directly above the flag is the reason for the flag.
        trailing_comment = "\n".join(
            itertools.takewhile(
                lambda ln: ln.strip().startswith("#") or not ln.strip(), reversed(preamble.splitlines())
            )
        )
        assert "modes" in trailing_comment and "interp-engine" in trailing_comment, (
            "RUN_JSPACE_SECTION is disabled without stating why in this notebook's terms, in the comment "
            "directly above it — the reason here is the unsettled intervention-mode question for the "
            "interp-engine backend, not the sibling notebook's collection reachability"
        )
