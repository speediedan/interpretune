"""Static checks for the component-gated interp-engine notebook example.

**Strictly static, and that is a posture decision rather than a maintenance one.** The interp-engine
adapter is hub-delivered and loads behind the ``IT_TRUST_REMOTE_CODE`` gate (#255). A core test that
EXECUTED this notebook would have to set that gate and pull the component, so interpretune's own suite
would routinely run code published from a repo it does not control -- and core's green would depend on
that component author's published revision. Core's tests habitually opting out of the trust gate would
undercut the argument for the gate.

So core owns notebook FORM (published copy in sync, parses, the delivery form it demonstrates) and the
component repo owns EXECUTION correctness, on GPU hardware, where the engine pin and the device live.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
NOTEBOOK_SUBPATH = ("interp_engine_example", "interp_engine_hub_adapter.ipynb")
NOTEBOOKS_ROOT = REPO_ROOT / "src" / "it_examples" / "notebooks"
# Built from PARTS, not by string-replacing "/dev/". That replace is a no-op on Windows, where the
# separator is a backslash, and it fails silently in the worst way: PUBLISHED_NOTEBOOK becomes the DEV
# path, so `test_dev_and_published_copies_both_exist` passes VACUOUSLY (the dev file exists, and it was
# asked about twice) while only the parts assertion below notices. Caught by Windows CI, which is the
# only runner where the two paths differ.
DEV_NOTEBOOK = NOTEBOOKS_ROOT.joinpath("dev", *NOTEBOOK_SUBPATH)
PUBLISHED_NOTEBOOK = NOTEBOOKS_ROOT.joinpath("publish", *NOTEBOOK_SUBPATH)


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cells"]


def _code_source(path: Path) -> str:
    return "\n".join("".join(c["source"]) for c in _cells(path) if c["cell_type"] == "code")


class TestNotebookForm:
    def test_dev_and_published_copies_both_exist(self):
        """Notebook tests run against PUBLISHED notebooks, so a dev-only notebook is invisible to them."""
        # The control: these must be DIFFERENT paths. If the published path were ever derived in a way
        # that collapsed onto the dev path, this test would assert the same file exists twice and pass
        # while checking nothing -- which is exactly what happened on Windows before the paths were
        # built from parts.
        assert DEV_NOTEBOOK != PUBLISHED_NOTEBOOK
        assert DEV_NOTEBOOK.is_file(), f"missing dev notebook: {DEV_NOTEBOOK}"
        assert PUBLISHED_NOTEBOOK.is_file(), (
            f"missing published copy: {PUBLISHED_NOTEBOOK}. Run `python scripts/publish_notebooks.py --force`."
        )

    def test_both_parse_as_notebooks(self):
        for path in (DEV_NOTEBOOK, PUBLISHED_NOTEBOOK):
            payload = json.loads(path.read_text())
            assert payload["nbformat"] == 4
            assert payload["cells"], f"{path} has no cells"

    def test_it_carries_a_papermill_parameters_cell(self):
        """Without it the notebook cannot be parameterized, which is how every other example is driven."""
        tagged = [c for c in _cells(DEV_NOTEBOOK) if "parameters" in (c.get("metadata", {}).get("tags") or [])]
        assert len(tagged) == 1, "expected exactly one cell tagged `parameters`"


class TestItDemonstratesTheHubDeliveryForm:
    """The notebook exists to show HUB delivery, so the import form is the substance, not style."""

    def test_it_does_not_import_the_component_directly(self):
        """A direct import is the pip-install form, which any package could demonstrate.

        The point of this example is the component arriving through the rails, so reaching for
        ``import interp_engine_adapter`` would quietly replace what it is demonstrating.
        """
        source = _code_source(DEV_NOTEBOOK)
        assert "import interp_engine_adapter" not in source
        assert "from interp_engine_adapter" not in source

    def test_it_reaches_the_component_through_the_rails(self):
        source = _code_source(DEV_NOTEBOOK)
        for expected in ("interpretune.hub", "load_hub_adapter", "loaded_adapter_module"):
            assert expected in source, f"the notebook should reach the component via {expected}"

    def test_it_opts_into_the_trust_gate_explicitly(self):
        """The gate refuses by default; opting in should be visible in the notebook rather than implied."""
        assert "IT_TRUST_REMOTE_CODE" in _code_source(DEV_NOTEBOOK)

    def test_it_uses_the_async_capture_path(self):
        """The sync surface refuses inside a live event loop, and Jupyter always has one.

        A notebook that used the sync path would fail at runtime for a reason unrelated to what it teaches, and reaching
        for nest_asyncio instead would hide why the sync API refused.
        """
        source = _code_source(DEV_NOTEBOOK)
        assert "capture_async" in source
        assert "nest_asyncio" not in source


class TestPackaging:
    """The component repo's CI receives this notebook with its interpretune install.

    Notebooks reach the wheel via ``include-package-data`` plus MANIFEST's recursive-include, NOT via a
    ``package-data`` glob -- ``[tool.setuptools.package-data]`` lists only ``*.yaml`` for it_examples,
    and MANIFEST's comment says notebooks are kept "in the sdist for now". So wheel inclusion is true by
    accident and documented as false, while another repo's CI depends on it. Asserted BY PATH rather
    than by count, because the failure mode is a glob change dropping one file.
    """

    def test_the_notebook_is_not_excluded_from_distribution(self):
        manifest = (REPO_ROOT / "MANIFEST.in").read_text()
        for line in manifest.splitlines():
            stripped = line.strip()
            if stripped.startswith(("prune ", "exclude ", "recursive-exclude ")):
                assert "interp_engine_example" not in stripped, f"MANIFEST.in excludes this notebook: {stripped}"
                assert "notebooks" not in stripped.split()[1:2], f"MANIFEST.in prunes notebooks: {stripped}"

    def test_the_published_copy_is_the_one_a_consumer_gets(self):
        """The consumer's pipeline runs the PUBLISHED notebook, so that is the path to pin."""
        assert PUBLISHED_NOTEBOOK.is_file()
        assert "publish" in PUBLISHED_NOTEBOOK.parts


class TestComponentGating:
    """Present by default, inert without the component.

    No flag, per the maintainer.
    """

    def test_it_is_inert_without_the_component_installed(self):
        """Nothing in core imports the notebook, so its presence costs nothing when the component is absent.

        This is the whole reason the example can ship by default rather than behind a flag: an unexecuted
        notebook has no import-time surface at all.
        """
        assert not any("interp_engine" in str(p) for p in (REPO_ROOT / "src" / "interpretune").rglob("*.py")), (
            "core interpretune should not reference the interp-engine example"
        )

    @pytest.mark.skipif(
        not (REPO_ROOT / "src" / "it_examples" / "experiments" / "interp_engine").is_dir(),
        reason="the CLI demo this notebook mirrors is absent",
    )
    def test_it_mirrors_the_cli_demo_rather_than_diverging(self):
        """Both forms should teach the same three points; a notebook that drifts teaches something else."""
        cli = (REPO_ROOT / "src/it_examples/experiments/interp_engine/hub_adapter_demo.py").read_text()
        notebook = _code_source(DEV_NOTEBOOK) + "\n".join(
            "".join(c["source"]) for c in _cells(DEV_NOTEBOOK) if c["cell_type"] == "markdown"
        )
        for shared in ("hook_mlp_in", "mlp.hook_in", "pre_feedforward_layernorm"):
            assert shared in cli and shared in notebook, f"{shared} should appear in both forms"
