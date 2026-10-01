"""Guards for per-change GPU test selection (CPU): every file is classified, every CI GPU test is reachable, and
the selection actually narrows a GPU phase."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gpu_ci.calibrate import collect_gpu_items
from tests.gpu_ci.select_gpu_tests import load_areas, select, selected_prefixes, write_selection_file

ROOT = Path(__file__).resolve().parents[2]
CI_TIERS_EXCLUDED = ("optional", "profiling", "benchmark")  # tiers no PR gate runs (profiling_ci is run)


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_areas()


def _tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # refuse rather than pass on an empty file list
        pytest.fail(f"cannot list tracked files to classify: {e}")
    files = [f for f in out.splitlines() if f]
    assert len(files) > 100, f"git ls-files listed only {len(files)} files; not a full checkout?"
    return files


def test_every_tracked_file_is_classified(spec):
    """An unclassified path forces a full GPU run; this makes the refusal happen when the file is added."""
    sel = select(_tracked(), spec, ROOT)
    unclassified = sorted(p for p, r in sel.reasons.items() if r.startswith("UNCLASSIFIED"))
    assert not unclassified, "classify these in tests/gpu_ci/areas.yaml:\n  " + "\n  ".join(unclassified)


def test_every_ci_gpu_test_is_reachable_from_an_area(spec):
    """A GPU test no area lists runs only when its own file changes or a run_all path does."""
    prefixes = [t for a in spec["areas"].values() for t in a["tests"]]
    orphans = [
        i["nodeid"]
        for i in collect_gpu_items()
        if not any(i.get(t) for t in CI_TIERS_EXCLUDED) and not any(i["nodeid"].startswith(p) for p in prefixes)
    ]
    assert not orphans, "GPU tests no area selects:\n  " + "\n  ".join(orphans)


@pytest.mark.parametrize(
    "path, expect",
    [
        ("pyproject.toml", "full"),
        ("docs/index.md", "none"),
        ("src/interpretune/brand_new_module.py", "full"),  # unclassified: refuse to guess
        ("src/interpretune/analysis/ops/bundled/jlens/jlens_ops.py", {"jlens"}),
        ("src/interpretune/adapters/circuit_tracer/adapter.py", {"circuit_tracer"}),
        ("tests/core/test_backend_conformance.py", "self"),
    ],
)
def test_each_rule_class_selects_what_it_should(spec, path, expect):
    sel = select([path], spec, ROOT)
    if expect == "full":
        assert sel.full and selected_prefixes(sel, spec) is None
    elif expect == "none":
        assert sel.empty
    elif expect == "self":
        assert sel.test_modules == {path} and not sel.full
    else:
        assert expect <= sel.areas and not sel.full


def test_a_support_module_selects_the_tests_importing_it(spec):
    sel = select(["tests/core/circuit_tracer_toy.py"], spec, ROOT)
    assert "tests/core/test_analysis_backend_parity.py" in sel.test_modules and not sel.full


def test_the_selection_file_narrows_a_gpu_phase(tmp_path, spec):
    """Positive control on the conftest filter: a one-area selection collects only that area's GPU tests."""
    sel = select(["src/interpretune/adapters/transformer_lens/adapter.py"], spec, ROOT)
    narrow = tmp_path / "sel.txt"
    write_selection_file(sel, spec, narrow)

    def collected(selection: Path | None) -> set[str]:
        env = {**os.environ, "IT_RUN_CUDA_TESTS": "1", "CUDA_VISIBLE_DEVICES": ""}
        env.pop("IT_GPU_SELECTION_FILE", None)
        if selection is not None:
            env["IT_GPU_SELECTION_FILE"] = str(selection)
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "src/it_examples/tests",
                "-q",
                "--collect-only",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        ).stdout
        return {line.strip() for line in out.splitlines() if "::" in line}

    full, narrowed = collected(None), collected(narrow)
    assert narrowed and narrowed < full, (len(narrowed), len(full))
    prefixes = selected_prefixes(sel, spec)
    assert all(any(n.startswith(p) for p in prefixes) for n in narrowed)
    full_file = tmp_path / "full.txt"
    full_file.write_text("full\n")
    assert collected(full_file) == full
