import json
import os
import statistics
import subprocess
import sys

import pytest


def test_import_interpretune_does_not_pull_adapters_and_is_fast():
    """Ensure importing the package doesn't eagerly import heavy adapters and finishes quickly.

    This runs a fresh Python subprocess to avoid contamination from the test runner's imports. The test asserts that
    commonly-heavy adapter packages are NOT in sys.modules after `import interpretune` and that the import completes
    within a configurable threshold (conservative defaults).

    To avoid CI flakes on slower runners, set the environment variable `IT_ALLOW_SLOW_IMPORT=1` to skip the timing
    assertion (adapter presence is still checked).
    """

    adapters = [
        "transformer_lens",
        "lightning",
        "circuit_tracer",
        "sae_lens",
    ]
    attempts = 3

    # The small helper script run in a clean subprocess
    check_script = (
        "import time,sys,json;"
        "t0=time.time();"
        "import interpretune;"
        "duration=time.time()-t0;"
        "mods={m: (m in sys.modules) for m in %s};"
        "print(json.dumps({'duration': duration, 'modules': mods}))" % (json.dumps(adapters),)
    )

    durations = []
    for _ in range(attempts):
        result = subprocess.run([sys.executable, "-c", check_script], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}\n{result.stdout}"

        payload = json.loads(result.stdout.strip())
        durations.append(float(payload.get("duration", 9999)))
        modules = payload.get("modules", {})

        # Assert adapters were not imported
        imported = [m for m, present in modules.items() if present]
        assert not imported, "Adapters were unexpectedly imported on package import: %s" % imported

    duration = statistics.median(durations)

    # Allow CI override to avoid flakes on slow runners
    if os.environ.get("IT_ALLOW_SLOW_IMPORT"):
        pytest.skip("Skipping import time assertion because IT_ALLOW_SLOW_IMPORT is set")

    # OS-specific threshold (seconds) - Windows runner's slower, Linux/macOS runners are faster (TODO: analyze diff)
    import platform

    system = platform.system()
    if system == "Windows":
        default_threshold = 8.0
    else:  # Linux, Darwin (macOS), and others
        default_threshold = 5.0

    # Allow override via environment variable
    try:
        threshold = float(os.environ.get("IT_IMPORT_TIME_THRESHOLD_SECONDS", str(default_threshold)))
    except ValueError:
        threshold = default_threshold

    assert duration < threshold, (
        f"interpretune median import time across {attempts} subprocesses took too long "
        f"({duration:.2f}s) — expected < {threshold:.1f}s on {system}; raw timings={durations}"
    )


def test_it_hub_resolves_in_fresh_process():
    """`it.hub` must resolve in a process that never imported interpretune.hub another way.

    Regression pin for the lazy-submodule recursion: resolving a direct-submodule attr by importing
    the PARENT with a fromlist re-enters the package __getattr__ for the same name — infinite
    recursion. Every in-repo consumer imported interpretune.hub.* directly first, which masked it;
    the first clean-process `it.hub.push(...)` (the seed publish) hit it. Subprocess-only by
    necessity: an in-process assertion could pass spuriously for exactly the masking reason.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import interpretune as it; assert it.hub.__name__ == 'interpretune.hub'"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_import_interpretune_does_not_import_the_testing_package():
    """`interpretune.testing` ships in the wheel for adopters and must cost nothing by default."""
    import subprocess

    script = (
        "import sys, interpretune; "
        "print(int(any(m == 'interpretune.testing' or m.startswith('interpretune.testing.') for m in sys.modules)))"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "0", "importing interpretune pulled in interpretune.testing"


def test_conformance_package_imports_without_pytest():
    """The conformance package is import-safe without its test-only dependency: pytest is imported lazily by the case
    and plugin modules, which a consumer only reaches from inside a pytest run."""
    import subprocess

    script = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'pytest' or name.startswith('pytest.'):\n"
        "            raise ImportError('pytest blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import interpretune.testing.conformance as c\n"
        "from interpretune.testing.conformance import ConformanceTarget, ConformanceInputs, Gate\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"
