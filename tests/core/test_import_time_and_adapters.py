import json
import os
import subprocess
import sys

import pytest


def test_import_interpretune_does_not_pull_adapters_and_is_fast():
    """Ensure importing the package doesn't eagerly import heavy adapters and finishes quickly.

    This runs a fresh Python subprocess to avoid contamination from the test runner's imports. The test asserts that
    commonly-heavy adapter packages are NOT in sys.modules after `import interpretune` and that the import completes
    within a configurable threshold (default 6 seconds).

    To avoid CI flakes on slower runners, set the environment variable `IT_ALLOW_SLOW_IMPORT=1` to skip the timing
    assertion (adapter presence is still checked).
    """

    adapters = [
        "transformer_lens",
        "lightning",
        "circuit_tracer",
        "sae_lens",
    ]

    # The small helper script run in a clean subprocess
    check_script = (
        "import time,sys,json;"
        "t0=time.time();"
        "import interpretune;"
        "duration=time.time()-t0;"
        "mods={m: (m in sys.modules) for m in %s};"
        "print(json.dumps({'duration': duration, 'modules': mods}))" % (json.dumps(adapters),)
    )

    result = subprocess.run([sys.executable, "-c", check_script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}\n{result.stdout}"

    payload = json.loads(result.stdout.strip())
    duration = float(payload.get("duration", 9999))
    modules = payload.get("modules", {})

    # Assert adapters were not imported
    imported = [m for m, present in modules.items() if present]
    assert not imported, "Adapters were unexpectedly imported on package import: %s" % imported

    # Allow CI override to avoid flakes on slow runners
    if os.environ.get("IT_ALLOW_SLOW_IMPORT"):
        pytest.skip("Skipping import time assertion because IT_ALLOW_SLOW_IMPORT is set")

    # OS-specific threshold (seconds) - Windows runner's slower, Linux/macOS runners are faster (TODO: analyze diff)
    import platform

    system = platform.system()
    if system == "Windows":
        default_threshold = 8.0
    else:  # Linux, Darwin (macOS), and others
        default_threshold = 3.0

    # Allow override via environment variable
    try:
        threshold = float(os.environ.get("IT_IMPORT_TIME_THRESHOLD_SECONDS", str(default_threshold)))
    except ValueError:
        threshold = default_threshold

    assert duration < threshold, (
        f"interpretune import took too long ({duration:.2f}s) — expected < {threshold:.1f}s on {system}"
    )
