import subprocess
from pathlib import Path
import sys


def test_publish_force_dry_run_runs_without_error():
    """Running nbpress's publisher over interpretune's notebooks with --force and --dry-run exits 0 and lists
    them."""
    repo_root = Path(__file__).resolve().parents[3]

    # --dry-run writes nothing
    result = subprocess.run(
        [sys.executable, "-m", "nbpress.publish", "--dry-run", "--force"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # the publisher found interpretune's notebooks through its [tool.nbpress] table
    assert "Found" in result.stdout
    assert "Notebooks:" in result.stdout
