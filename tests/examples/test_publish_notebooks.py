import subprocess
from pathlib import Path
import sys


def test_publish_force_dry_run_runs_without_error():
    """Running the publisher with --force and --dry-run should exit 0 and print results."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "publish_notebooks.py"

    # Call the script with --force --dry-run to avoid writing files
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--force"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    # Script should exit successfully
    assert result.returncode == 0

    # Output should indicate the script saw files to process
    assert "Found" in result.stdout
    assert "Notebooks:" in result.stdout
