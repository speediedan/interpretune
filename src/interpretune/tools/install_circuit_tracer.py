"""Circuit-tracer installation utilities for interpretune."""

import argparse
import os
import subprocess
from pathlib import Path


def install_circuit_tracer(use_commit_pin: bool = True, verbose: bool = False):
    """Install circuit-tracer with appropriate dependencies."""

    # Inline the utility functions to avoid importing interpretune package
    def should_use_commit_pin():
        """Check if we should use commit pin based on environment variable."""
        return os.getenv("IT_USE_CT_COMMIT_PIN", "").lower() in ["true", "1", "yes"]

    def get_circuit_tracer_commit():
        """Get the pinned circuit-tracer commit hash."""
        repo_root = Path(__file__).parent.parent.parent.parent
        pin_file = repo_root / "requirements" / "ci" / "circuit_tracer_pin.txt"

        if pin_file.exists():
            with open(pin_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
        return "75fd21f666cbdaece57eca561b4342ad84746b40"  # fallback

    # Determine installation mode
    if use_commit_pin is None:
        use_commit_pin = should_use_commit_pin()

    print(f"Installing circuit-tracer (commit_pin={use_commit_pin})...")

    # Build requirements list
    requirements = []

    # Add circuit-tracer itself
    if use_commit_pin:
        commit_hash = get_circuit_tracer_commit()
        circuit_tracer_url = f"git+https://github.com/speediedan/circuit-tracer.git@{commit_hash}"
        requirements.append(f"{circuit_tracer_url}")
    else:
        requirements.append("circuit-tracer")

    if not requirements:
        print("No circuit-tracer requirements to install.")
        return

    # Install each requirement
    for req in requirements:
        # Split requirement and any flags
        parts = req.split()
        base_req = parts[0]
        flags = parts[1:] if len(parts) > 1 else []

        cmd = ["pip", "install", base_req] + flags

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)
            if verbose and result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {base_req}: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            return False

    print("✅ Circuit-tracer installation completed successfully!")
    return True


def main():
    """CLI entry point for circuit-tracer installation."""
    parser = argparse.ArgumentParser(description="Install circuit-tracer for interpretune examples")
    parser.add_argument("--ct-commit-pin", action="store_true", help="To use commit-based installation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # If --ct-commit-pin is provided, we enable commit pin usage
    use_commit_pin = bool(getattr(args, "ct_commit_pin", False))

    # Set environment variable for consistency
    if use_commit_pin:
        os.environ["IT_USE_CT_COMMIT_PIN"] = "1"
    else:
        os.environ["IT_USE_CT_COMMIT_PIN"] = "0"

    success = install_circuit_tracer(use_commit_pin, args.verbose)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
