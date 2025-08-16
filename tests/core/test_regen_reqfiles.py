from pathlib import Path
import importlib.util


def load_regen_module():
    spec = importlib.util.spec_from_file_location(
        "regen_reqfiles",
        Path(__file__).resolve().parents[2] / "requirements" / "regen_reqfiles.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_pip_compile_inputs_writes_files(tmp_path):
    regen = load_regen_module()

    # build a minimal pyproject dict with dependencies, post_upgrades mapping, and platform_dependent packages
    pyproject = {
        "project": {
            "dependencies": [
                "packageA >=1.0",
                "datasets >= 2.0",
                "fsspec >= 2023.1",
            ],
            "optional-dependencies": {
                "examples": ["example_pkg >=0.1", "datasets >= 2.0"],
                "lightning": ["bitsandbytes", "peft", "nvidia-cublas-cu12", "nvidia-nccl-cu12"]
            },
        },
        "tool": {
            "ci_pinning": {
                "post_upgrades": {"datasets": "4.0.0", "fsspec": "2025.3.0"},
                "platform_dependent": ["bitsandbytes", "nvidia-*"]
            }
        },
    }

    ci_out = tmp_path / "ci"
    ci_out.mkdir()

    req_in_path, post_path, platform_path = regen.generate_pip_compile_inputs(pyproject, str(ci_out))

    # requirements.in should be created and should NOT contain datasets, fsspec, bitsandbytes, or nvidia packages
    req_in = (ci_out / "requirements.in").read_text()
    assert "packageA" in req_in
    assert "peft" in req_in  # should be included
    assert "datasets" not in req_in  # excluded as post_upgrade
    assert "fsspec" not in req_in    # excluded as post_upgrade
    assert "bitsandbytes" not in req_in  # excluded as platform_dependent
    assert "nvidia-cublas-cu12" not in req_in  # excluded as platform_dependent (nvidia-* pattern)
    assert "nvidia-nccl-cu12" not in req_in    # excluded as platform_dependent (nvidia-* pattern)

    # post_upgrades.txt should exist and pin the specified versions
    post_text = Path(post_path).read_text()
    assert "datasets==4.0.0" in post_text
    assert "fsspec==2025.3.0" in post_text

    # platform_dependent.txt should exist and contain bitsandbytes and nvidia packages
    platform_text = Path(platform_path).read_text()
    assert "bitsandbytes" in platform_text
    assert "nvidia-cublas-cu12" in platform_text  # matched by nvidia-* pattern
    assert "nvidia-nccl-cu12" in platform_text    # matched by nvidia-* pattern


def test_post_process_pinned_requirements(tmp_path):
    regen = load_regen_module()

    # Create a mock requirements.txt with pinned nvidia packages
    requirements_path = tmp_path / "requirements.txt"
    requirements_content = """# This is a generated file
absl-py==2.3.1
nvidia-cublas-cu12==12.8.4.1
    # via torch
nvidia-nccl-cu12==2.27.3
    # via torch
torch==2.8.0
    # via transformers
transformers==4.55.2
"""
    requirements_path.write_text(requirements_content)

    # Create existing platform_dependent.txt with bitsandbytes
    platform_path = tmp_path / "platform_dependent.txt"
    platform_path.write_text("bitsandbytes\n")

    # Platform patterns to match
    platform_patterns = ["bitsandbytes", "nvidia-*"]

    # Run post-processing
    regen.post_process_pinned_requirements(str(requirements_path), str(platform_path), platform_patterns)

    # Check that nvidia packages were removed from requirements.txt
    requirements_final = requirements_path.read_text()
    assert "nvidia-cublas-cu12" not in requirements_final
    assert "nvidia-nccl-cu12" not in requirements_final
    assert "torch==2.8.0" in requirements_final  # non-nvidia packages should remain
    assert "transformers==4.55.2" in requirements_final

    # Check that nvidia packages were added to platform_dependent.txt
    platform_final = platform_path.read_text()
    assert "bitsandbytes" in platform_final  # existing package should remain
    assert "nvidia-cublas-cu12" in platform_final  # should be added
    assert "nvidia-nccl-cu12" in platform_final  # should be added
