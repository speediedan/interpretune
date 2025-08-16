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
                "examples": ["example_pkg >=0.1", "datasets >= 2.0", "some_transitive_dep"],
                "lightning": ["bitsandbytes", "peft", "finetuning-scheduler >= 2.5.0"]
            },
        },
        "tool": {
            "ci_pinning": {
                "post_upgrades": {"datasets": "4.0.0", "fsspec": "2025.3.0"},
                "platform_dependent": ["bitsandbytes"]
            }
        },
    }

    ci_out = tmp_path / "ci"
    ci_out.mkdir()

    req_in_path, post_path, platform_path, direct_packages = regen.generate_pip_compile_inputs(pyproject, str(ci_out))

    # Check that direct_packages contains the expected packages
    assert "packagea" in direct_packages  # core dependency (normalized to lowercase)
    assert "peft" in direct_packages  # key package from lightning group
    assert "finetuning-scheduler" in direct_packages  # key package from lightning group
    assert "bitsandbytes" in direct_packages  # platform-dependent but still tracked as direct
    # packages excluded due to post_upgrades should not be in direct_packages
    assert "datasets" not in direct_packages
    assert "fsspec" not in direct_packages

    # requirements.in should be created and should contain core deps and key optional deps
    req_in = (ci_out / "requirements.in").read_text()
    assert "packageA" in req_in  # core dependency
    assert "peft" in req_in  # key package from lightning group
    assert "finetuning-scheduler" in req_in  # key package from lightning group
    assert "datasets" not in req_in  # excluded as post_upgrade
    assert "fsspec" not in req_in    # excluded as post_upgrade
    assert "bitsandbytes" not in req_in  # excluded as platform_dependent
    # Non-key packages from optional dependencies should NOT be included
    assert "example_pkg" not in req_in  # not in key packages list
    assert "some_transitive_dep" not in req_in  # not in key packages list

    # post_upgrades.txt should exist and pin the specified versions
    post_text = Path(post_path).read_text()
    assert "datasets==4.0.0" in post_text
    assert "fsspec==2025.3.0" in post_text

    # platform_dependent.txt should exist and contain bitsandbytes
    platform_text = Path(platform_path).read_text()
    assert "bitsandbytes" in platform_text


def test_post_process_pinned_requirements(tmp_path):
    regen = load_regen_module()

    # Create a mock requirements.txt with direct dependencies and transitive dependencies
    requirements_path = tmp_path / "requirements.txt"
    requirements_content = """# This is a generated file
absl-py==2.3.1
    # via transformers
torch==2.8.0
    # via -r requirements.in
transformers==4.55.2
    # via -r requirements.in
peft==0.17.0
    # via -r requirements.in
aiosignal==1.4.0
    # via aiohttp
aiohttp==3.12.15
    # via boostedblob
numpy==1.26.4
    # via torch
"""
    requirements_path.write_text(requirements_content)

    # Create existing platform_dependent.txt with bitsandbytes
    platform_path = tmp_path / "platform_dependent.txt"
    platform_path.write_text("bitsandbytes\n")

    # Platform patterns to match
    platform_patterns = ["bitsandbytes"]

    # Direct packages list (only the packages we explicitly specify)
    direct_packages = ["torch", "transformers", "peft"]

    # Run post-processing
    regen.post_process_pinned_requirements(str(requirements_path), str(platform_path), platform_patterns, direct_packages)

    # Check that only direct dependencies remain in requirements.txt
    requirements_final = requirements_path.read_text()
    assert "torch==2.8.0" in requirements_final  # direct dependency should remain
    assert "transformers==4.55.2" in requirements_final  # direct dependency should remain
    assert "peft==0.17.0" in requirements_final  # direct dependency should remain

    # Transitive dependencies should be removed
    assert "absl-py==2.3.1" not in requirements_final  # transitive dependency should be removed
    assert "aiosignal==1.4.0" not in requirements_final  # transitive dependency should be removed
    assert "aiohttp==3.12.15" not in requirements_final  # transitive dependency should be removed
    assert "numpy==1.26.4" not in requirements_final  # transitive dependency should be removed

    # Check that platform_dependent.txt still contains bitsandbytes
    platform_final = platform_path.read_text()
    assert "bitsandbytes" in platform_final  # existing package should remain
