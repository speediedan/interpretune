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
                "lightning": ["bitsandbytes", "peft"]
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

    req_in_path, post_path, platform_path = regen.generate_pip_compile_inputs(pyproject, str(ci_out))

    # requirements.in should be created and should NOT contain datasets, fsspec, or bitsandbytes
    req_in = (ci_out / "requirements.in").read_text()
    assert "packageA" in req_in
    assert "peft" in req_in  # should be included
    assert "datasets" not in req_in  # excluded as post_upgrade
    assert "fsspec" not in req_in    # excluded as post_upgrade
    assert "bitsandbytes" not in req_in  # excluded as platform_dependent

    # post_upgrades.txt should exist and pin the specified versions
    post_text = Path(post_path).read_text()
    assert "datasets==4.0.0" in post_text
    assert "fsspec==2025.3.0" in post_text

    # platform_dependent.txt should exist and contain bitsandbytes
    platform_text = Path(platform_path).read_text()
    assert "bitsandbytes" in platform_text
