from __future__ import annotations

from pathlib import Path

import pytest

from it_examples.utils.analysis_injection.config_parser import (
    merge_config_dict,
    parse_config_dict,
)


def test_merge_config_dict_supports_simple_recursive_merge(tmp_path: Path) -> None:
    base = {
        "settings": {
            "log_dir": "/tmp/base",
        },
        "file_hooks": {
            "ap_existing": {
                "file_path": "module.py",
                "regex_pattern": ".*existing",
                "insert_after": True,
                "enable": True,
            }
        },
    }

    override = {
        "settings": {"log_dir": "/tmp/override"},
        "file_hooks": {
            "ap_existing": {
                "enable": False,  # Disable existing hook
            },
            "ap_new": {
                "file_path": "module.py",
                "regex_pattern": ".*new",
                "insert_after": False,
                "enable": True,
            },
        },
    }

    merged = merge_config_dict(base, override)

    assert merged["settings"]["log_dir"] == "/tmp/override"

    # Check existing hook was updated
    existing_hook = merged["file_hooks"]["ap_existing"]
    assert existing_hook["enable"] is False
    assert existing_hook["regex_pattern"] == ".*existing"  # Unchanged

    # Check new hook was added
    new_hook = merged["file_hooks"]["ap_new"]
    assert new_hook["file_path"] == "module.py"
    assert new_hook["regex_pattern"] == ".*new"
    assert new_hook["insert_after"] is False
    assert new_hook["enable"] is True


def test_parse_config_resolves_module_path(tmp_path: Path) -> None:
    module_path = tmp_path / "analysis_points.py"
    module_path.write_text("AP_FUNCTIONS = {}\n")

    raw_config = {
        "settings": {
            "analysis_points_module_path": "./analysis_points.py",
        },
        "file_hooks": {},
    }

    config = parse_config_dict(raw_config, source_path=tmp_path / "config.yaml")
    assert config.analysis_points_module_path == module_path


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"file_hooks": {"missing_file_path": {"regex_pattern": ".*"}}},
        {"file_hooks": {"missing_regex": {"file_path": "test.py"}}},
    ],
)
def test_parse_config_validates_hook_definitions(invalid_config: dict) -> None:
    with pytest.raises(ValueError):
        parse_config_dict(invalid_config)
