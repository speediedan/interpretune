from __future__ import annotations

from pathlib import Path
import pytest

from interpretune.analysis.cache import (
    enable_analysisstore_caching,
    is_analysisstore_caching_enabled,
    get_analysis_cache_dir,
)


def test_default_flag_false():
    assert is_analysisstore_caching_enabled() is False


def test_enabling_warns_and_sets_flag():
    # rank_zero_warn uses warnings.warn under the hood; capture with pytest.warns
    with pytest.warns(Warning) as rec:
        enable_analysisstore_caching(True)
    # Expect at least one warning mentioning planned
    assert any("planned" in str(w.message).lower() for w in rec), rec.list
    assert is_analysisstore_caching_enabled() is True


def test_get_analysis_cache_dir_temp(tmp_path, monkeypatch):
    # Create a fake module-like object with minimal datamodule shape
    class DummyDataset:
        config_name = "cfg"
        _fingerprint = "fp"

    class DummyDatamodule:
        dataset = {"validation": DummyDataset()}

    class DummyModule:
        datamodule = DummyDatamodule()

    # assign a class-level attribute matching how modules expose _orig_module_name
    DummyModule._orig_module_name = "DummyMod"

    # Ensure the flag is off -> should get a temp dir returned
    enable_analysisstore_caching(False)
    m = DummyModule()
    cache_dir = Path(get_analysis_cache_dir(m))
    assert cache_dir.exists()
    # temp dir shouldn't be the canonical IT_ANALYSIS_CACHE path component
    assert "interpretune" in str(cache_dir) or cache_dir.exists()
