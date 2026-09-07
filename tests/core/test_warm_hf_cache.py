"""The warm step that lets hosted CI run the suite offline (see docs/ci_hub_cache.md)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "warm_hf_cache.py"
MANIFEST = REPO_ROOT / "tests" / "hf_warm_manifest.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location("warm_hf_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestManifest:
    def test_the_committed_manifest_parses_and_names_what_the_suite_loads(self):
        manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        assert isinstance(manifest.get("cache_version"), int)
        model_ids = {entry["repo_id"] for entry in manifest["models"]}
        assert {"gpt2", "openai-community/gpt2"} <= model_ids, "both gpt2 spellings cache separately"
        datasets = {(entry["path"], entry.get("config_name")) for entry in manifest["datasets"]}
        assert ("aps/super_glue", "rte") in datasets


class TestWarmScript:
    def test_dry_run_plans_every_entry_without_fetching(self, monkeypatch, capsys):
        script = _load_script()
        # Any real fetch would be a defect in dry-run mode; make one fail loudly.
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        assert script.main(["--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "model gpt2" in out and "model openai-community/gpt2" in out
        assert "dataset aps/super_glue (config=rte" in out

    def test_refuses_to_run_with_the_hub_client_offline(self, monkeypatch):
        script = _load_script()
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(SystemExit, match="HF_HUB_OFFLINE"):
            script.main(["--dry-run"])

    def test_retry_backs_off_then_reraises_the_last_error(self, monkeypatch):
        script = _load_script()
        sleeps: list[float] = []
        monkeypatch.setattr(script.time, "sleep", sleeps.append)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("429 rate limited")

        script._retry("thing", flaky, attempts=4)
        assert calls["n"] == 3 and sleeps == [20.0, 40.0]

        def broken():
            raise ConnectionError("still 429")

        with pytest.raises(ConnectionError, match="still 429"):
            script._retry("thing", broken, attempts=2)


class TestOfflineSkip:
    """The conftest skips hf_live tests under HF_HUB_OFFLINE=1; a probe marked hf_live proves it in a
    subprocess."""

    @pytest.mark.hf_live
    def test_probe_must_not_run_offline(self):
        assert os.environ.get("HF_HUB_OFFLINE", "") not in {"1", "true"}, "an hf_live test ran under HF_HUB_OFFLINE=1"

    def test_hf_live_tests_skip_when_the_hub_client_is_offline(self):
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        probe = f"{Path(__file__).as_posix()}::TestOfflineSkip::test_probe_must_not_run_offline"
        result = subprocess.run(
            [sys.executable, "-m", "pytest", probe, "-p", "no:cacheprovider", "-rs", "-q"],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert "1 skipped" in result.stdout, result.stdout[-1500:] + result.stderr[-500:]
        assert "HF_HUB_OFFLINE=1 is set" in result.stdout


class TestOfflineMissIsLegible:
    def test_the_report_section_names_the_manifest(self, monkeypatch):
        from tests import conftest as suite_conftest

        try:
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError:  # pragma: no cover - the suite always has the Hub client
            pytest.skip("huggingface_hub not installed")
        inner = LocalEntryNotFoundError("Cannot find the requested files in the disk cache")
        outer = OSError("We couldn't connect to the Hub")
        outer.__cause__ = inner
        assert suite_conftest._is_offline_cache_miss(outer)
        assert not suite_conftest._is_offline_cache_miss(ValueError("unrelated"))
        assert "tests/hf_warm_manifest.yaml" in suite_conftest._OFFLINE_MISS_HINT
