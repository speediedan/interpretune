"""``hub://`` references to a cached component's files, and a CLI session based on a cached component configuration
(#847).

The ``rte`` component is published locally into a throwaway cache, which the resolver (cache-only) and the CLI's
``session_ref`` read through the components module's default cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jsonargparse import Namespace

from interpretune.base.components.cli import ITSessionMixin
from interpretune.hub import components
from interpretune.utils import MisconfigurationException

RTE_COMPONENT_DIR = Path(__file__).resolve().parents[2] / "src" / "it_examples" / "examples" / "rte"
SCHEDULE_REF = "hub://speediedan/rte/ft_schedules/rte_ft.gpt2.yaml"


@pytest.fixture
def rte_cache(tmp_path, monkeypatch) -> Path:
    from tests.rte_component import rte_entrypoint_src

    entrypoint = rte_entrypoint_src()  # from the warmed default cache, before it is swapped out
    cache = tmp_path / "components"
    monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", cache)
    components.local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=entrypoint, cache_dir=cache)
    return cache


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("hub://org/repo/a/b.yaml", ("org/repo", None, "a/b.yaml")),
        ("hub://org/repo@abc123/b.yaml", ("org/repo", "abc123", "b.yaml")),
    ],
)
def test_a_hub_file_ref_parses(ref, expected):
    assert components.parse_hub_file_ref(ref) == expected


@pytest.mark.parametrize("ref", ["hub://org/repo", "hub://org//b.yaml", "org/repo/b.yaml"])
def test_a_malformed_hub_file_ref_is_refused(ref):
    with pytest.raises(ValueError, match="reference"):
        components.parse_hub_file_ref(ref)


def test_extra_files_publish_and_resolve(rte_cache):
    path = components.resolve_component_file(SCHEDULE_REF)
    assert path.is_file() and path.parts[-2:] == ("ft_schedules", "rte_ft.gpt2.yaml")
    assert components.resolve_component_file("hub://speediedan/rte/runs/rte_ft.gpt2.lightning.yaml").is_file()


def test_a_missing_file_names_the_fetch(rte_cache):
    with pytest.raises(components.ComponentFileNotCachedError, match=r"interpretune\.hub\.pull\('speediedan/rte'\)"):
        components.resolve_component_file("hub://speediedan/rte/ft_schedules/absent.yaml")


def test_a_path_leaving_the_snapshot_is_refused(rte_cache):
    with pytest.raises(ValueError, match="outside"):
        components.resolve_component_file("hub://speediedan/rte/../../../etc/passwd")


def test_cli_arguments_resolve_hub_refs(rte_cache):
    resolved = str(components.resolve_component_file(SCHEDULE_REF))
    args = ["fit", "--config", SCHEDULE_REF, f"--config={SCHEDULE_REF}", "--trainer.max_epochs=2"]
    assert ITSessionMixin._resolve_hub_args(args) == ["fit", "--config", resolved, f"--config={resolved}", args[-1]]
    unchanged = ["--config", "plain.yaml"]
    assert ITSessionMixin._resolve_hub_args(unchanged) is unchanged


def test_parsed_values_resolve_hub_refs(rte_cache):
    resolved = str(components.resolve_component_file(SCHEDULE_REF))
    callback = {"class_path": "x.Y", "init_args": {"schedule": SCHEDULE_REF, "other": "hub-ish"}}
    config = Namespace(run_cfg=Namespace(callbacks=[callback]), name="plain")
    ITSessionMixin._resolve_hub_refs(config)
    assert config.run_cfg.callbacks[0]["init_args"] == {"schedule": resolved, "other": "hub-ish"}
    assert config.name == "plain"


def test_a_session_ref_seeds_the_session_and_files_merge_over_it(rte_cache, tmp_path):
    import yaml

    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"registered_cfg": {"datamodule_cfg": {"train_batch_size": 2}}}))
    mapping = ITSessionMixin()._session_mapping_from_sources([overlay], None, "speediedan/rte:rte_ft.gpt2.core")
    assert set(mapping) == {"shared_config", "registered_cfg"}
    assert mapping["shared_config"]["model_name_or_path"] == "gpt2"
    dm_cfg = mapping["registered_cfg"]["datamodule_cfg"]
    assert dm_cfg["train_batch_size"] == 2 and dm_cfg["eval_batch_size"] == 4


@pytest.mark.parametrize("ref", ["speediedan/rte", ":rte_ft.gpt2.core", "speediedan/rte:"])
def test_a_malformed_session_ref_is_refused(ref):
    with pytest.raises(MisconfigurationException, match="session_ref"):
        ITSessionMixin._hub_session_mapping(ref)


def test_an_undeclared_session_ref_names_the_available_configurations(rte_cache):
    with pytest.raises(KeyError, match="rte_ft.gpt2.core"):
        ITSessionMixin._hub_session_mapping("speediedan/rte:rte_ft.absent.core")


def test_extra_files_are_pulled_at_the_pinned_commit(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        (tmp_path / "runs").mkdir()
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    manifest = {"extra_files": ["ft_schedules/", "runs"]}
    assert components.pull_component_extra_files("org/repo", manifest, "abc123") == [tmp_path / "runs"]
    ((repo_id, kwargs),) = calls
    assert repo_id == "org/repo" and kwargs["revision"] == "abc123"
    assert kwargs["allow_patterns"] == ["ft_schedules", "ft_schedules/**", "runs", "runs/**"]
    assert components.pull_component_extra_files("org/repo", {}, "abc123") == [] and len(calls) == 1
