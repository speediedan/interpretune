"""Hub adapter components named by a run configuration's ``hub_adapters`` (#872): pinned entries, the trust gate
before any download, a pull only on a cache miss, the run's provenance, and the CLI loading them before it parses.

Tested against the trivial fixture component of ``test_hub_adapters.py``, published locally into a throwaway cache.
"""

from __future__ import annotations

import pytest
import yaml

from tests.core.test_hub_adapters import FIXTURE_ADAPTER, REGISTERS_DECLARED, _write_component
from tests.core.test_hub_adapters import restore_adapter_enum  # noqa: F401  (fixture, used by name)

REPO = "org/fixture-adapter"


@pytest.fixture()
def published(tmp_path, monkeypatch, restore_adapter_enum):  # noqa: F811
    """A fixture adapter component published locally, the cache it lives in, and its revision."""
    from interpretune.hub import components

    component = _write_component(
        tmp_path, REGISTERS_DECLARED + "\n    class FixtureAdapterConfig:\n        stable = True\n"
    )
    cache = tmp_path / "components"
    monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", cache)
    revision = components.local_publish(component, REPO, cache_dir=cache)
    return cache, revision


@pytest.fixture()
def trusted(monkeypatch):
    from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

    monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")


@pytest.fixture()
def registry(monkeypatch):
    """A fresh composition registry standing in for the global one, so a load leaves no composition behind."""
    import interpretune.adapters
    from interpretune.adapters.registration import CompositionRegistry

    fresh = CompositionRegistry()
    monkeypatch.setattr(interpretune.adapters, "ADAPTER_REGISTRY", fresh)
    return fresh


class TestEntries:
    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ("org/repo@abc1234", ("org/repo", "abc1234")),
            ("org/it-interp-engine-adapter@" + "a" * 40, ("org/it-interp-engine-adapter", "a" * 40)),
            ("org/repo@local" + "0" * 35, ("org/repo", "local" + "0" * 35)),
        ],
    )
    def test_a_pinned_entry_parses(self, entry, expected):
        from interpretune.hub.adapters import parse_hub_adapter_entry

        assert parse_hub_adapter_entry(entry) == expected

    @pytest.mark.parametrize("entry", ["org/repo", "org/repo@main", "org/repo@v1.0", "org/repo@abc12", "repo@abc1234"])
    def test_an_unpinned_entry_is_refused(self, entry):
        from interpretune.hub.adapters import AdapterComponentError, parse_hub_adapter_entry

        with pytest.raises(AdapterComponentError, match="not pinned"):
            parse_hub_adapter_entry(entry)


class TestLoadRunHubAdapters:
    def test_a_cached_entry_loads_without_a_pull_and_is_recorded(self, published, trusted, registry, monkeypatch):
        import interpretune.hub.api
        from interpretune.hub.adapters import load_run_hub_adapters, loaded_hub_adapters

        cache, revision = published
        monkeypatch.setattr(interpretune.hub.api, "pull", lambda *a, **k: pytest.fail("a cached entry was pulled"))
        (load,) = load_run_hub_adapters([f"{REPO}@{revision[:12]}"], cache_dir=cache, registry=registry)
        assert [m.name for m in load.members] == [FIXTURE_ADAPTER]
        assert loaded_hub_adapters()[REPO] == {"revision": revision, "adapters": [FIXTURE_ADAPTER]}
        # idempotent: naming it again executes nothing new and records the same revision
        load_run_hub_adapters(f"{REPO}@{revision}", cache_dir=cache, registry=registry)
        assert loaded_hub_adapters()[REPO]["revision"] == revision

    def test_an_uncached_hub_revision_is_pulled_at_that_revision(self, published, trusted, registry, monkeypatch):
        import interpretune.hub.api
        from interpretune.hub.adapters import AdapterComponentError, load_run_hub_adapters

        cache, _ = published
        calls = []
        monkeypatch.setattr(interpretune.hub.api, "pull", lambda repo_id, **kwargs: calls.append((repo_id, kwargs)))
        # the fake pull caches nothing, so the load that follows finds no such revision
        with pytest.raises(KeyError, match="no cached revision"):
            load_run_hub_adapters([f"{REPO}@abcdef1"], cache_dir=cache, registry=registry)
        assert calls == [(REPO, {"revision": "abcdef1", "cache_dir": cache})]
        with pytest.raises(AdapterComponentError, match="local-publish revision"):
            load_run_hub_adapters([f"{REPO}@local" + "f" * 35], cache_dir=cache, registry=registry)

    def test_an_untrusted_entry_is_refused_before_any_download(self, published, registry, monkeypatch):
        import interpretune.hub.api
        from interpretune.hub.adapters import load_run_hub_adapters
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR, RemoteCodeNotTrustedError

        cache, _ = published
        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        monkeypatch.setattr(interpretune.hub.api, "pull", lambda *a, **k: pytest.fail("pulled before the trust gate"))
        with pytest.raises(RemoteCodeNotTrustedError):
            load_run_hub_adapters([f"{REPO}@abcdef1"], cache_dir=cache, registry=registry)


class TestUnregisteredAdapter:
    def test_the_error_names_the_cached_component_that_declares_it(self, published):
        from interpretune.config.shared import adapter_seq_to_list

        _, revision = published
        with pytest.raises(KeyError, match=rf"hub_adapters: \[{REPO}@{revision[:12]}\]"):
            adapter_seq_to_list(["core", FIXTURE_ADAPTER])

    def test_without_a_cached_provider_the_error_names_the_key(self, tmp_path, monkeypatch):
        from interpretune.config.loading import resolve_adapter_ctx
        from interpretune.hub import components

        monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", tmp_path / "empty")
        with pytest.raises(KeyError, match=r"hub_adapters: \[<org>/<repo>@<commit>\]"):
            resolve_adapter_ctx({"adapter_ctx": ["core", "not_an_adapter"]})


class TestCLI:
    @pytest.fixture()
    def captured(self, monkeypatch):
        from interpretune.hub import adapters

        calls = []
        monkeypatch.setattr(adapters, "load_run_hub_adapters", lambda entries, *a, **k: calls.append(list(entries)))
        return calls

    def test_entries_are_read_from_configs_and_argv_in_order(self, tmp_path, captured):
        from interpretune.base.components.cli import ITSessionMixin

        first, second, plain = (tmp_path / f"{n}.yaml" for n in ("first", "second", "plain"))
        first.write_text(yaml.safe_dump({"hub_adapters": ["org/a@abc1234"]}))
        second.write_text(yaml.safe_dump({"hub_adapters": ["org/b@abc1234"], "seed_everything": 1}))
        plain.write_text(yaml.safe_dump({"seed_everything": 1}))
        load = ITSessionMixin._load_hub_adapters_from_args
        # a later file replaces an earlier one's list, one without the key leaves it alone
        assert load(["--config", str(first), f"--config={second}", "-c", str(plain)]) == ["org/b@abc1234"]
        # a later command-line value replaces the files' list, as jsonargparse merges a list
        assert load(["--config", str(first), "--hub_adapters=[org/c@abc1234]"]) == ["org/c@abc1234"]
        assert load(["fit", "--hub_adapters", "org/d@abc1234", "--trainer.max_epochs=1"]) == ["org/d@abc1234"]
        assert load(["--config", str(plain)]) == []
        assert captured == [["org/b@abc1234"], ["org/c@abc1234"], ["org/d@abc1234"], []]

    def test_the_parser_accepts_the_key(self, tmp_path, captured):
        from jsonargparse import ArgumentParser

        from interpretune.base.components.cli import ITSessionMixin

        parser = ArgumentParser()
        parser.add_argument("-c", "--config", action="config")
        ITSessionMixin().add_arguments_to_parser(parser)
        config = tmp_path / "run.yaml"
        config.write_text(yaml.safe_dump({"hub_adapters": ["org/a@abc1234"]}))
        assert parser.parse_args(["--config", str(config)]).hub_adapters == ["org/a@abc1234"]
        assert parser.parse_args([]).hub_adapters == []
