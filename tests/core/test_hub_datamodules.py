"""#128: datamodules as independently shareable/addressable hub components.

``RegisteredCfg`` binds datamodule and module inseparably, which is right for task components. These
tests pin the datamodule-only half: the ``datamodule`` manifest kind, name-addressed standalone
payloads, cross-repo REPLACEMENT references (``ref: <org>/<repo>#<name>``), and the strictly two-path
consumption contract -- an inlined ``datamodule_cfg`` is used wholesale, a referenced payload is used
wholesale, and nothing ever merges across the two paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interpretune.hub.manifest import ComponentManifestError, validate_component_manifest

RTE_COMPONENT_DIR = Path(__file__).parent.parent.parent / "src" / "it_examples" / "examples" / "rte"
RTE_ENTRYPOINT = RTE_COMPONENT_DIR.parent.parent / "experiments" / "rte_boolq.py"


def _manifest(**overrides):
    base = {"it_schema_version": 1, "kinds": ["datamodule"], "datamodules": {"dm": {"config": "configs/dm.yaml"}}}
    base.update(overrides)
    return base


@pytest.fixture()
def rte_cache(tmp_path):
    """The in-tree RTE component (which declares the datamodule kind) local-published into a tmp cache."""
    from interpretune.hub.components import local_publish

    cache = tmp_path / "components"
    local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
    return cache


class TestDataModuleKindManifest:
    def test_kind_requires_a_datamodules_index(self):
        with pytest.raises(ComponentManifestError, match="requires a non-empty `datamodules` index"):
            validate_component_manifest(_manifest(datamodules=None), source="t")

    def test_entry_requires_a_config_path(self):
        with pytest.raises(ComponentManifestError, match="requires a repo-relative `config` path"):
            validate_component_manifest(_manifest(datamodules={"dm": {"entrypoint": "x.py"}}), source="t")

    def test_in_repo_rte_manifest_declares_the_kind_and_validates(self):
        import yaml

        manifest = validate_component_manifest(
            yaml.safe_load((RTE_COMPONENT_DIR / "it_component.yaml").read_text(encoding="utf-8")),
            source="rte",
        )
        assert "datamodule" in manifest["kinds"] and "rte_boolq" in manifest["datamodules"]


class TestStandaloneResolutionAndHydration:
    def test_resolve_names_availables_on_a_bad_name(self, rte_cache):
        from interpretune.hub.components import resolve_datamodule_config

        with pytest.raises(KeyError, match=r"declares no datamodule 'nope'.*rte_boolq"):
            resolve_datamodule_config("speediedan/rte", "nope", cache_dir=rte_cache)

    def test_payload_must_be_module_free(self, rte_cache, tmp_path):
        """The resolver enforces the datamodule-only half of the two-path contract at the source."""
        import shutil

        import yaml

        from interpretune.hub.components import local_publish, resolve_datamodule_config

        tainted = tmp_path / "component"
        shutil.copytree(RTE_COMPONENT_DIR, tainted)
        payload_path = tainted / "configs" / "datamodule.rte_boolq.yaml"
        payload = yaml.safe_load(payload_path.read_text(encoding="utf-8"))
        payload["module_cfg"] = {"task_name": "smuggled"}
        payload_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        cache = tmp_path / "cache"
        local_publish(tainted, "someorg/tainted", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        with pytest.raises(ComponentManifestError, match="must not carry module configuration"):
            resolve_datamodule_config("someorg/tainted", "rte_boolq", cache_dir=cache)

    def test_load_datamodule_hydrates_without_module_coupling(self, rte_cache):
        import interpretune as it
        from interpretune.registry import RegisteredDataModuleCfg

        dm = it.hub.load_datamodule("speediedan/rte", "rte_boolq", cache_dir=rte_cache)
        assert isinstance(dm, RegisteredDataModuleCfg)
        assert not hasattr(dm, "module_cfg"), "the datamodule half must not smuggle a module in"
        # the payload's OWN shared_config applied through the one merge site
        assert dm.datamodule_cfg.task_name == "rte"
        assert dm.datamodule_cfg.train_batch_size == 2
        assert dm.datamodule_cls.__name__ == "RTEBoolqDataModule"


class TestReplacementReferences:
    @pytest.fixture()
    def ref_cache(self, rte_cache, monkeypatch):
        """Point the ref resolver's default cache at the tmp publish (refs resolve cross-repo, so they use the
        components cache rather than a per-call override)."""
        import interpretune.hub.components as components

        monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", rte_cache)
        return rte_cache

    @staticmethod
    def _body(dm_cfg, **registered_extra):
        registered = {"datamodule_cfg": dm_cfg, "module_cfg": {"task_name": "t", "model_name_or_path": "gpt2"}}
        registered.update(registered_extra)
        return {"adapter_ctx": ["core"], "registered_cfg": registered}

    def test_ref_resolves_wholesale_including_the_class(self, ref_cache):
        from interpretune.config.loading import load_session_cfg

        loaded = load_session_cfg(self._body({"ref": "speediedan/rte#rte_boolq"}))
        assert loaded.datamodule_cfg.task_name == "rte"
        assert loaded.datamodule_cls.__name__ == "RTEBoolqDataModule"

    def test_referring_shared_config_does_not_leak_into_the_ref(self, ref_cache):
        """THE no-merge pin: the referenced payload's own shared_config wins, the referring body's is
        never layered in. A ref is a replacement, not a base."""
        from interpretune.config.loading import load_session_cfg

        body = {
            "adapter_ctx": ["core"],
            # module_cfg deliberately omits task_name so shared_config is what supplies it -- shared
            # fills gaps rather than overriding explicit body values, so an explicit task_name here
            # would mask whether shared reached the module half at all
            # class_path form, as every published module config uses -- it is the branch through
            # which it_cfg_factory applies shared_config (the plain-dict branch ignores it)
            "registered_cfg": {
                "datamodule_cfg": {"ref": "speediedan/rte#rte_boolq"},
                "module_cfg": {
                    "class_path": "interpretune.config.module.ITConfig",
                    "init_args": {"model_name_or_path": "gpt2"},
                },
            },
            "shared_config": {"task_name": "MUST_NOT_LEAK"},
        }
        loaded = load_session_cfg(body)
        assert loaded.datamodule_cfg.task_name == "rte"
        # ...while the module half still receives the referring body's shared_config
        assert loaded.module_cfg.task_name == "MUST_NOT_LEAK"

    def test_ref_with_extra_keys_is_rejected(self):
        from interpretune.config.loading import load_session_cfg

        with pytest.raises(ValueError, match="must contain ONLY `ref`"):
            load_session_cfg(self._body({"ref": "a/b#c", "train_batch_size": 4}))

    def test_ref_with_local_datamodule_cls_is_rejected(self, ref_cache):
        """Declaring the class locally while ref'ing the config would be a partial merge by the back
        door -- the reference supplies both halves."""
        from interpretune.config.loading import load_session_cfg

        with pytest.raises(ValueError, match="must not also declare `datamodule_cls`"):
            load_session_cfg(
                self._body(
                    {"ref": "speediedan/rte#rte_boolq"},
                    datamodule_cls="it_examples.experiments.rte_boolq.RTEBoolqDataModule",
                )
            )

    @pytest.mark.parametrize("bad", ["no-hash", "org/repo#", "#name", "org#name", "a/b/c#d", "a/b#c#d"])
    def test_malformed_refs_are_rejected_loudly(self, bad):
        from interpretune.config.loading import parse_datamodule_ref

        with pytest.raises(ValueError, match="must be `<org>/<repo>#<name>`"):
            parse_datamodule_ref(bad)

    def test_uncached_ref_names_the_fetch_command(self, tmp_path, monkeypatch):
        import interpretune.hub.components as components
        from interpretune.config.loading import load_session_cfg

        monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", tmp_path / "empty")
        with pytest.raises(KeyError, match="interpretune.hub.pull"):
            load_session_cfg(self._body({"ref": "someorg/absent#dm"}))
