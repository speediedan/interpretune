"""Component manifest, card-generation, and publish-tree parity tests (interpretune#1 / hub design v2)."""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
import yaml

from interpretune.hub.cards import generate_component_card
from interpretune.hub.manifest import (
    ComponentManifestError,
    derive_config_key,
    load_component_manifest,
    validate_component_manifest,
)
from interpretune.hub.publish import build_component_tree

RTE_COMPONENT_DIR = Path(__file__).parent.parent.parent / "src" / "it_examples" / "examples" / "rte"
RTE_ENTRYPOINT = RTE_COMPONENT_DIR.parent.parent / "experiments" / "rte_boolq.py"


class TestComponentManifest:
    def test_in_repo_manifest_validates(self):
        manifest = load_component_manifest(RTE_COMPONENT_DIR / "it_component.yaml")
        assert manifest["kinds"] == ["module", "datamodule"]
        assert len(manifest["module"]["configs"]) == 6

    def test_schema_version_is_mandatory(self):
        with pytest.raises(ComponentManifestError, match="it_schema_version"):
            validate_component_manifest({"kinds": ["module"], "module": {"configs": {}}})

    def test_unknown_kind_rejected(self):
        with pytest.raises(ComponentManifestError, match="kinds"):
            validate_component_manifest({"it_schema_version": 1, "kinds": ["sorcery"]})

    def test_derived_keys_use_canonical_alphabetical_composition(self):
        # canonical order sorts by adapter value (the name string): nnsight < sae_lens
        key = derive_config_key(
            {"task_variant": "rte_demo", "model": "gpt2", "composition": ["sae_lens", "nnsight"], "extensions": []}
        )
        assert key == "rte_demo.gpt2.nnsight+sae_lens"

    def test_derived_descriptor_is_materialized_from_extensions(self):
        key = derive_config_key(
            {
                "task_variant": "rte_demo",
                "model": "gemma2",
                "composition": ["circuit_tracer"],
                "extensions": ["neuronpedia"],
            }
        )
        assert key == "rte_demo.gemma2.circuit_tracer.neuronpedia"

    def test_core_composition_is_explicit(self):
        key = derive_config_key({"task_variant": "rte", "model": "gpt2", "composition": [], "extensions": []})
        assert key == "rte.gpt2.core"


class TestOpsKindSpec:
    """``kinds: [ops]`` had no manifest spec at all: ``validate_component_manifest`` special-cased only ``module``
    and ``promptconfigs``, and the sole ops-aware code copied an unvalidated ``ops.files`` list at publish time.

    That list is what makes op discovery manifest-routed rather than a blind glob over every YAML in the repo, which is
    also what makes the registration claim "one manifest fetch per logical load" true for the ops kind (#266 Phase 3).
    """

    _VALID = {"it_schema_version": 1, "kinds": ["ops"], "ops": {"files": ["concept_ops.yaml"]}}

    def test_valid_ops_manifest_accepted(self):
        assert validate_component_manifest(dict(self._VALID))["ops"]["files"] == ["concept_ops.yaml"]

    def test_ops_kind_may_be_combined_with_others(self):
        manifest = dict(self._VALID, kinds=["ops", "module"], module={"configs": {"rte.gpt2.core": {}}})
        assert validate_component_manifest(manifest)["kinds"] == ["ops", "module"]

    @pytest.mark.parametrize(
        "ops, why",
        [
            (None, "no ops block at all"),
            ({}, "no files key"),
            ({"files": []}, "empty file list"),
            ({"files": "concept_ops.yaml"}, "a bare string rather than a list"),
            ({"files": ["concept_ops.yaml", ""]}, "an empty path entry"),
            ({"files": [{"path": "concept_ops.yaml"}]}, "a non-string entry"),
        ],
    )
    def test_malformed_ops_declarations_rejected(self, ops, why):
        manifest = {"it_schema_version": 1, "kinds": ["ops"]}
        if ops is not None:
            manifest["ops"] = ops
        with pytest.raises(ComponentManifestError, match="`ops.files`|kind `ops`"):
            validate_component_manifest(manifest, source=why)

    def test_manifest_may_not_list_itself_as_an_op_file(self):
        """The manifest declares the op definitions; parsing it as one fails on its own scalar keys."""
        manifest = {"it_schema_version": 1, "kinds": ["ops"], "ops": {"files": ["it_component.yaml"]}}
        with pytest.raises(ComponentManifestError, match="must not list it_component.yaml"):
            validate_component_manifest(manifest)


class TestGeneratedCards:
    def test_card_carries_discovery_sentinel_and_dataset_mirror(self):
        manifest = load_component_manifest(RTE_COMPONENT_DIR / "it_component.yaml")
        card = generate_component_card(manifest, "speediedan/rte")
        assert card.data.library_name == "interpretune"
        for expected_tag in ("interpretune", "interpretune-module", "interpretune-datamodule", "task:rte"):
            assert expected_tag in card.data.tags
        assert "aps/super_glue" in card.data.datasets

    def test_every_publish_produces_a_card(self, tmp_path):
        """No publish path may produce a card-less repo — the card IS the discovery sentinel."""
        out = tmp_path / "build"
        manifest = build_component_tree(RTE_COMPONENT_DIR, out, entrypoint_src=RTE_ENTRYPOINT)
        generate_component_card(manifest, "speediedan/rte").save(out / "README.md")
        assert (out / "README.md").exists()
        assert "library_name: interpretune" in (out / "README.md").read_text(encoding="utf-8")


class TestPublishTreeParity:
    """The in-repo tree mirrors the Hub tree: publishing is a copy plus generated additions, nothing else."""

    def test_built_tree_mirrors_in_repo_tree(self, tmp_path):
        out = tmp_path / "build"
        manifest = build_component_tree(RTE_COMPONENT_DIR, out, entrypoint_src=RTE_ENTRYPOINT)

        # every in-repo file is copied byte-identical
        for rel in ["it_component.yaml"] + sorted(manifest["module"]["configs"].values()):
            assert filecmp.cmp(RTE_COMPONENT_DIR / rel, out / rel, shallow=False), f"drift in {rel}"
        # the generated additions are exactly the declared entrypoint (card is added by publish_component)
        built = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        source = {p.relative_to(RTE_COMPONENT_DIR).as_posix() for p in RTE_COMPONENT_DIR.rglob("*") if p.is_file()}
        assert built - source == {manifest["module"]["entrypoint"]}
        assert source - built == set()

    def test_parity_check_blocks_drifted_config(self, tmp_path):
        import shutil

        src_copy = tmp_path / "component"
        shutil.copytree(RTE_COMPONENT_DIR, src_copy)
        drifted = src_copy / "configs" / "rte_demo.gpt2.sae_lens.yaml"
        drifted.write_text(
            drifted.read_text(encoding="utf-8").replace("model: gpt2", "model: gpt3000"), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="parity violation"):
            build_component_tree(src_copy, tmp_path / "build", entrypoint_src=RTE_ENTRYPOINT)

    def test_missing_entrypoint_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="entrypoint"):
            build_component_tree(RTE_COMPONENT_DIR, tmp_path / "build", entrypoint_src=tmp_path / "nope.py")


class TestManifestFirstOffline:
    """Local resolution must never touch the network (design invariant).

    The socket-blocked resolution leg lives in ``TestHubVerbSurface.test_load_returns_hydrated_registered_cfg``
    (the post-flip surface); this class keeps the schema-roundtrip half.
    """

    def test_hub_config_body_roundtrips_registry_schema(self):
        """A fetched configuration body is exactly what the local loader consumes — one schema, no adapters."""
        cfg_path = RTE_COMPONENT_DIR / "configs" / "rte_demo.gemma2.circuit_tracer.yaml"
        body = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert {"task_variant", "model", "composition", "reg_info", "shared_config", "registered_cfg"} <= set(body)


class TestLocalPublishBridge:
    """The local-publish bridge + cache-only resolution (design v3 §11.2): the 4c acceptance core."""

    def test_bridge_roundtrip_sockets_blocked(self, tmp_path, monkeypatch):
        """Local-publish a seed -> cache -> resolve -> load a session cfg, with the network unreachable."""
        import socket

        def _blocked(*args, **kwargs):
            raise AssertionError("cache-backed resolution attempted a network connection")

        monkeypatch.setattr(socket.socket, "connect", _blocked)
        from interpretune.config.loading import load_session_cfg
        from interpretune.hub.components import local_publish, resolve_component_config

        cache = tmp_path / "components"
        rev = local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        assert rev.startswith("local") and len(rev) == 40
        key, body = resolve_component_config("speediedan/rte", "rte_demo.gemma2.circuit_tracer", cache_dir=cache)
        loaded = load_session_cfg(body, expected_key=key)
        assert type(loaded.datamodule_cfg).__name__ == "ITDataModuleConfig"
        assert loaded.module_cfg.optimizer_init["class_path"] == "torch.optim.AdamW"  # materialized default

    def test_bridge_is_idempotent_and_tracks_content(self, tmp_path):
        from interpretune.hub.components import local_publish

        cache = tmp_path / "components"
        rev1 = local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        rev2 = local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        assert rev1 == rev2, "unchanged content must map to the same pseudo-revision"
        snapshots = tmp_path / "components" / "models--speediedan--rte" / "snapshots"
        assert len(list(snapshots.iterdir())) == 1

    def test_uncached_component_names_the_fetch_command(self, tmp_path):
        from interpretune.hub.components import resolve_component_config

        with pytest.raises(KeyError, match="interpretune.hub.pull"):
            resolve_component_config("someorg/absent", "rte.x.core", cache_dir=tmp_path / "empty")


@pytest.fixture()
def seeded_cache(tmp_path):
    """A components cache holding the in-tree rte seed, materialized via the local-publish bridge."""
    from it_examples.seeds import ensure_local_seeds

    cache = tmp_path / "components"
    ensure_local_seeds(cache_dir=cache)
    return cache


class TestHubVerbSurface:
    """The ratified 5e verb surface: ``it.hub.pull`` / ``it.hub.load`` / ``ITSession.from_hub``."""

    def test_load_returns_hydrated_registered_cfg(self, seeded_cache, monkeypatch):
        import socket

        monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        import interpretune as it

        dm_cfg, m_cfg, dm_cls, m_cls = it.hub.load(
            "speediedan/rte", "rte_demo.gemma2.circuit_tracer", cache_dir=seeded_cache
        )
        assert type(dm_cfg).__name__ == "ITDataModuleConfig"
        assert m_cls.__name__ == "RTEBoolqModule"
        assert m_cfg.optimizer_init["class_path"] == "torch.optim.AdamW"  # materialized default survives

    def test_load_uncached_raises_with_fetch_command(self, tmp_path):
        import interpretune as it

        with pytest.raises(KeyError, match="interpretune.hub.pull"):
            it.hub.load("someorg/absent", "rte.x.core", cache_dir=tmp_path / "empty")

    def test_from_hub_constructs_session_cfg_path(self, seeded_cache, monkeypatch):
        """from_hub routes the cached body through the one-door loader before session construction."""
        from interpretune.session import ITSession

        captured = {}

        def _capture_init(self, session_cfg, *args, **kwargs):
            captured["cfg"] = session_cfg

        monkeypatch.setattr(ITSession, "__init__", _capture_init)
        ITSession.from_hub("speediedan/rte", "rte_demo.gemma2.circuit_tracer", cache_dir=seeded_cache)
        cfg = captured["cfg"]
        assert type(cfg).__name__ == "ITSessionConfig"
        assert cfg.module_cls.__name__ == "RTEBoolqModule"


class TestComponentRequires:
    """`requires:` enforcement — each failure mode fails informatively at resolution time."""

    @staticmethod
    def _publish_with_requires(tmp_path, requires_patch: str) -> Path:
        import shutil

        from interpretune.hub.components import local_publish

        src_copy = tmp_path / "component"
        shutil.copytree(RTE_COMPONENT_DIR, src_copy)
        manifest_path = src_copy / "it_component.yaml"
        patched = manifest_path.read_text(encoding="utf-8").replace("interpretune: '>=0.1.dev0'", requires_patch)
        assert patched != manifest_path.read_text(encoding="utf-8"), "requires patch did not apply"
        manifest_path.write_text(patched, encoding="utf-8")
        cache = tmp_path / "components"
        local_publish(src_copy, "someorg/patched", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        return cache

    @pytest.mark.parametrize(
        ("requires_patch", "match"),
        [
            ("interpretune: '>=999.0'", "requires interpretune"),
            ("interpretune: '>=0.1.dev0'\n  extra_unknown_adapter_sentinel: true", None),  # control: still passes
        ],
        ids=["unsatisfied-interpretune-floor", "unknown-extra-key-ignored"],
    )
    def test_interpretune_floor(self, tmp_path, requires_patch, match):
        from interpretune.hub.components import ComponentRequirementError, resolve_component_config

        cache = self._publish_with_requires(tmp_path, requires_patch)
        if match:
            with pytest.raises(ComponentRequirementError, match=match):
                resolve_component_config("someorg/patched", "rte_demo.gpt2.sae_lens", cache_dir=cache)
        else:
            key, _ = resolve_component_config("someorg/patched", "rte_demo.gpt2.sae_lens", cache_dir=cache)
            assert key == "rte_demo.gpt2.sae_lens"

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            (("- nnsight", "- no_such_adapter"), "does not provide"),
            (("pip: []", "pip:\n  - definitely-not-a-real-package-xyz"), "not installed"),
            (("pip: []", "pip:\n  - pytest>=999.0"), "is installed"),
        ],
        ids=["unknown-adapter", "missing-pip-package", "unsatisfied-pip-specifier"],
    )
    def test_requires_failure_modes(self, tmp_path, mutation, match):
        import shutil

        from interpretune.hub.components import (
            ComponentRequirementError,
            local_publish,
            resolve_component_config,
        )

        src_copy = tmp_path / "component"
        shutil.copytree(RTE_COMPONENT_DIR, src_copy)
        manifest_path = src_copy / "it_component.yaml"
        old, new = mutation
        patched = manifest_path.read_text(encoding="utf-8").replace(old, new)
        assert patched != manifest_path.read_text(encoding="utf-8"), "requires mutation did not apply"
        manifest_path.write_text(patched, encoding="utf-8")
        cache = tmp_path / "components"
        local_publish(src_copy, "someorg/patched", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        with pytest.raises(ComponentRequirementError, match=match):
            resolve_component_config("someorg/patched", "rte_demo.gpt2.sae_lens", cache_dir=cache)


class TestBareKeyAliasing:
    """Collision-aware bare-key aliasing atop namespaced hub registration."""

    def _register_from_cache(self, seeded_cache, registry, monkeypatch, alias_bare_key=True):
        """Route register_component_config's fetch through the cache (no network in tests)."""
        from interpretune.hub import components as hub_components

        def _cache_pull(repo_id, key, revision=None, cache_dir=None, token=None):
            return hub_components.resolve_component_config(repo_id, key, cache_dir=seeded_cache)

        monkeypatch.setattr(hub_components, "pull_component_config", _cache_pull)
        return hub_components.register_component_config(
            "speediedan/rte", "rte_demo.gpt2.sae_lens", target_registry=registry, alias_bare_key=alias_bare_key
        )

    def test_namespaced_and_bare_keys_both_register(self, seeded_cache, monkeypatch):
        from interpretune.registry import ModuleRegistry

        registry = ModuleRegistry()
        namespaced = self._register_from_cache(seeded_cache, registry, monkeypatch)
        assert namespaced == "speediedan.rte.rte_demo.gpt2.sae_lens"
        assert registry.get(namespaced) is not None
        assert registry.get("rte_demo.gpt2.sae_lens") is not None

    def test_bare_key_collision_keeps_existing_entry(self, seeded_cache, monkeypatch, recwarn):
        from interpretune.registry import ModuleRegistry

        registry = ModuleRegistry()
        sentinel = {"existing": True}
        registry["rte_demo.gpt2.sae_lens"] = sentinel
        self._register_from_cache(seeded_cache, registry, monkeypatch)
        assert registry["rte_demo.gpt2.sae_lens"] is sentinel  # never silently overridden
        assert any("already registered" in str(w.message) for w in recwarn.list)

    def test_alias_can_be_disabled(self, seeded_cache, monkeypatch):
        from interpretune.registry import ModuleRegistry

        registry = ModuleRegistry()
        self._register_from_cache(seeded_cache, registry, monkeypatch, alias_bare_key=False)
        assert registry.get("speediedan.rte.rte_demo.gpt2.sae_lens") is not None
        assert "rte_demo.gpt2.sae_lens" not in registry
