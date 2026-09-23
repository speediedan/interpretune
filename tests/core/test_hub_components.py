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
from tests.rte_component import rte_entrypoint_src

RTE_COMPONENT_DIR = Path(__file__).parent.parent.parent / "src" / "it_examples" / "examples" / "rte"


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
        manifest = build_component_tree(RTE_COMPONENT_DIR, out, entrypoint_src=rte_entrypoint_src())
        generate_component_card(manifest, "speediedan/rte").save(out / "README.md")
        assert (out / "README.md").exists()
        assert "library_name: interpretune" in (out / "README.md").read_text(encoding="utf-8")


class TestPublishTreeParity:
    """The in-repo tree mirrors the Hub tree: publishing is a copy plus generated additions, nothing else."""

    def test_built_tree_mirrors_in_repo_tree(self, tmp_path):
        out = tmp_path / "build"
        manifest = build_component_tree(RTE_COMPONENT_DIR, out, entrypoint_src=rte_entrypoint_src())

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
            build_component_tree(src_copy, tmp_path / "build", entrypoint_src=rte_entrypoint_src())

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
        rev = local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src(), cache_dir=cache)
        assert rev.startswith("local") and len(rev) == 40
        key, body = resolve_component_config("speediedan/rte", "rte_demo.gemma2.circuit_tracer", cache_dir=cache)
        loaded = load_session_cfg(body, expected_key=key)
        assert type(loaded.datamodule_cfg).__name__ == "ITDataModuleConfig"
        assert loaded.module_cfg.optimizer_init["class_path"] == "torch.optim.AdamW"  # materialized default

    def test_bridge_is_idempotent_and_tracks_content(self, tmp_path):
        from interpretune.hub.components import local_publish

        cache = tmp_path / "components"
        rev1 = local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src(), cache_dir=cache)
        rev2 = local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src(), cache_dir=cache)
        assert rev1 == rev2, "unchanged content must map to the same pseudo-revision"
        snapshots = tmp_path / "components" / "models--speediedan--rte" / "snapshots"
        assert len(list(snapshots.iterdir())) == 1

    def test_uncached_component_names_the_fetch_command(self, tmp_path):
        from interpretune.hub.components import resolve_component_config

        with pytest.raises(KeyError, match="interpretune.hub.pull"):
            resolve_component_config("someorg/absent", "rte.x.core", cache_dir=tmp_path / "empty")


@pytest.fixture()
def seeded_cache(tmp_path):
    """A components cache holding the seeds plus the rte component, materialized via bridges.

    The experiment is Hub-resident now, so its half is a local publish of the in-tree component dir with the entrypoint
    sourced from the warmed snapshot (mirroring the socket-blocked bridge test below): tmp-cache tests keep exercising
    the current tree, cache-only.
    """
    from interpretune.hub.components import local_publish
    from it_examples.seeds import ensure_local_seeds
    from tests.rte_component import rte_entrypoint_src

    cache = tmp_path / "components"
    ensure_local_seeds(cache_dir=cache)
    local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src(), cache_dir=cache)
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
        local_publish(src_copy, "someorg/patched", entrypoint_src=rte_entrypoint_src(), cache_dir=cache)
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
        local_publish(src_copy, "someorg/patched", entrypoint_src=rte_entrypoint_src(), cache_dir=cache)
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


def _experiment_component_dir(root: Path, *, key: str = "demo_experiment", name: str | None = None) -> Path:
    """A minimal valid experiment component tree: manifest + one definition config + pipeline file."""
    from interpretune.hub.manifest import IT_COMPONENT_MANIFEST

    component = root / "component"
    (component / "configs").mkdir(parents=True)
    manifest = {
        "it_schema_version": 1,
        "kinds": ["experiment"],
        "requires": {"interpretune": ">=0.1.dev0"},
        "experiments": {
            key: {"config": f"configs/{key}.yaml", "pipeline": "pipeline/run.py", "files": ["notes.md"]},
        },
    }
    (component / IT_COMPONENT_MANIFEST).write_text(__import__("yaml").safe_dump(manifest), encoding="utf-8")
    (component / "pipeline").mkdir()
    (component / "pipeline" / "run.py").write_text("ENTRY = True\n", encoding="utf-8")
    (component / "notes.md").write_text("notes\n", encoding="utf-8")
    (component / "configs" / f"{key}.yaml").write_text(
        __import__("yaml").safe_dump({"EXPERIMENT_NAME": name or key, "PROMPT": {"text": "hi"}}),
        encoding="utf-8",
    )
    return component


class TestExperimentKindSpec:
    """``kinds: [experiment]`` per the #498 ruling: own parity, snapshot confinement, declared sessions."""

    _VALID_ENTRY = {"config": "configs/demo_experiment.yaml", "pipeline": "pipeline/run.py"}

    def test_valid_experiment_manifest_accepted(self):
        manifest = {
            "it_schema_version": 1,
            "kinds": ["experiment"],
            "requires": {"interpretune": ">=0.1.dev0", "components": ["speediedan/rte"]},
            "experiments": {"demo_experiment": dict(self._VALID_ENTRY)},
        }
        assert validate_component_manifest(manifest)["experiments"]["demo_experiment"]["config"] == (
            "configs/demo_experiment.yaml"
        )

    def test_experiment_kind_may_combine_with_module(self):
        manifest = {
            "it_schema_version": 1,
            "kinds": ["experiment", "module"],
            "module": {"configs": {"rte.gpt2.core": {}}},
            "experiments": {"demo_experiment": dict(self._VALID_ENTRY)},
        }
        assert validate_component_manifest(manifest)["kinds"] == ["experiment", "module"]

    @pytest.mark.parametrize(
        ("experiments", "why"),
        [
            (None, "no experiments block at all"),
            ({}, "empty index"),
            ({"demo": {}}, "entry without a config"),
            ({"demo": {"config": "/abs/path.yaml"}}, "absolute config path"),
            ({"demo": {"config": "../escape.yaml"}}, "config escaping the component"),
            ({"demo": {"config": 5}}, "non-string config"),
            ({"demo": {"config": "configs/x.yaml", "pipeline": "../run.py"}}, "pipeline escaping"),
        ],
    )
    def test_malformed_experiment_declarations_rejected(self, experiments, why):
        manifest = {"it_schema_version": 1, "kinds": ["experiment"]}
        if experiments is not None:
            manifest["experiments"] = experiments
        with pytest.raises(ComponentManifestError, match="`experiments`|experiment entry"):
            validate_component_manifest(manifest, source=why)

    def test_experiment_key_derives_from_experiment_name(self):
        from interpretune.hub.manifest import check_experiment_key_parity, derive_experiment_key

        assert derive_experiment_key({"EXPERIMENT_NAME": "demo_experiment"}) == "demo_experiment"
        with pytest.raises(ComponentManifestError, match="EXPERIMENT_NAME"):
            derive_experiment_key({"PROMPT": {}})
        with pytest.raises(ValueError, match="parity violation"):
            check_experiment_key_parity(Path("other.yaml"), {"EXPERIMENT_NAME": "demo_experiment"})

    def test_requires_components_axis_shape(self):
        good = {"it_schema_version": 1, "kinds": ["experiment"], "experiments": {"d": dict(self._VALID_ENTRY)}}
        validate_component_manifest(dict(good, requires={"components": ["speediedan/rte"]}))
        with pytest.raises(ComponentManifestError, match="components"):
            validate_component_manifest(dict(good, requires={"components": ["not-a-repo-ref"]}))

    def test_built_tree_carries_experiment_payloads_and_parity(self, tmp_path):
        component = _experiment_component_dir(tmp_path)
        out = tmp_path / "build"
        build_component_tree(component, out)
        for rel in ["it_component.yaml", "configs/demo_experiment.yaml", "pipeline/run.py", "notes.md"]:
            assert filecmp.cmp(component / rel, out / rel, shallow=False), f"drift in {rel}"

    def test_parity_check_blocks_drifted_experiment_config(self, tmp_path):
        component = _experiment_component_dir(tmp_path)
        drifted = component / "configs" / "demo_experiment.yaml"
        drifted.write_text(
            drifted.read_text(encoding="utf-8").replace("demo_experiment", "renamed_experiment"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="parity violation"):
            build_component_tree(component, tmp_path / "build")

    def test_missing_experiment_payload_refused(self, tmp_path):
        component = _experiment_component_dir(tmp_path)
        (component / "pipeline" / "run.py").unlink()
        with pytest.raises(FileNotFoundError, match="experiment payload"):
            build_component_tree(component, tmp_path / "build")

    def test_module_request_against_experiment_component_names_kinds(self, tmp_path):
        from interpretune.hub.components import local_publish, resolve_component_config

        cache = tmp_path / "components"
        local_publish(_experiment_component_dir(tmp_path), "speediedan/demo-exp", cache_dir=cache)
        with pytest.raises(KeyError, match="not `module`"):
            resolve_component_config("speediedan/demo-exp", "demo_experiment", cache_dir=cache)

    def test_experiment_request_against_module_component_names_kinds(self, seeded_cache):
        from interpretune.hub.components import resolve_experiment_config

        with pytest.raises(KeyError, match="not `experiment`"):
            resolve_experiment_config("speediedan/rte", "rte_demo.gpt2.sae_lens", cache_dir=seeded_cache)

    def test_snapshot_experiment_loads_confined(self, tmp_path):
        from interpretune.harness.experiments import load_snapshot_experiment
        from interpretune.hub.components import local_publish

        component = _experiment_component_dir(tmp_path)
        cache = tmp_path / "components"
        local_publish(component, "speediedan/demo-exp", cache_dir=cache)
        key, resolved, snapshot, manifest = load_snapshot_experiment(
            "speediedan/demo-exp", "demo_experiment", cache_dir=cache
        )
        assert key == "demo_experiment" and resolved["EXPERIMENT_NAME"] == "demo_experiment"
        assert manifest["kinds"] == ["experiment"]

    def test_missing_required_component_names_the_fetch(self, tmp_path):
        from interpretune.harness.experiments import load_snapshot_experiment
        from interpretune.hub.components import local_publish

        component = _experiment_component_dir(tmp_path)
        manifest_path = component / "it_component.yaml"
        body = __import__("yaml").safe_load(manifest_path.read_text(encoding="utf-8"))
        body["requires"] = {"interpretune": ">=0.1.dev0", "components": ["speediedan/rte"]}
        manifest_path.write_text(__import__("yaml").safe_dump(body), encoding="utf-8")
        cache = tmp_path / "components"
        local_publish(component, "speediedan/demo-exp", cache_dir=cache)
        with pytest.raises(KeyError, match="speediedan/rte"):
            load_snapshot_experiment("speediedan/demo-exp", "demo_experiment", cache_dir=cache)

    def test_escaping_extends_refused_inside_snapshot(self, tmp_path):
        from interpretune.harness.experiments import load_snapshot_experiment
        from interpretune.hub.components import local_publish

        component = _experiment_component_dir(tmp_path)
        evil = component / "configs" / "demo_experiment.yaml"
        evil.write_text(
            __import__("yaml").safe_dump({"EXPERIMENT_NAME": "demo_experiment", "EXTENDS": "../../outside.yaml"}),
            encoding="utf-8",
        )
        cache = tmp_path / "components"
        local_publish(component, "speediedan/demo-exp", cache_dir=cache)
        with pytest.raises(ValueError, match="outside the component snapshot"):
            load_snapshot_experiment("speediedan/demo-exp", "demo_experiment", cache_dir=cache)

    def test_symlinked_snapshot_files_load_confined(self, tmp_path):
        """Hub snapshots store files as symlinks into a shared `blobs/` directory: following those links answers
        physical storage, not the declared tree, so confinement judges the lexical path and this layout loads
        instead of refusing as an escape."""
        from interpretune.harness.config import load_experiment_config

        root = tmp_path / "snap"
        (root / "configs").mkdir(parents=True)
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "C").write_text(
            __import__("yaml").safe_dump({"EXPERIMENT_NAME": "demo", "EXTENDS": "base.yaml", "A": 1}),
            encoding="utf-8",
        )
        (blobs / "B").write_text(__import__("yaml").safe_dump({"B": 2}), encoding="utf-8")
        (root / "configs" / "child.yaml").symlink_to(blobs / "C")
        (root / "configs" / "base.yaml").symlink_to(blobs / "B")
        resolved = load_experiment_config(root / "configs" / "child.yaml", _root=root)
        assert resolved["EXPERIMENT_NAME"] == "demo"
        assert resolved["A"] == 1 and resolved["B"] == 2

    def test_absolute_extends_outside_snapshot_refused(self, tmp_path):
        """An existing absolute path with a colon takes the path branch, not the exemption.

        Regression: the exemption keyed on `:` in the raw text, so `/tmp/.../esc:ape.yaml`
        loaded while the same path without the colon was refused (and on Windows every
        absolute path has one, voiding the check where CI runs it).
        """
        from interpretune.harness.config import load_experiment_config

        root = tmp_path / "snapshot"
        (root / "configs").mkdir(parents=True)
        outside = tmp_path / "esc:ape.yaml"
        outside.write_text(__import__("yaml").safe_dump({"X": 1}), encoding="utf-8")
        cfg = root / "configs" / "demo_experiment.yaml"
        cfg.write_text(
            __import__("yaml").safe_dump({"EXPERIMENT_NAME": "demo_experiment", "EXTENDS": str(outside)}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="outside the component snapshot"):
            load_experiment_config(cfg, _root=root)

    def test_package_resource_extends_passes_confinement(self, tmp_path):
        """Positive control: the installed-package form is exempt from snapshot confinement."""
        from interpretune.harness.config import load_experiment_config

        root = tmp_path / "snapshot"
        (root / "configs").mkdir(parents=True)
        cfg = root / "configs" / "demo_experiment.yaml"
        cfg.write_text(
            __import__("yaml").safe_dump(
                {"EXPERIMENT_NAME": "demo_experiment", "EXTENDS": "interpretune.harness:configs/base.yaml"}
            ),
            encoding="utf-8",
        )
        resolved = load_experiment_config(cfg, _root=root)
        assert resolved["EXPERIMENT_NAME"] == "demo_experiment"

    def test_pull_path_returns_the_revision_snapshot(self, tmp_path, monkeypatch):
        """The pull payload root is `snapshots/<sha>`, not `snapshots/`: a cross-revision EXTENDS must be refused
        from the returned dir."""
        from interpretune.hub import components as hub_components
        from interpretune.hub.components import pull_experiment_payloads

        sha = "abc123"
        cache = tmp_path / "components"

        def _fake_download(repo_id, rel, *, revision, cache_dir, token, **kw):
            dest = Path(cache_dir) / "models--speediedan--demo-exp" / "snapshots" / revision / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("x: 1\n", encoding="utf-8")
            return str(dest)

        monkeypatch.setattr(hub_components, "hf_hub_download", _fake_download)
        manifest = {
            "experiments": {
                "demo_experiment": {"config": "configs/demo_experiment.yaml", "pipeline": "pipeline/run.py"}
            }
        }
        snapshot = pull_experiment_payloads("speediedan/demo-exp", manifest, "demo_experiment", sha, cache_dir=cache)
        assert snapshot == cache / "models--speediedan--demo-exp" / "snapshots" / sha

        from interpretune.harness.config import load_experiment_config

        evil = snapshot / "configs" / "evil.yaml"
        evil.write_text(
            __import__("yaml").safe_dump({"EXPERIMENT_NAME": "evil", "EXTENDS": "../../other-sha/outside.yaml"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="outside the component snapshot"):
            load_experiment_config(evil, _root=snapshot)

    def test_card_renders_experiment_definitions(self, tmp_path):
        component = _experiment_component_dir(tmp_path)
        manifest = load_component_manifest(component / "it_component.yaml")
        card = generate_component_card(manifest, "speediedan/demo-exp")
        assert "interpretune-experiment" in card.data.tags
        assert "## Experiment definitions" in card.text

    def test_orphan_experiments_block_refused(self):
        manifest = {
            "it_schema_version": 1,
            "kinds": ["module"],
            "module": {"configs": {"rte.gpt2.core": {}}},
            "experiments": {"demo_experiment": {"config": "configs/demo_experiment.yaml"}},
        }
        with pytest.raises(ComponentManifestError, match="orphan"):
            validate_component_manifest(manifest)

    def test_non_yaml_experiment_config_refused(self):
        manifest = {
            "it_schema_version": 1,
            "kinds": ["experiment"],
            "experiments": {"demo_experiment": {"config": "configs/demo_experiment.yml"}},
        }
        with pytest.raises(ComponentManifestError, match="`.yaml`"):
            validate_component_manifest(manifest)

    def test_requires_components_revision_form(self):
        good = {
            "it_schema_version": 1,
            "kinds": ["experiment"],
            "experiments": {"d": {"config": "configs/d.yaml"}},
        }
        validate_component_manifest(dict(good, requires={"components": ["speediedan/rte@738e4122"]}))
        for bad in (["speediedan/rte@rev@extra"], ["not-a-repo-ref@"], ["speediedan/rte@"]):
            with pytest.raises(ComponentManifestError, match="components"):
                validate_component_manifest(dict(good, requires={"components": bad}))

    def test_pinned_required_component_checked_at_its_revision(self, tmp_path):
        from interpretune.harness.experiments import load_snapshot_experiment
        from interpretune.hub.components import local_publish

        component = _experiment_component_dir(tmp_path)
        manifest_path = component / "it_component.yaml"
        body = __import__("yaml").safe_load(manifest_path.read_text(encoding="utf-8"))
        body["requires"] = {"interpretune": ">=0.1.dev0", "components": ["speediedan/rte@deadbeef"]}
        manifest_path.write_text(__import__("yaml").safe_dump(body), encoding="utf-8")
        cache = tmp_path / "components"
        local_publish(component, "speediedan/demo-exp", cache_dir=cache)
        with pytest.raises(KeyError, match="required by experiment `speediedan/demo-exp#demo_experiment`"):
            load_snapshot_experiment("speediedan/demo-exp", "demo_experiment", cache_dir=cache)

    def test_hub_experiment_verbs_resolve(self):
        import interpretune as it

        assert callable(it.hub.pull_experiment) and callable(it.hub.load_experiment)


class TestSnapshotRewrite:
    """Restored snapshot-rewrite machinery for the concept-direction experiment (#498)."""

    def test_unknown_rewrite_name_refused_at_build(self, tmp_path):
        from interpretune.hub.publish import build_component_tree

        component = _experiment_component_dir(tmp_path)
        manifest_path = component / "it_component.yaml"
        body = __import__("yaml").safe_load(manifest_path.read_text(encoding="utf-8"))
        body["experiment_snapshot_rewrite"] = "no-such-rewrite"
        manifest_path.write_text(__import__("yaml").safe_dump(body), encoding="utf-8")
        with pytest.raises(ComponentManifestError, match="unknown `experiment_snapshot_rewrite`"):
            build_component_tree(component, tmp_path / "build")

    def _staged_tree(self, root: Path, *, unmapped: bool = False) -> Path:
        import yaml

        out = root / "staged"
        sources = {
            "concept_direction/concept_direction.py": (
                "from it_examples.experiments.notebook.concept_direction.concept_direction "
                "import NotebookHarnessConfig\nVALUE = 1\n"
            ),
            "pipeline_patterns.py": "PATTERN = True\n",
            "concept_direction/analysis/concept_direction_analysis.py": "ANALYSIS = True\n",
            "concept_direction/analysis/intervention_drift_analysis.py": (
                ("import it_examples.foo\n" if unmapped else "") + "DRIFT = True\n"
            ),
        }
        for rel, text in sources.items():
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        manifest = {
            "experiments": {
                "demo": {
                    "config": "configs/demo.yaml",
                    "pipeline": "concept_direction/concept_direction.py",
                    "files": ["pipeline_patterns.py"],
                }
            },
            "experiment_snapshot_rewrite": "concept-direction-v1",
        }
        (out / "it_component.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
        return out

    def test_concept_direction_rewrite_restructures_and_updates_manifest(self, tmp_path):
        import yaml

        from interpretune.hub.publish import EXPERIMENT_SNAPSHOT_REWRITES

        out = self._staged_tree(tmp_path)
        manifest = yaml.safe_load((out / "it_component.yaml").read_text(encoding="utf-8"))
        EXPERIMENT_SNAPSHOT_REWRITES["concept-direction-v1"](out, manifest)

        assert (out / "exp" / "concept_direction.py").is_file()
        assert (out / "exp" / "pipeline_patterns.py").is_file()
        assert (out / "exp" / "analysis" / "concept_direction_analysis.py").is_file()
        assert (out / "exp" / "_prompt_shim.py").is_file()
        assert not (out / "concept_direction" / "concept_direction.py").exists()
        assert not (out / "pipeline_patterns.py").exists()
        rewritten = (out / "exp" / "concept_direction.py").read_text(encoding="utf-8")
        assert "from exp.concept_direction import NotebookHarnessConfig" in rewritten
        assert "it_examples" not in rewritten
        entry = manifest["experiments"]["demo"]
        assert entry["pipeline"] == "exp/concept_direction.py"
        assert "exp/pipeline_patterns.py" in entry["files"]
        assert "exp/_prompt_shim.py" in entry["files"]
        assert manifest["experiment_snapshot_rewrite"] == "concept-direction-v1"

    def test_unmapped_it_examples_import_refused(self, tmp_path):
        import yaml

        from interpretune.hub.publish import EXPERIMENT_SNAPSHOT_REWRITES

        out = self._staged_tree(tmp_path, unmapped=True)
        manifest = yaml.safe_load((out / "it_component.yaml").read_text(encoding="utf-8"))
        with pytest.raises(ValueError, match="refuses unmapped it_examples imports"):
            EXPERIMENT_SNAPSHOT_REWRITES["concept-direction-v1"](out, manifest)
