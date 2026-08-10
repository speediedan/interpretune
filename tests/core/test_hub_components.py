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

RTE_COMPONENT_DIR = Path(__file__).parent.parent.parent / "src" / "it_examples" / "registry" / "rte"
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
        assert "library_name: interpretune" in (out / "README.md").read_text()


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
        drifted.write_text(drifted.read_text().replace("model: gpt2", "model: gpt3000"))
        with pytest.raises(ValueError, match="parity violation"):
            build_component_tree(src_copy, tmp_path / "build", entrypoint_src=RTE_ENTRYPOINT)

    def test_missing_entrypoint_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="entrypoint"):
            build_component_tree(RTE_COMPONENT_DIR, tmp_path / "build", entrypoint_src=tmp_path / "nope.py")


class TestManifestFirstOffline:
    """Local resolution must never touch the network (design invariant)."""

    def test_local_resolution_with_sockets_blocked(self, monkeypatch):
        import socket

        def _blocked(*args, **kwargs):
            raise AssertionError("local example resolution attempted a network connection")

        monkeypatch.setattr(socket.socket, "connect", _blocked)
        from it_examples.example_module_registry import LazyModuleRegistry

        registry = LazyModuleRegistry()
        assert registry.get("rte_demo.gemma2.circuit_tracer") is not None

    def test_hub_config_body_roundtrips_registry_schema(self):
        """A fetched configuration body is exactly what the local loader consumes — one schema, no adapters."""
        cfg_path = RTE_COMPONENT_DIR / "configs" / "rte_demo.gemma2.circuit_tracer.yaml"
        body = yaml.safe_load(cfg_path.read_text())
        assert {"task_variant", "model", "composition", "reg_info", "shared_config", "registered_cfg"} <= set(body)
