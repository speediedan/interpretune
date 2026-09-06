"""#375 / #444: per-architecture component maps as a DATA hub kind (``hookmaps``).

Pins the kind contract: manifest shape, publish-time structural check, cache-only registration through the
vocabulary's own registry (so a hub map is indistinguishable from a bundled one to every consumer), the
agree-or-refuse collision rule for architectures already registered, and the absence of a trust gate.
"""

from __future__ import annotations

import pytest
import yaml

from interpretune.hub.manifest import ComponentManifestError, validate_component_manifest

GPT2_ROWS = {
    "embed": {"module": "transformer.wte", "kind": "embed"},
    "blocks.{i}": {"module": "transformer.h.{i}", "kind": "block"},
    "blocks.{i}.ln1": {"module": "transformer.h.{i}.ln_1", "kind": "norm"},
    "blocks.{i}.attn": {"module": "transformer.h.{i}.attn", "kind": "attn"},
    "blocks.{i}.mlp": {"module": "transformer.h.{i}.mlp", "kind": "mlp"},
    "ln_final": {"module": "transformer.ln_f", "kind": "norm"},
    "unembed": {"module": "lm_head", "kind": "unembed"},
}


def _manifest(**overrides):
    base = {"it_schema_version": 1, "kinds": ["hookmaps"], "hookmaps": {"files": ["maps/toy.yaml"]}}
    base.update(overrides)
    return base


def _component(tmp_path, architecture="ToyForCausalLM", rows=None, facts=None, name="toy.yaml"):
    """An in-tree-shaped hookmaps component directory with one document."""
    root = tmp_path / "component"
    (root / "maps").mkdir(parents=True)
    (root / "it_component.yaml").write_text(yaml.safe_dump(_manifest(hookmaps={"files": [f"maps/{name}"]})))
    doc = {"schema_version": 1, "architecture": architecture, "components": rows or GPT2_ROWS}
    if facts is not None:
        doc["facts"] = facts
    (root / "maps" / name).write_text(yaml.safe_dump(doc))
    return root


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Every test starts from the bundled maps alone and leaves nothing behind."""
    from interpretune.analysis.points import component_map as cm

    cm._load_bundled()  # save AFTER the bundled maps are in, or the restore leaves an empty registry marked loaded
    saved = dict(cm._REGISTRY)
    yield
    cm._REGISTRY.clear()
    cm._REGISTRY.update(saved)


class TestHookMapsKindManifest:
    def test_kind_requires_a_files_list(self):
        with pytest.raises(ComponentManifestError, match="requires a non-empty `hookmaps.files` list"):
            validate_component_manifest(_manifest(hookmaps={"files": []}), source="t")
        with pytest.raises(ComponentManifestError, match="requires a non-empty `hookmaps.files` list"):
            validate_component_manifest(_manifest(hookmaps=None), source="t")

    def test_files_must_not_include_the_manifest(self):
        with pytest.raises(ComponentManifestError, match="must not list it_component.yaml"):
            validate_component_manifest(_manifest(hookmaps={"files": ["it_component.yaml"]}), source="t")

    def test_valid_manifest_and_card(self):
        from interpretune.hub.cards import generate_component_card

        manifest = validate_component_manifest(_manifest(), source="t")
        card = str(generate_component_card(manifest, "org/maps"))
        assert "interpretune-hookmaps" in card
        assert "## Component maps (hookmaps)" in card and "maps/toy.yaml" in card
        assert "pull_hookmaps" in card and "no code executes" in card


class TestPublishStructuralCheck:
    def test_a_missing_document_is_refused(self, tmp_path):
        from interpretune.hub.publish import build_component_tree

        root = _component(tmp_path)
        (root / "maps" / "toy.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="hookmaps document 'maps/toy.yaml'"):
            build_component_tree(root, tmp_path / "out")

    def test_a_document_that_is_not_a_map_is_refused(self, tmp_path):
        from interpretune.hub.publish import build_component_tree

        root = _component(tmp_path)
        (root / "maps" / "toy.yaml").write_text(
            yaml.safe_dump({"schema_version": 1, "architecture": "X"})
        )  # no components
        with pytest.raises(ValueError, match="missing the 'components' key"):
            build_component_tree(root, tmp_path / "out")
        (root / "maps" / "toy.yaml").write_text(
            yaml.safe_dump(
                {"schema_version": 1, "architecture": "X", "components": {"embed": {"module": "m", "kind": "sorcery"}}}
            )
        )
        with pytest.raises(ValueError, match="unknown component kind 'sorcery'"):
            build_component_tree(root, tmp_path / "out")

    def test_a_valid_document_is_staged_verbatim(self, tmp_path):
        from interpretune.hub.publish import build_component_tree

        root = _component(tmp_path)
        out = tmp_path / "out"
        build_component_tree(root, out)
        assert (out / "maps" / "toy.yaml").read_bytes() == (root / "maps" / "toy.yaml").read_bytes()


class TestCacheOnlyRegistration:
    def test_load_registers_a_new_architecture_for_every_consumer(self, tmp_path, monkeypatch):
        import socket

        import interpretune as it
        from interpretune.analysis.points.component_map import component_map_for, known_architectures
        from interpretune.hub.components import local_publish

        cache = tmp_path / "cache"
        local_publish(_component(tmp_path), "org/maps", cache_dir=cache)
        monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        monkeypatch.delenv("IT_TRUST_REMOTE_CODE", raising=False)  # data kind: no trust gate to satisfy

        assert "ToyForCausalLM" not in known_architectures()
        assert it.hub.load_hookmaps("org/maps", cache_dir=cache) == ["ToyForCausalLM"]
        cmap = component_map_for("ToyForCausalLM")
        assert cmap.components["blocks.{i}.mlp"].module == "transformer.h.{i}.mlp"
        assert cmap.source.startswith("org/maps@") and cmap.source.endswith(":toy.yaml")
        # the point resolver is one such consumer: a component point resolves through the hub map
        from interpretune.analysis.points.resolution import TensorRef, resolve
        from interpretune.analysis.points.vocabulary import parse

        ref = resolve(parse("blocks.3.mlp.hook_out"), cmap)
        assert isinstance(ref, TensorRef) and ref.module_path == "transformer.h.3.mlp" and ref.io == "output"

    def test_uncached_component_names_the_fetch(self, tmp_path):
        import interpretune as it

        with pytest.raises(KeyError, match=r"not in the local cache.*interpretune.hub.pull\('org/absent'\)"):
            it.hub.load_hookmaps("org/absent", cache_dir=tmp_path)

    def test_manifest_only_snapshot_names_the_materializing_verb(self, tmp_path):
        from interpretune.hub.components import local_publish
        from interpretune.hub.hookmaps import HookMapComponentError, load_hub_hookmaps

        cache = tmp_path / "cache"
        local_publish(_component(tmp_path), "org/maps", cache_dir=cache)
        snapshots = cache / "models--org--maps" / "snapshots"
        for doc in snapshots.rglob("toy.yaml"):
            doc.unlink()
        with pytest.raises(HookMapComponentError, match=r"not present in the snapshot.*pull_hookmaps\('org/maps'\)"):
            load_hub_hookmaps("org/maps", cache_dir=cache)

    def test_component_without_the_kind_is_refused(self, tmp_path):
        from interpretune.hub.hookmaps import HookMapComponentError, declared_hookmap_files

        with pytest.raises(HookMapComponentError, match="does not declare the `hookmaps` kind"):
            declared_hookmap_files({"it_schema_version": 1, "kinds": ["module"]}, source="t")


class TestCollisionRule:
    def _bundled_gpt2_document(self):
        from interpretune.analysis.points.component_map import component_map_for

        bundled = component_map_for("GPT2LMHeadModel")
        rows = {name: {"module": e.module, "kind": e.kind} for name, e in bundled.components.items()}
        return rows, dict(bundled.facts), bundled

    def test_an_identical_republished_map_is_a_no_op(self, tmp_path):
        import interpretune as it
        from interpretune.analysis.points.component_map import component_map_for
        from interpretune.hub.components import local_publish

        rows, facts, bundled = self._bundled_gpt2_document()
        cache = tmp_path / "cache"
        local_publish(_component(tmp_path, "GPT2LMHeadModel", rows, facts), "org/maps", cache_dir=cache)
        assert it.hub.load_hookmaps("org/maps", cache_dir=cache) == ["GPT2LMHeadModel"]
        assert component_map_for("GPT2LMHeadModel") is bundled, "an agreeing hub map must not displace the bundled one"

    def test_a_disagreeing_map_is_refused_unless_replace(self, tmp_path):
        import interpretune as it
        from interpretune.analysis.points.component_map import component_map_for
        from interpretune.hub.components import local_publish
        from interpretune.hub.hookmaps import HookMapComponentError

        rows, facts, _ = self._bundled_gpt2_document()
        rows["blocks.{i}.mlp"] = {"module": "transformer.h.{i}.feed_forward", "kind": "mlp"}
        cache = tmp_path / "cache"
        local_publish(_component(tmp_path, "GPT2LMHeadModel", rows, facts), "org/maps", cache_dir=cache)
        with pytest.raises(HookMapComponentError, match="disagrees with the one already registered.*replace=True"):
            it.hub.load_hookmaps("org/maps", cache_dir=cache)
        assert component_map_for("GPT2LMHeadModel").components["blocks.{i}.mlp"].module == "transformer.h.{i}.mlp"
        assert it.hub.load_hookmaps("org/maps", cache_dir=cache, replace=True) == ["GPT2LMHeadModel"]
        assert component_map_for("GPT2LMHeadModel").components["blocks.{i}.mlp"].module == (
            "transformer.h.{i}.feed_forward"
        )


class TestPullMaterializesDeclaredPayloads:
    """The download half: a key-less ``it.hub.pull`` leaves a snapshot the cache-only loaders can complete."""

    def test_declared_payloads_cover_every_cache_only_loader(self):
        from interpretune.hub.components import declared_component_payloads

        manifest = {
            "kinds": ["adapters", "promptconfigs", "hookmaps", "ops", "module"],
            "adapters": {"entrypoint": "adapter.py", "declares": ["x"]},
            "promptconfigs": {"entrypoint": "prompts.py", "definitions": {"d": {}}},
            "hookmaps": {"files": ["maps/a.yaml", "maps/b.yaml"]},
            "ops": {"files": ["ops.yaml"]},  # own cache and verb: never fetched here
            "module": {"configs": {"k": "configs/k.yaml"}},  # fetched by key, not wholesale
        }
        assert declared_component_payloads(manifest) == ["adapter.py", "prompts.py", "maps/a.yaml", "maps/b.yaml"]

    def test_pull_fetches_the_payloads_pinned_to_the_manifest_commit(self, tmp_path, monkeypatch):
        import interpretune as it
        from interpretune.hub import components

        root = _component(tmp_path)
        calls = []

        def fake_download(repo_id, filename, revision=None, cache_dir=None, token=None, **_):
            calls.append((filename, revision))
            snap = tmp_path / "cache" / "models--org--maps" / "snapshots" / "abc123def456"
            dest = snap / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((root / filename).read_bytes())
            return str(dest)

        monkeypatch.setattr(components, "hf_hub_download", fake_download)
        manifest, commit = it.hub.pull("org/maps", cache_dir=tmp_path / "cache")
        assert commit == "abc123def456" and manifest["kinds"] == ["hookmaps"]
        assert calls == [("it_component.yaml", None), ("maps/toy.yaml", "abc123def456")]

    def test_pull_hookmaps_fetches_then_registers(self, tmp_path, monkeypatch):
        import interpretune as it
        from interpretune.analysis.points.component_map import known_architectures
        from interpretune.hub import components

        root = _component(tmp_path)
        cache = tmp_path / "cache"
        snap = cache / "models--org--maps" / "snapshots" / "abc123def456"

        def fake_download(repo_id, filename, revision=None, cache_dir=None, token=None, **_):
            dest = snap / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((root / filename).read_bytes())
            (snap.parent.parent / "refs").mkdir(exist_ok=True)
            (snap.parent.parent / "refs" / "main").write_text("abc123def456")
            return str(dest)

        monkeypatch.setattr(components, "hf_hub_download", fake_download)
        paths, commit = it.hub.pull_hookmaps("org/maps", cache_dir=cache)
        assert [p.name for p in paths] == ["toy.yaml"] and commit == "abc123def456"
        assert "ToyForCausalLM" in known_architectures()
