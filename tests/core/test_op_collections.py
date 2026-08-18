"""The ``collection:`` header an op YAML may declare, and its compatibility window (#266 Phase 3, D8).

A collection versions a *contract set* -- the names, schemas and traits its ops present to callers -- and declares one
window against the installed interpretune. There is no cross-collection resolution and no solver: an incompatible
collection is skipped whole with a warning, or raises under ``IT_STRICT_OP_LOAD=1``.

The window is skipped *whole* rather than per-op on purpose: compatibility is declared once per collection, so a
partial load would present half a contract set, which is harder to diagnose than an absent one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from interpretune.analysis.ops.collection import COLLECTION_HEADER_KEY, CollectionSpec
from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

BUNDLED_ROOT = Path(__file__).parent.parent.parent / "src" / "interpretune" / "analysis" / "ops" / "bundled"
_IMPL = "interpretune.analysis.ops.bundled.core.core_ops.model_fwd_impl"


def _bundled_family_yamls() -> list[Path]:
    yamls = sorted(BUNDLED_ROOT.glob("*/*.yaml"))
    assert yamls, f"no bundled family YAMLs under {BUNDLED_ROOT}"
    return [p for p in yamls if p.name != "composites.yaml"]


def _collection_yaml(header: str, op_name: str = "my_collection_op") -> str:
    return (
        f"{header}\n{op_name}:\n"
        f"  description: fixture op\n"
        f"  implementation: {_IMPL}\n"
        f"  input_schema: {{}}\n"
        f"  output_schema: {{}}\n"
    )


def _load(tmp_path, text: str) -> AnalysisOpDispatcher:
    op_dir = tmp_path / "collection"
    op_dir.mkdir()
    (op_dir / "ops.yaml").write_text(text)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dispatcher = AnalysisOpDispatcher(yaml_paths=[op_dir], enable_hub_ops=False)
    dispatcher._cache_manager.cache_dir = cache_dir
    return dispatcher


class TestCollectionSpec:
    def test_minimal_header(self):
        spec = CollectionSpec.from_raw({"name": "concept", "version": "0.3.0"})
        assert (spec.name, spec.version, spec.requires) == ("concept", "0.3.0", {})

    def test_none_means_no_header(self):
        assert CollectionSpec.from_raw(None) is None

    @pytest.mark.parametrize(
        "raw, match",
        [
            ({"version": "1.0"}, "requires a `name`"),
            ({"name": "c"}, "requires a `version`"),
            ({"name": "c", "version": "not-a-version"}, "PEP 440"),
            ({"name": "", "version": "1.0"}, "non-empty string"),
            ({"name": "c", "version": "1.0", "extra": 1}, "Unrecognized"),
            ("just a string", "must be a mapping"),
        ],
    )
    def test_malformed_headers_rejected(self, raw, match):
        with pytest.raises(ValueError, match=match):
            CollectionSpec.from_raw(raw)

    def test_requires_must_use_the_component_grammar(self):
        """A bare list appears in early design notes; the shared enforcement machinery reads a mapping."""
        with pytest.raises(ValueError, match="must be a mapping"):
            CollectionSpec.from_raw({"name": "c", "version": "1.0", "requires": ["interpretune>=0.1"]})

    def test_round_trips_through_to_dict(self):
        raw = {"name": "c", "version": "1.2.3", "requires": {"interpretune": ">=0.1.0.dev0"}}
        assert CollectionSpec.from_raw(raw).to_dict() == raw


class TestCompatibilityWindow:
    def test_no_window_is_always_compatible(self):
        assert CollectionSpec(name="c", version="1.0", requires={}).incompatibility() is None

    def test_satisfiable_window_is_compatible(self):
        spec = CollectionSpec(name="c", version="1.0", requires={"interpretune": ">=0.1.0.dev0"})
        assert spec.incompatibility() is None

    def test_unsatisfiable_window_explains_itself(self):
        spec = CollectionSpec(name="c", version="1.0", requires={"interpretune": ">=99"})
        message = spec.incompatibility()
        assert message is not None and ">=99" in message

    def test_incompatible_collection_is_skipped_whole(self, tmp_path):
        header = 'collection:\n  name: my_collection\n  version: 1.2.3\n  requires:\n    interpretune: ">=99"\n'
        dispatcher = _load(tmp_path, _collection_yaml(header))
        with pytest.warns(UserWarning, match="Skipping op collection"):
            dispatcher.load_definitions()

        assert "my_collection_op" not in dispatcher._op_definitions
        assert "model_fwd" in dispatcher._op_definitions, "one bad collection must not take the session down"

    def test_compatible_collection_loads(self, tmp_path):
        header = 'collection:\n  name: my_collection\n  version: 1.2.3\n  requires:\n    interpretune: ">=0.1.0.dev0"\n'
        dispatcher = _load(tmp_path, _collection_yaml(header))
        dispatcher.load_definitions()
        assert "my_collection_op" in dispatcher._op_definitions

    def test_incompatible_collection_raises_under_strict_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        header = 'collection:\n  name: my_collection\n  version: 1.2.3\n  requires:\n    interpretune: ">=99"\n'
        dispatcher = _load(tmp_path, _collection_yaml(header))
        with pytest.raises(OpLoadError, match="Skipping op collection"):
            dispatcher.load_definitions()

    def test_malformed_header_is_fail_soft(self, tmp_path):
        """The ops still load, just without collection identity: a bad header is not a reason to lose the ops."""
        dispatcher = _load(tmp_path, _collection_yaml("collection:\n  bogus_key: true\n"))
        with pytest.warns(UserWarning, match="Ignoring invalid `collection` header"):
            dispatcher.load_definitions()
        assert "my_collection_op" in dispatcher._op_definitions


class TestHeaderIsNotAnOperation:
    def test_collection_key_is_not_registered_as_an_op(self, tmp_path):
        header = "collection:\n  name: my_collection\n  version: 1.2.3\n"
        dispatcher = _load(tmp_path, _collection_yaml(header))
        dispatcher.load_definitions()
        assert COLLECTION_HEADER_KEY not in dispatcher._op_definitions
        assert "my_collection_op" in dispatcher._op_definitions

    def test_the_header_is_not_registered_as_an_op_on_the_HUB_path_either(self, tmp_path, monkeypatch):
        """The header is dropped BEFORE hub namespacing, which is where this went wrong.

        Namespacing rewrites every top-level key, so the header arrived as ``<user>.<repo>.collection`` and no
        longer matched the header key -- it was then registered as an op, giving every hub collection a junk
        ``collection`` op whose "definition" was the header mapping. Bundled files are not namespaced, which is
        why the equality check held there and hid it; the first evidence was a generated stub declaring a
        ``collection`` function.
        """
        from tests.hub_op_fixtures import write_cached_op_collection

        hub_cache = tmp_path / "hub"
        header = "collection:\n  name: hub_collection\n  version: 2.0.0\n"
        write_cached_op_collection(
            hub_cache, repo_id="someone/ops", op_files={"ops.yaml": _collection_yaml(header, "hub_op")}
        )
        monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache)
        monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_OP_PATHS", [])

        dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
        (tmp_path / "cache").mkdir()
        dispatcher._cache_manager.cache_dir = tmp_path / "cache"
        dispatcher.load_definitions()

        assert "someone.ops.hub_op" in dispatcher._op_definitions, "fixture collection failed to load"
        assert not [name for name in dispatcher._op_definitions if name.split(".")[-1] == COLLECTION_HEADER_KEY]
        # and the header still parsed, so collection identity is intact
        assert dispatcher._op_definitions["someone.ops.hub_op"].collection_name == "hub_collection"


class TestBundledFamiliesDeclareCollections:
    """The CI assertion D8 asks for, in the form that actually protects.

    A naive window is worse than none here: ``setuptools_scm`` produces ``0.1.0.devN+g<sha>`` between tags, and a
    ``>=0.1`` floor does NOT match that (a dev release sorts *before* its release), so declaring the window from D8's
    illustrative snippet would skip every bundled op in any source checkout, including CI. Bundled families
    therefore declare no window -- they ship in the wheel, so the window is vacuous by construction -- and this test
    fails loudly if one is ever added that the installed interpretune does not satisfy.
    """

    @pytest.mark.parametrize("yaml_path", _bundled_family_yamls(), ids=lambda p: p.parent.name)
    def test_family_declares_a_valid_collection_header(self, yaml_path):
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        spec = CollectionSpec.from_raw(content.get(COLLECTION_HEADER_KEY))
        assert spec is not None, f"{yaml_path.parent.name} declares no `{COLLECTION_HEADER_KEY}` header"
        assert spec.name == yaml_path.parent.name, "collection name should match the family directory"

    @pytest.mark.parametrize("yaml_path", _bundled_family_yamls(), ids=lambda p: p.parent.name)
    def test_family_window_is_satisfied_by_the_installed_interpretune(self, yaml_path):
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        spec = CollectionSpec.from_raw(content.get(COLLECTION_HEADER_KEY))
        assert spec is not None
        assert spec.incompatibility() is None, (
            f"{yaml_path.parent.name} declares a window the installed interpretune does not satisfy, so its ops "
            "would be silently skipped. Note a `>=0.1`-style floor does not match a `0.1.0.devN` source install."
        )

    def test_every_family_is_discovered_with_its_collection(self):
        """Asserted on the OpDef fields, NOT on ``_op_collections``.

        ``_op_collections`` is populated only while YAMLs are compiled, so on a warm cache it is empty and the same
        assertion against it passes cold and fails warm. That is precisely why collection identity is a cache-
        serialized ``OpDef`` field: it has to survive the cache to be reportable at runtime.
        """
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        declared = {op_def.collection_name for op_def in DISPATCHER._op_definitions.values() if op_def.collection_name}
        assert {p.parent.name for p in _bundled_family_yamls()} <= declared

    def test_collection_identity_survives_the_cache(self, tmp_path):
        """A warm load must still report which collection an op came from, and at what version."""
        header = "collection:\n  name: my_collection\n  version: 4.5.6\n"
        first = _load(tmp_path, _collection_yaml(header))
        first.load_definitions()
        assert (first._op_definitions["my_collection_op"].collection_name) == "my_collection"

        # Same YAML dir and cache dir: this load hits the cache rather than recompiling.
        second = AnalysisOpDispatcher(yaml_paths=[tmp_path / "collection"], enable_hub_ops=False)
        second._cache_manager.cache_dir = tmp_path / "cache"
        second.load_definitions()
        assert not second._op_collections, "expected a cache hit (no compile-time collection parsing)"
        cached = second._op_definitions["my_collection_op"]
        assert (cached.collection_name, cached.collection_version) == ("my_collection", "4.5.6")
