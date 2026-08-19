"""Op-name normalization must be symmetric between storage and lookup, and collisions must be loud.

``AnalysisOpDispatcher._normalize_op_name`` case-folds and maps ``-``/``/``, and its docstring has always said it does
so "for consistent lookup". Only the storage half implemented that: definitions registered under the normalized name
while ``_resolve_name_safe`` did not normalize, so an op declared ``my-hyphen-op`` or ``MyCasedOp`` was unreachable by
the name its own author wrote and ``get_op`` raised ``Unknown operation`` with nothing warned at load (#266 Phase 3,
D12).

Fixing lookup alone would be a regression rather than an improvement: two declared names that normalize to one key
collapse silently, last-writer-wins, so a symmetric lookup would *succeed* and hand back a different collection's op --
turning an error path into a wrong-answer path. The collision check therefore lands with the symmetry fix, and these
tests pin both halves together.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

# A real bundled impl, so definitions instantiate rather than failing for an unrelated reason.
_IMPL = "interpretune.analysis.ops.bundled.core.core_ops.model_fwd_impl"


def _op_yaml(name: str) -> str:
    return (
        f"\n{name}:\n"
        f"  description: fixture op for normalization tests\n"
        f"  implementation: {_IMPL}\n"
        f"  input_schema: {{}}\n"
        f"  output_schema: {{}}\n"
    )


def _collection(root, dir_name: str, yaml_text: str):
    """Write a one-file op collection and return its directory."""
    op_dir = root / dir_name
    op_dir.mkdir()
    (op_dir / f"{dir_name}.yaml").write_text(yaml_text)
    return op_dir


def _dispatcher(cache_dir, *op_dirs) -> AnalysisOpDispatcher:
    dispatcher = AnalysisOpDispatcher(yaml_paths=list(op_dirs), enable_hub_ops=False)
    dispatcher._cache_manager.cache_dir = cache_dir
    return dispatcher


@pytest.fixture
def cache_dir(tmp_path):
    """A per-test cache dir; the shared default would leak compiled definitions between tests."""
    path = tmp_path / "op_cache"
    path.mkdir()
    return path


class TestLookupIsSymmetricWithStorage:
    """Every transform ``_normalize_op_name`` applies must be applied on the lookup side too."""

    @pytest.mark.parametrize(
        "declared, normalized",
        [
            ("my-hyphen-op", "my_hyphen_op"),  # '-' -> '_'
            ("MyCasedOp", "mycasedop"),  # case-fold
            ("some/slashed_op", "some.slashed_op"),  # '/' -> '.'
            ("plain_op", "plain_op"),  # control: unchanged by normalization
        ],
    )
    def test_declared_name_resolves(self, tmp_path, cache_dir, declared, normalized):
        """The author's own spelling must work.

        This is the defect: it did not.
        """
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", _op_yaml(declared)))
        dispatcher.load_definitions()

        assert normalized in dispatcher._op_definitions, "registration should still normalize"
        assert dispatcher._resolve_name_safe(declared) == normalized
        assert dispatcher.get_op(declared) is not None

    @pytest.mark.parametrize("probe", ["my_hyphen_op", "MY-HYPHEN-OP", "my-hyphen-op"])
    def test_all_spellings_of_one_op_resolve(self, tmp_path, cache_dir, probe):
        """Normalization is many-to-one, so every spelling that normalizes alike must resolve alike."""
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", _op_yaml("my-hyphen-op")))
        dispatcher.load_definitions()
        assert dispatcher.get_op(probe) is not None

    def test_genuinely_unknown_names_still_raise(self, tmp_path, cache_dir):
        """Symmetry must not degrade into resolving anything: an absent op is still an error."""
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", _op_yaml("plain_op")))
        dispatcher.load_definitions()
        with pytest.raises(ValueError, match="Unknown operation"):
            dispatcher.get_op("no_such_op")


class TestNormalizationCollisionsAreReported:
    """Two declared names collapsing to one key is an authoring error, not a silent overwrite."""

    _COLLIDING = _op_yaml("my-collide-op") + _op_yaml("my_collide_op")

    def test_collision_within_one_file_warns(self, tmp_path, cache_dir):
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", self._COLLIDING))
        with pytest.warns(UserWarning, match="Operation name collision"):
            dispatcher.load_definitions()

    def test_collision_across_two_collections_warns(self, tmp_path, cache_dir):
        """The realistic case: no single file holds both, so a per-file check cannot see it.

        ``IT_ANALYSIS_OP_PATHS`` is colon-separated, so two local collections is ordinary configuration.
        """
        first = _collection(tmp_path, "coll_a", _op_yaml("my-collide-op"))
        second = _collection(tmp_path, "coll_b", _op_yaml("my_collide_op"))
        dispatcher = _dispatcher(cache_dir, first, second)
        with pytest.warns(UserWarning, match="Operation name collision") as caught:
            dispatcher.load_definitions()

        message = str(caught[0].message)
        # Both sides must be nameable. OpDef.source cannot do this: it is a category
        # (bundled | local | hub:<user.repo>), so two local collections both report "local".
        assert "my-collide-op" in message and "my_collide_op" in message
        assert str(first) in message and str(second) in message

    def test_collision_raises_under_strict_load(self, tmp_path, cache_dir, monkeypatch):
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", self._COLLIDING))
        with pytest.raises(OpLoadError, match="Operation name collision"):
            dispatcher.load_definitions()

    def test_distinct_names_do_not_collide(self, tmp_path, cache_dir, recwarn):
        """Control: names that normalize differently must not be reported."""
        yaml_text = _op_yaml("first_op") + _op_yaml("second-op")
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", yaml_text))
        dispatcher.load_definitions()

        assert {"first_op", "second_op"} <= set(dispatcher._op_definitions)
        assert not [w for w in recwarn if "collision" in str(w.message)]

    def test_declaration_sites_are_not_carried_on_opdef(self, tmp_path, cache_dir):
        """The path lives in a load-time side map, deliberately not on the cache-serialized OpDef.

        ``OpDef`` is serialized into the generated cache, so a field here would force a
        ``CACHE_FORMAT_VERSION`` bump and invalidate every user's cache for a diagnostics-only string.
        """
        from dataclasses import fields

        from interpretune.analysis.ops.compiler.cache_manager import OpDef

        assert not [f.name for f in fields(OpDef) if "path" in f.name or "file" in f.name]

        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", _op_yaml("plain_op")))
        dispatcher.load_definitions()
        assert dispatcher._op_declaration_sites["plain_op"].endswith("ops.yaml")


class TestStrictLoadVetoesTheCache:
    """``IT_STRICT_OP_LOAD`` has to recompile, or a warm cache silently disables every check it guards."""

    def test_warm_cache_does_not_defeat_strict_load(self, tmp_path, cache_dir, monkeypatch):
        """Measured regression: this previously loaded clean and reported nothing.

        Every check routed through ``op_load_failure`` runs while definitions are COMPILED -- failed compiles,
        unresolvable ``importable_params``, invalid ``op_state``, unsanctioned hub params, name collisions -- so
        reading a precompiled artifact skips all of them. A non-strict run would warn once, cache, and every
        subsequent strict run would reuse that cache and pass.
        """
        op_dir = _collection(tmp_path, "ops", _op_yaml("my-collide-op") + _op_yaml("my_collide_op"))

        # Warm the cache with a non-strict load.
        with pytest.warns(UserWarning, match="Operation name collision"):
            _dispatcher(cache_dir, op_dir).load_definitions()
        assert list(cache_dir.glob("op_definitions_*.py")), "expected the non-strict load to cache"

        # Same cache dir, strict enabled: must still fail.
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        with pytest.raises(OpLoadError, match="Operation name collision"):
            _dispatcher(cache_dir, op_dir).load_definitions()

    def test_non_strict_loads_still_use_the_cache(self, tmp_path, cache_dir, recwarn):
        """The veto is scoped to strict mode; ordinary sessions must keep their cache hit."""
        op_dir = _collection(tmp_path, "ops", _op_yaml("plain_op"))
        _dispatcher(cache_dir, op_dir).load_definitions()

        second = _dispatcher(cache_dir, op_dir)
        second.load_definitions()
        assert "plain_op" in second._op_definitions
        assert not [w for w in recwarn if "collision" in str(w.message)]


class TestNonOpYamlIsNotParsedAsOps:
    """A YAML that is not an op-definitions file must not be able to drop every op in the process.

    ``it_component.yaml`` sits at the root of every interpretune component repo, op collections included, and shares
    the ``.yaml`` suffix that discovery keys on. Feeding it to the op compiler raised ``AttributeError: 'int' object
    has no attribute 'get'`` on its own scalar keys (``it_schema_version: 1``) from
    ``_compile_required_ops_schemas``, which catches only ``ValueError`` -- so the whole load died and
    ``_op_definitions`` came back EMPTY, bundled ops included (#266 Phase 3).
    """

    _MANIFEST = "it_schema_version: 1\nkinds: [ops]\nops:\n  files: [my_ops.yaml]\n"

    def test_a_collection_dir_with_its_manifest_loads_cleanly(self, tmp_path, cache_dir, recwarn):
        """The ordinary local-authoring shape: one manifest plus the op YAMLs it declares, same directory."""
        op_dir = _collection(tmp_path, "collection", _op_yaml("my_op"))
        (op_dir / "it_component.yaml").write_text(self._MANIFEST)

        dispatcher = _dispatcher(cache_dir, op_dir)
        dispatcher.load_definitions()

        assert "my_op" in dispatcher._op_definitions
        assert "model_fwd" in dispatcher._op_definitions, "bundled ops must survive"
        # None of the manifest's own keys may be registered as operations.
        assert not {"it_schema_version", "kinds", "ops"} & set(dispatcher._op_definitions)
        assert not [w for w in recwarn if "mapping" in str(w.message)]

    def test_manifest_is_excluded_from_discovery(self, tmp_path, cache_dir):
        op_dir = _collection(tmp_path, "collection", _op_yaml("my_op"))
        (op_dir / "it_component.yaml").write_text(self._MANIFEST)

        discovered = _dispatcher(cache_dir, op_dir)._discover_yaml_files([op_dir])
        assert [p.name for p in discovered] == ["collection.yaml"]

    def test_scalar_entry_is_skipped_not_fatal(self, tmp_path, cache_dir):
        """Defence in depth for any other non-op YAML: contain the bad entry, keep the rest."""
        yaml_text = "some_scalar: 1\n" + _op_yaml("good_op")
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", yaml_text))
        with pytest.warns(UserWarning, match="must be a mapping"):
            dispatcher.load_definitions()

        assert "good_op" in dispatcher._op_definitions
        assert "model_fwd" in dispatcher._op_definitions

    def test_scalar_entry_raises_under_strict_load(self, tmp_path, cache_dir, monkeypatch):
        """Must survive the per-file fail-soft handler, which would otherwise swallow it to a debug line."""
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        yaml_text = "some_scalar: 1\n" + _op_yaml("good_op")
        dispatcher = _dispatcher(cache_dir, _collection(tmp_path, "ops", yaml_text))
        with pytest.raises(OpLoadError, match="must be a mapping"):
            dispatcher.load_definitions()


class TestBundledOpsAreUnaffected:
    """The bundled set must neither collide nor change shape under the symmetric lookup."""

    def test_no_bundled_name_collisions(self, recwarn):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        assert not [w for w in recwarn if "collision" in str(w.message)]

    def test_bundled_names_are_already_normalized(self):
        """If a bundled op needed normalizing, its declared and registered names would differ."""
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        assert all(name == DISPATCHER._normalize_op_name(name) for name in DISPATCHER._op_definitions)
