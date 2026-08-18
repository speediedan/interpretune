"""Cross-collection ``required_ops`` declarability and its failure mode (#266 Phase 3, §3.6a item 1).

§3.6a recorded this as an open liftability gap against `7b44ba5`: `sae` and `attribution` both require
`get_answer_indices`, which only `core` defines, and an op whose `required_ops` do not resolve was
**popped behind a warning** -- so a family published alone would silently lose ops. The exit criterion is
"cross-family ``required_ops`` declarable with a loud failure mode".

These tests establish which half needed building and which was already true, because asserting a criterion
satisfied without measuring it is how a gap gets closed on paper only:

- **Declarable** was already true, and by a mechanism worth pinning down so a future refactor of
  ``resolve_required_ops`` does not remove it by accident: resolution matches on the op BASENAME across the
  whole registry, and ``_apply_hub_namespacing`` namespaces an op's name and aliases but NOT its
  ``required_ops`` entries. A hub collection asking for ``get_answer_indices`` therefore reaches the
  bundled op.
- **Loud** was made true in Phase 2, which routed the pop through ``op_load_failure``.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError

# The dispatcher MODULE object, not a dotted string: `monkeypatch.setattr("a.b.c.NAME", ...)` walks the
# path with `getattr`, and `interpretune.analysis` is a lazy PEP 562 package that does not expose `ops` as an
# attribute until something imports it -- so the string form resolves only when an unrelated earlier import
# happened to set it, i.e. it passes in isolation and fails inside the suite.
from interpretune.analysis.ops import dispatcher as dispatcher_module
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

# `get_answer_indices` is the real cross-family dependency §3.6a names, defined only by the `core` family.
BUNDLED_DEPENDENCY = "get_answer_indices"

COLLECTION_YAML = f"""collection:
  name: dependent_collection
  version: 0.1.0

needs_a_bundled_op:
  description: a hub op depending on an op it does not define
  implementation: dep_impl.dep_fn
  required_ops: [{BUNDLED_DEPENDENCY}]
  input_schema: {{}}
  output_schema: {{}}
"""

UNRESOLVABLE_YAML = """collection:
  name: broken_collection
  version: 0.1.0

needs_a_nonexistent_op:
  description: a hub op depending on an op nothing defines
  implementation: dep_impl.dep_fn
  required_ops: [no_such_op_anywhere]
  input_schema: {}
  output_schema: {}
"""

DEP_IMPL = '''"""In-repo implementation module for the fixture collection."""


def dep_fn(module, analysis_batch, batch, batch_idx):
    """Stand in for a real op body."""
    return analysis_batch
'''


def _load(tmp_path, monkeypatch, ops_yaml: str) -> AnalysisOpDispatcher:
    from tests.hub_op_fixtures import write_cached_op_collection

    hub_cache = tmp_path / "hub"
    write_cached_op_collection(
        hub_cache, repo_id="depender/ops", op_files={"ops.yaml": ops_yaml, "dep_impl.py": DEP_IMPL}
    )
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr(dispatcher_module, "IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_OP_PATHS", [])
    dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
    (tmp_path / "cache").mkdir()
    dispatcher._cache_manager.cache_dir = tmp_path / "cache"
    return dispatcher


class TestCrossCollectionDeclarability:
    def test_a_hub_op_may_require_a_bundled_op_it_does_not_define(self, tmp_path, monkeypatch):
        dispatcher = _load(tmp_path, monkeypatch, COLLECTION_YAML)
        dispatcher.load_definitions()

        op_def = dispatcher._op_definitions["depender.ops.needs_a_bundled_op"]
        assert op_def.required_ops == [BUNDLED_DEPENDENCY], (
            "a cross-collection dependency must resolve to the bundled op, not to a namespaced name that does not exist"
        )

    def test_the_op_survives_the_load(self, tmp_path, monkeypatch):
        """The §3.6a symptom was the op being POPPED, so its presence is the thing to assert."""
        dispatcher = _load(tmp_path, monkeypatch, COLLECTION_YAML)
        dispatcher.load_definitions()
        assert "depender.ops.needs_a_bundled_op" in dispatcher._op_definitions

    def test_required_ops_entries_are_not_namespaced(self, tmp_path, monkeypatch):
        """The mechanism declarability rests on: namespacing rewrites the op name and its aliases, never its
        dependencies.

        Were `required_ops` namespaced wholesale, every entry would become `<user>.<repo>.<op>` and a
        collection could not reference anything outside itself. Exercised on a path INSIDE the hub cache, so
        namespacing actually runs -- on any other path it returns the content unchanged and the assertion
        would hold for the wrong reason.
        """
        dispatcher = _load(tmp_path, monkeypatch, COLLECTION_YAML)
        hub_yaml = tmp_path / "hub" / "models--depender--ops" / "snapshots" / "abc123" / "ops.yaml"
        assert hub_yaml.is_file(), "fixture layout changed"

        namespaced = dispatcher._apply_hub_namespacing(
            {"an_op": {"required_ops": [BUNDLED_DEPENDENCY], "aliases": ["an_alias"]}}, hub_yaml
        )

        assert list(namespaced) == ["depender.ops.an_op"], "the op NAME is namespaced"
        entry = namespaced["depender.ops.an_op"]
        assert entry["aliases"] == ["depender.ops.an_alias"], "aliases are namespaced too"
        assert entry["required_ops"] == [BUNDLED_DEPENDENCY], "dependencies are NOT namespaced"

    def test_schema_from_the_required_bundled_op_is_inherited(self, tmp_path, monkeypatch):
        """Resolution exists so the dependency's schema is compiled in, which is what makes it useful.

        Asserted as "the dependent op gained fields it did not declare" rather than as a specific set
        inclusion: the fixture declares `input_schema: {}`, so any field present came from the dependency,
        and that stays true regardless of which fields the compiler considers already satisfied by the
        dependency's own outputs.
        """
        dispatcher = _load(tmp_path, monkeypatch, COLLECTION_YAML)
        dispatcher.load_definitions()
        dependent = dispatcher._op_definitions["depender.ops.needs_a_bundled_op"]
        inherited = set(dependent.input_schema) | set(dependent.output_schema)
        assert inherited, (
            "the op declares empty schemas, so an empty compiled schema means the bundled dependency "
            "contributed nothing and resolution was cosmetic"
        )


class TestUnresolvableDependencyIsLoud:
    def test_an_unresolvable_dependency_warns_and_drops_only_that_op(self, tmp_path, monkeypatch):
        dispatcher = _load(tmp_path, monkeypatch, UNRESOLVABLE_YAML)
        with pytest.warns(UserWarning, match="no_such_op_anywhere"):
            dispatcher.load_definitions()
        assert "depender.ops.needs_a_nonexistent_op" not in dispatcher._op_definitions
        assert "model_fwd" in dispatcher._op_definitions, "one bad dependency must not take the session down"

    def test_strict_loading_turns_it_into_an_error(self, tmp_path, monkeypatch):
        """The 'loud failure mode' half of the exit criterion: silent op loss is what §3.6a objected to."""
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        dispatcher = _load(tmp_path, monkeypatch, UNRESOLVABLE_YAML)
        with pytest.raises(OpLoadError, match="no_such_op_anywhere"):
            dispatcher.load_definitions()
