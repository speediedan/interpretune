"""Bare-name precedence opt-in and ``op_info`` introspection (#266 Phase 3, D8 item 2).

Bundled ops win bare names by default, which is what makes a session work offline and identically for everyone. Opting
into a hub collection's copy of an op is explicit, per-namespace, reversible within a session, and never a side effect
of pulling one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The dispatcher MODULE object, not a dotted string: `monkeypatch.setattr("a.b.c.NAME", ...)` walks the
# path with `getattr`, and `interpretune.analysis` is a lazy PEP 562 package that does not expose `ops` as an
# attribute until something imports it -- so the string form resolves only when an unrelated earlier import
# happened to set it, i.e. it passes in isolation and fails inside the suite.
from interpretune.analysis.ops import dispatcher as dispatcher_module
from interpretune.analysis.ops.dispatcher import (
    IT_OP_PRECEDENCE_ENV_VAR,
    AnalysisOpDispatcher,
    OpCandidate,
    OpResolution,
)

_IMPL = "interpretune.analysis.ops.bundled.core.core_ops.model_fwd_impl"

# `model_fwd` deliberately collides with a BUNDLED op name: shadowing a bundled op is the case the opt-in
# exists for, and the case where getting resolution wrong is worst. Implementations are referenced the way a
# real hub collection references them -- `<in-repo module>.<function>`, resolved through the dynamic-module
# path -- not as installed package paths, which that path cannot parse.
HUB_OPS = """collection:
  name: rival_collection
  version: 9.9.9

model_fwd:
  description: a hub op claiming a bundled name
  implementation: rival_impl.rival_model_fwd
  input_schema: {}
  output_schema: {}

hub_only_op:
  description: a hub op claiming no bundled name
  implementation: rival_impl.rival_model_fwd
  input_schema: {}
  output_schema: {}
"""

HUB_IMPL = '''"""In-repo implementation module for the fixture collection."""


def rival_model_fwd(module, analysis_batch, batch, batch_idx):
    """Stand in for a real op body; identity is all the resolution tests need."""
    return analysis_batch
'''


@pytest.fixture
def hub_dispatcher(tmp_path, monkeypatch) -> AnalysisOpDispatcher:
    """A loaded dispatcher whose hub cache holds one collection shadowing a bundled op name."""
    from tests.hub_op_fixtures import write_cached_op_collection

    hub_cache = tmp_path / "hub"
    write_cached_op_collection(
        hub_cache, repo_id="rival/ops", op_files={"ops.yaml": HUB_OPS, "rival_impl.py": HUB_IMPL}
    )
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr(dispatcher_module, "IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_OP_PATHS", [])
    monkeypatch.delenv(IT_OP_PRECEDENCE_ENV_VAR, raising=False)

    dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
    dispatcher._cache_manager.cache_dir = tmp_path / "cache"
    (tmp_path / "cache").mkdir()
    dispatcher.load_definitions()
    assert "rival.ops.model_fwd" in dispatcher._op_definitions, "fixture collection failed to load"
    return dispatcher


class TestDefaultPrecedence:
    def test_bundled_wins_a_contested_bare_name_by_default(self, hub_dispatcher):
        assert hub_dispatcher._resolve_name_safe("model_fwd") == "model_fwd"
        assert hub_dispatcher._op_definitions["model_fwd"].source == "bundled"

    def test_precedence_is_empty_without_an_opt_in(self, hub_dispatcher):
        assert hub_dispatcher.op_precedence == []

    def test_an_uncontested_hub_name_needs_no_opt_in(self, hub_dispatcher):
        """Precedence resolves collisions; it is not what makes hub ops reachable."""
        assert hub_dispatcher._op_definitions["hub_only_op"].name == "rival.ops.hub_only_op"


class TestPreferOps:
    def test_opting_in_flips_the_contested_bare_name(self, hub_dispatcher):
        assert hub_dispatcher.prefer_ops("rival/ops") == ["rival.ops"]
        assert hub_dispatcher._resolve_name_safe("model_fwd") == "rival.ops.model_fwd"

    def test_the_bundled_definition_stays_addressable(self, hub_dispatcher):
        """Re-ranking happens at lookup, so preferring a collection cannot make the bundled copy unreachable.

        A bundled op's only name IS its bare name, so a design that rebound ``_op_definitions`` keys would
        evict it rather than deprioritize it.
        """
        hub_dispatcher.prefer_ops("rival/ops")
        assert hub_dispatcher._op_definitions["model_fwd"].source == "bundled"

    def test_fully_qualified_names_are_never_re_ranked(self, hub_dispatcher):
        hub_dispatcher.prefer_ops("rival/ops")
        assert hub_dispatcher._resolve_name_safe("rival.ops.model_fwd") == "rival.ops.model_fwd"

    def test_opting_in_is_reversible_within_a_session(self, hub_dispatcher):
        hub_dispatcher.prefer_ops("rival/ops")
        assert hub_dispatcher.prefer_ops() == []
        assert hub_dispatcher._resolve_name_safe("model_fwd") == "model_fwd"

    def test_repo_id_and_namespace_forms_are_equivalent(self, hub_dispatcher):
        """``user/repo`` and ``user.repo`` both work: the same normalization storage uses."""
        assert hub_dispatcher.prefer_ops("rival/ops") == hub_dispatcher.prefer_ops("rival.ops", replace=True)

    def test_later_declarations_take_priority(self, hub_dispatcher):
        hub_dispatcher.prefer_ops("first/one")
        assert hub_dispatcher.prefer_ops("rival/ops") == ["first.one", "rival.ops"]
        # re-declaring moves a namespace to the front of the queue rather than duplicating it
        assert hub_dispatcher.prefer_ops("first/one") == ["rival.ops", "first.one"]

    def test_replace_discards_prior_declarations(self, hub_dispatcher):
        hub_dispatcher.prefer_ops("first/one", "second/two")
        assert hub_dispatcher.prefer_ops("rival/ops", replace=True) == ["rival.ops"]

    def test_preferring_an_uncached_namespace_is_inert(self, hub_dispatcher):
        """A namespace with nothing behind it must not shadow or break anything."""
        hub_dispatcher.prefer_ops("nobody/nothing")
        assert hub_dispatcher._resolve_name_safe("model_fwd") == "model_fwd"

    def test_flipping_precedence_clears_bound_dispatch_entries(self, hub_dispatcher):
        """An op instantiated before the flip must not keep being served from the dispatch table.

        Asserted on the table and on resolution rather than by instantiating the hub op afterwards: the hub
        impl import goes through the dynamic-module path, whose network fetch this fixture's cache patching
        does not cover. The end-to-end flip is exercised by the opt-in demo notebook.
        """
        assert hub_dispatcher.get_op("model_fwd").name == "model_fwd"
        assert hub_dispatcher._dispatch_table, "expected the instantiated op to be bound"
        hub_dispatcher.prefer_ops("rival/ops")
        assert not hub_dispatcher._dispatch_table
        assert hub_dispatcher._resolve_name_safe("model_fwd") == "rival.ops.model_fwd"


class TestEnvVarParity:
    def test_env_var_declares_precedence_for_scripted_runs(self, hub_dispatcher, monkeypatch):
        monkeypatch.setenv(IT_OP_PRECEDENCE_ENV_VAR, "rival/ops")
        assert hub_dispatcher.op_precedence == ["rival.ops"]
        assert hub_dispatcher._resolve_name_safe("model_fwd") == "rival.ops.model_fwd"

    def test_env_var_is_ordered_and_comma_separated(self, hub_dispatcher, monkeypatch):
        monkeypatch.setenv(IT_OP_PRECEDENCE_ENV_VAR, "a/one, b/two ,c/three")
        assert hub_dispatcher.op_precedence == ["a.one", "b.two", "c.three"]

    def test_empty_and_whitespace_entries_are_dropped(self, hub_dispatcher, monkeypatch):
        monkeypatch.setenv(IT_OP_PRECEDENCE_ENV_VAR, " , rival/ops ,, ")
        assert hub_dispatcher.op_precedence == ["rival.ops"]

    def test_explicit_declarations_outrank_the_env_var(self, hub_dispatcher, monkeypatch):
        monkeypatch.setenv(IT_OP_PRECEDENCE_ENV_VAR, "from/env")
        hub_dispatcher.prefer_ops("rival/ops")
        assert hub_dispatcher.op_precedence == ["rival.ops", "from.env"]

    def test_env_var_is_read_on_every_access(self, hub_dispatcher, monkeypatch):
        """The dispatcher is a module-level singleton built at import; a later export must still be seen."""
        assert hub_dispatcher.op_precedence == []
        monkeypatch.setenv(IT_OP_PRECEDENCE_ENV_VAR, "rival/ops")
        assert hub_dispatcher.op_precedence == ["rival.ops"]


class TestOpInfo:
    def test_reports_the_active_definition_and_its_provenance(self, hub_dispatcher):
        info = hub_dispatcher.op_info("model_fwd")
        assert isinstance(info, OpResolution)
        assert info.resolved == "model_fwd"
        assert info.active.source == "bundled"
        assert info.active.collection == "core"

    def test_reports_the_alternatives(self, hub_dispatcher):
        info = hub_dispatcher.op_info("model_fwd")
        rivals = [c for c in info.alternatives if c.name == "rival.ops.model_fwd"]
        assert rivals and rivals[0].source == "hub:rival.ops"
        assert rivals[0].collection == "rival_collection" and rivals[0].version == "9.9.9"

    def test_reports_collection_identity_of_the_winner_after_a_flip(self, hub_dispatcher):
        hub_dispatcher.prefer_ops("rival/ops")
        info = hub_dispatcher.op_info("model_fwd")
        assert (info.resolved, info.active.version) == ("rival.ops.model_fwd", "9.9.9")
        assert info.precedence == ("rival.ops",)

    def test_flags_a_hub_op_shadowing_a_bundled_one(self, hub_dispatcher):
        assert not hub_dispatcher.op_info("model_fwd").is_shadowing_bundled
        hub_dispatcher.prefer_ops("rival/ops")
        assert hub_dispatcher.op_info("model_fwd").is_shadowing_bundled

    def test_an_op_is_not_an_alternative_to_itself(self, hub_dispatcher):
        """Bare-name aliasing registers a second key for the SAME OpDef; that is not a second candidate."""
        info = hub_dispatcher.op_info("hub_only_op")
        assert info.resolved == "rival.ops.hub_only_op"
        assert info.alternatives == ()

    def test_resolves_through_aliases(self, hub_dispatcher):
        """``model_forward`` is a declared alias of the bundled ``model_fwd``."""
        assert hub_dispatcher.op_info("model_forward").resolved == "model_fwd"

    def test_accepts_the_names_normalization_accepts(self, hub_dispatcher):
        assert hub_dispatcher.op_info("Model-Fwd").resolved == "model_fwd"

    def test_an_unregistered_name_raises_rather_than_reporting_nothing(self, hub_dispatcher):
        """A typo must not come back as a resolution: an empty answer reads as a finding."""
        with pytest.raises(ValueError, match="Unknown operation: no_such_op"):
            hub_dispatcher.op_info("no_such_op")

    def test_str_is_readable_and_names_the_precedence(self, hub_dispatcher):
        hub_dispatcher.prefer_ops("rival/ops")
        rendered = str(hub_dispatcher.op_info("model_fwd"))
        assert "rival.ops.model_fwd" in rendered
        assert "collection rival_collection 9.9.9" in rendered
        assert "also available" in rendered and "bundled" in rendered
        assert "precedence: ['rival.ops']" in rendered

    def test_str_names_the_default_when_there_is_no_opt_in(self, hub_dispatcher):
        assert "bundled ops win bare names" in str(hub_dispatcher.op_info("model_fwd"))


class TestCachedRevisionReporting:
    def test_bundled_ops_report_no_revision(self, hub_dispatcher):
        assert hub_dispatcher.op_info("model_fwd").active.revision is None

    def test_hub_ops_report_the_cached_revision_without_network_access(self, tmp_path, monkeypatch):
        from interpretune.analysis.ops.dispatcher import _cached_op_revision

        cache = tmp_path / "hub"
        refs = cache / "models--rival--ops" / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text("c0ffee" * 6 + "abcd")
        monkeypatch.setattr(dispatcher_module, "IT_ANALYSIS_HUB_CACHE", cache)
        assert _cached_op_revision("hub:rival.ops") == "c0ffee" * 6 + "abcd"

    def test_an_uncached_hub_source_reports_none_rather_than_raising(self, tmp_path, monkeypatch):
        from interpretune.analysis.ops.dispatcher import _cached_op_revision

        monkeypatch.setattr(dispatcher_module, "IT_ANALYSIS_HUB_CACHE", tmp_path / "absent")
        assert _cached_op_revision("hub:rival.ops") is None


class TestHubVerbSurface:
    def test_prefer_ops_and_op_info_are_exposed_on_it_hub(self):
        import interpretune as it

        assert callable(it.hub.prefer_ops) and callable(it.hub.op_info)

    def test_candidate_str_degrades_gracefully_without_collection_identity(self):
        rendered = str(OpCandidate(name="u.r.op", source="hub:u.r", collection=None, version=None, revision=None))
        assert rendered == "u.r.op [hub:u.r]"

    def test_candidate_str_marks_an_unversioned_collection(self):
        candidate = OpCandidate(name="op", source="local", collection="mine", version=None, revision=None)
        assert "collection mine (unversioned)" in str(candidate)


def test_precedence_does_not_leak_between_dispatchers(tmp_path, monkeypatch):
    """Precedence is per-dispatcher state, so a test or notebook flip cannot escape its own dispatcher."""
    monkeypatch.delenv(IT_OP_PRECEDENCE_ENV_VAR, raising=False)
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_OP_PATHS", [])
    first = AnalysisOpDispatcher(yaml_paths=[Path(tmp_path)], enable_hub_ops=False)
    second = AnalysisOpDispatcher(yaml_paths=[Path(tmp_path)], enable_hub_ops=False)
    first.prefer_ops("rival/ops")
    assert second.op_precedence == []
