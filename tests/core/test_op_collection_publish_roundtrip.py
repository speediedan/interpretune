"""Publish -> pull -> op-parity round trip on the concept family (#266 Phase 3, exit criteria).

Two claims about the bundled-op refactor were previously evidenced only indirectly, and this is where both
become demonstrated rather than argued:

- **Hub provenance.** ``OpDef.source == "hub:<user.repo>"`` was asserted only by a unit test that
  monkeypatches ``get_hub_namespace``, so it proved the plumbing agreed with itself.
- **Hub declarability.** That a hub op can declare what a bundled op declares -- ``op_state`` and the
  behavioral traits -- rested on shared-code-path reasoning. ``concept_direction`` is the op that declares
  ``op_state``, so the round trip covers it for free.

The trip runs entirely offline through the local-publish path: the built tree is installed into a temporary
ops cache in HF layout, which is exactly what a real ``pull_ops`` leaves behind. That keeps this in the hosted
matrix (no network, no token, no trust consent beyond the suite's own) while still exercising the real builder,
the real manifest routing and the real dispatcher.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

# The dispatcher MODULE object, not a dotted string: `monkeypatch.setattr("a.b.c.NAME", ...)` walks the
# path with `getattr`, and `interpretune.analysis` is a lazy PEP 562 package that does not expose `ops` as an
# attribute until something imports it -- so the string form resolves only when an unrelated earlier import
# happened to set it, i.e. it passes in isolation and fails inside the suite.
from interpretune.analysis.ops import dispatcher as dispatcher_module
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher
from interpretune.hub.manifest import IT_COMPONENT_MANIFEST
from interpretune.hub.opcollections import resolve_cached_op_files
from interpretune.hub.publish import build_op_collection_tree

CONCEPT_FAMILY = (
    Path(__file__).parent.parent.parent / "src" / "interpretune" / "analysis" / "ops" / "bundled" / "concept"
)
REPO_ID = "speediedan/concept_direction_ops"
NAMESPACE = "speediedan.concept_direction_ops"
REVISION = "f" * 40


@pytest.fixture(scope="module")
def published_tree(tmp_path_factory) -> Path:
    """The publishable ops repo, built from the in-tree concept family exactly as a publish would."""
    out = tmp_path_factory.mktemp("published_collection") / "build"
    build_op_collection_tree(CONCEPT_FAMILY, out, REPO_ID)
    return out


@pytest.fixture
def pulled_dispatcher(published_tree, tmp_path, monkeypatch) -> AnalysisOpDispatcher:
    """A dispatcher whose ops cache holds the published collection, installed in HF cache layout."""
    hub_cache = tmp_path / "hub"
    snapshot = hub_cache / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION
    snapshot.parent.mkdir(parents=True)
    shutil.copytree(published_tree, snapshot)
    refs = snapshot.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text(REVISION, encoding="utf-8")

    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr(dispatcher_module, "IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_OP_PATHS", [])
    monkeypatch.delenv("IT_OP_PRECEDENCE", raising=False)

    dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
    (tmp_path / "cache").mkdir()
    dispatcher._cache_manager.cache_dir = tmp_path / "cache"
    dispatcher.load_definitions()
    return dispatcher


class TestPublishedTree:
    def test_the_tree_is_a_well_formed_ops_repo(self, published_tree):
        """Well-formed in the sense the loader means: manifest-routed resolution finds its op files."""
        resolved = resolve_cached_op_files(published_tree, source=REPO_ID)
        assert [p.name for p in resolved] == ["concept_ops.yaml"]

    def test_an_op_collection_load_fetches_the_manifest_first(self, published_tree):
        """The ``countDownloads: path:"it_component.yaml"`` contract claimed by registration PR A.

        Asserted structurally: resolution reads the manifest to learn what the op files ARE, so no path
        exists that reaches an op YAML without reading the manifest first.
        """
        manifest = yaml.safe_load((published_tree / IT_COMPONENT_MANIFEST).read_text(encoding="utf-8"))
        assert manifest["kinds"] == ["ops"] and manifest["ops"]["files"] == ["concept_ops.yaml"]
        (published_tree / IT_COMPONENT_MANIFEST).rename(published_tree / "_moved")
        try:
            with pytest.raises(Exception, match=f"no {IT_COMPONENT_MANIFEST}"):
                resolve_cached_op_files(published_tree, source=REPO_ID)
        finally:
            (published_tree / "_moved").rename(published_tree / IT_COMPONENT_MANIFEST)

    def test_implementations_are_rewritten_to_the_grammar_the_hub_loader_parses(self, published_tree):
        """``get_function_from_dynamic_module`` parses ``<module>.<function>``, not an installed package path."""
        content = yaml.safe_load((published_tree / "concept_ops.yaml").read_text(encoding="utf-8"))
        impls = [op["implementation"] for name, op in content.items() if name != "collection"]
        assert impls, "no ops in the published collection"
        for impl in impls:
            assert impl.startswith("concept_ops."), impl
            assert impl.count(".") == 1, f"the dynamic-module path parses exactly one dot: {impl}"

    def test_the_referenced_module_ships_with_the_collection(self, published_tree):
        assert (published_tree / "concept_ops.py").is_file()

    def test_the_published_yaml_is_marked_generated(self, published_tree):
        """An editable-looking copy of a generated file is how single-sourcing quietly dies."""
        head = (published_tree / "concept_ops.yaml").read_text(encoding="utf-8")[:400]
        assert "GENERATED" in head and "do not edit" in head
        assert "bundled/concept/concept_ops.yaml" in head, "the banner must name the source of truth"

    def test_the_collection_handle_distinguishes_itself_from_the_bundled_family(self, published_tree):
        content = yaml.safe_load((published_tree / "concept_ops.yaml").read_text(encoding="utf-8"))
        assert content["collection"]["name"] == "concept_direction_ops"

    def test_the_card_carries_the_discovery_sentinel_and_an_ops_section(self, published_tree):
        from interpretune.hub.cards import generate_component_card
        from interpretune.hub.manifest import load_component_manifest

        card = generate_component_card(load_component_manifest(published_tree / IT_COMPONENT_MANIFEST), REPO_ID)
        rendered = str(card)
        assert "library_name: interpretune" in rendered
        assert "interpretune-ops" in rendered, "the tag list_models(filter=...) discovery queries"
        assert "## Operations" in rendered and "pull_ops" in rendered


class TestPulledOpParity:
    def test_every_bundled_concept_op_arrives_namespaced(self, pulled_dispatcher):
        bundled_names = [
            name
            for name in yaml.safe_load((CONCEPT_FAMILY / "concept_ops.yaml").read_text(encoding="utf-8"))
            if name != "collection"
        ]
        for name in bundled_names:
            assert f"{NAMESPACE}.{name}" in pulled_dispatcher._op_definitions, name

    def test_hub_provenance_is_recorded_on_the_pulled_ops(self, pulled_dispatcher):
        """The claim that was previously evidenced only by monkeypatching ``get_hub_namespace``."""
        op_def = pulled_dispatcher._op_definitions[f"{NAMESPACE}.concept_direction"]
        assert op_def.source == f"hub:{NAMESPACE}"

    def test_declared_collection_identity_survives_the_trip(self, pulled_dispatcher):
        op_def = pulled_dispatcher._op_definitions[f"{NAMESPACE}.concept_direction"]
        assert (op_def.collection_name, op_def.collection_version) == ("concept_direction_ops", "0.1.0")

    def test_op_state_survives_the_trip(self, pulled_dispatcher):
        """``concept_direction`` is the op that declares cross-batch state, so this is not vacuous."""
        bundled = pulled_dispatcher._op_definitions["concept_direction"]
        pulled = pulled_dispatcher._op_definitions[f"{NAMESPACE}.concept_direction"]
        assert bundled.op_state is not None, "fixture assumption: the bundled op declares op_state"
        assert pulled.op_state == bundled.op_state

    @pytest.mark.parametrize("trait", ["uses_default_hooks", "requires_grad", "per_latent_preds"])
    def test_behavioral_traits_survive_the_trip(self, pulled_dispatcher, trait):
        for name in ("concept_direction", "model_fwd_intervention"):
            bundled = pulled_dispatcher._op_definitions[name]
            pulled = pulled_dispatcher._op_definitions[f"{NAMESPACE}.{name}"]
            assert getattr(pulled, trait) == getattr(bundled, trait), name

    def test_schemas_survive_the_trip(self, pulled_dispatcher):
        bundled = pulled_dispatcher._op_definitions["concept_direction"]
        pulled = pulled_dispatcher._op_definitions[f"{NAMESPACE}.concept_direction"]
        assert pulled.input_schema == bundled.input_schema
        assert pulled.output_schema == bundled.output_schema

    def test_bundled_ops_still_win_bare_names_after_the_pull(self, pulled_dispatcher):
        """Pulling a collection must not change what existing code resolves to."""
        assert pulled_dispatcher._op_definitions["concept_direction"].source == "bundled"
        assert pulled_dispatcher._resolve_name_safe("concept_direction") == "concept_direction"

    def test_mirroring_a_bundled_family_loads_without_warnings(self, published_tree, tmp_path, monkeypatch):
        """A mirror collides on EVERY name by construction, and those collisions are the documented default.

        This emitted nine warnings before the contest reporter distinguished by-design collisions (bundled incumbent,
        hub challenger) from genuinely ambiguous ones. Warning on working-as-designed behavior for the primary demo path
        is how a warning channel stops being read, so the quiet is asserted.
        """
        import warnings

        hub_cache = tmp_path / "hub"
        snapshot = hub_cache / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION
        snapshot.parent.mkdir(parents=True)
        shutil.copytree(published_tree, snapshot)
        monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache)
        monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_OP_PATHS", [])

        dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
        (tmp_path / "cache").mkdir()
        dispatcher._cache_manager.cache_dir = tmp_path / "cache"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dispatcher.load_definitions()
        contests = [
            w for w in caught if "already has an assigned op" in str(w.message) or "already exists" in str(w.message)
        ]
        assert not contests, [str(w.message) for w in contests]

    def test_a_contest_between_two_non_bundled_collections_still_warns(self, pulled_dispatcher):
        """The volume reduction is scoped: ambiguity that a user has to resolve must stay loud."""
        from interpretune.analysis.ops.compiler.cache_manager import OpDef
        from interpretune.analysis.ops.base import OpSchema

        def _fixture(source: str) -> OpDef:
            return OpDef(
                name="x",
                description="",
                implementation="m.f",
                input_schema=OpSchema(),
                output_schema=OpSchema(),
                source=source,
            )

        with pytest.warns(UserWarning, match="contested"):
            pulled_dispatcher._report_bare_name_contest(_fixture("hub:a.b"), _fixture("hub:c.d"), "contested name")
        with pytest.warns(UserWarning, match="contested"):
            pulled_dispatcher._report_bare_name_contest(_fixture("local"), _fixture("hub:c.d"), "contested name")

    def test_op_info_reports_both_copies_and_the_flip(self, pulled_dispatcher):
        info = pulled_dispatcher.op_info("concept_direction")
        assert info.active.source == "bundled" and info.active.collection == "concept"
        assert [c.name for c in info.alternatives] == [f"{NAMESPACE}.concept_direction"]
        assert info.alternatives[0].revision == REVISION
        assert not info.is_shadowing_bundled

        pulled_dispatcher.prefer_ops(REPO_ID)
        flipped = pulled_dispatcher.op_info("concept_direction")
        assert flipped.resolved == f"{NAMESPACE}.concept_direction"
        assert flipped.active.collection == "concept_direction_ops"
        assert flipped.is_shadowing_bundled
