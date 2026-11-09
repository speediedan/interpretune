"""The venngram support declaration (#887): derived from a backend's own records, refusing by name."""

from dataclasses import dataclass, field

import pytest

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends.capabilities import InterventionSupport
from interpretune.analysis.venngram import check_recipe, require_supported, support_mapping, venngram_support
from interpretune.utils import MisconfigurationException

BOOK = {
    "schema_version": 1,
    "name": "units",
    "portability": "checkpoint",
    "applies_to": {"models": [{"id": "m", "revision": "r1"}]},
    "sets": {"chess": {"point": "mlp.out.hook_in", "layers": [15], "units": {15: [7, 4041]}}},
}


@dataclass(frozen=True)
class _Capture:
    capturable: frozenset = field(default_factory=lambda: frozenset({"mlp.hook_out", "hook_resid_pre"}))


CONCEPT = {"values": [1.0, 0.0, 0.0, 0.0], "shape": [4], "basis": "jlens_folded"}


def _recipe(operands=None, **stage):
    from venngram.recipe import Recipe

    doc = {
        "schema_version": 2,
        "kind": "recipe",
        "name": "dose-the-units",
        "model": {"id": "m", "revision": "r1"},
        "book": BOOK,
        "stages": [stage],
    }
    if operands:
        doc["operands"] = operands
    return Recipe.from_mapping(doc)


def _support(capture=True):
    return venngram_support(InterventionSupport.every(), _Capture() if capture else None)


class TestDerivedSupport:
    def test_scopes_and_ops_come_from_the_backend_record(self):
        support = _support()
        # every core venngram op has a native mode now, `scale` included
        assert support.ops == frozenset({"add", "replace", "scale", "project", "reject", "patch", "clamp"})
        assert support.tokens == frozenset({"all", "last"})
        assert support.targets == frozenset({"units", "activation"}) and support.values == frozenset(
            {"dose", "doses", "from"}
        )
        assert support.sites == frozenset(
            {"mlp.hook_out", "hook_resid_pre", "blocks.*.mlp.hook_out", "blocks.*.hook_resid_pre"}
        )
        assert support.bases == frozenset({"jlens_folded", "jlens_unfolded"}) and support.name == "interpretune"

    def test_the_default_bases_are_interpretunes_jlens_bases(self):
        from interpretune.analysis.optools import JLENS_BASIS_NAMES
        from interpretune.analysis.venngram.support import DEFAULT_BASES

        assert DEFAULT_BASES == frozenset(JLENS_BASIS_NAMES.values())

    def test_ops_are_bounded_by_the_backend_modes(self):
        from interpretune.analysis.backends.capabilities import PositionScope

        support = venngram_support(InterventionSupport(position_scopes={PositionScope.LAST_TOKEN}, ops={"replace"}))
        assert support.ops == frozenset({"replace"}) and support.tokens == frozenset({"last"})

    def test_no_capture_record_declares_no_site_bound(self):
        assert _support(capture=False).sites is None


class TestCheckRecipe:
    def test_a_unit_dose_add_at_a_capturable_site_is_supported(self):
        recipe = _recipe(site="mlp.hook_out", ops=[{"op": "add", "target": {"set": "chess"}, "value": {"dose": 0.8}}])
        assert check_recipe(recipe, _support()) == []
        require_supported(recipe, _support())

    def test_the_mlp_hidden_input_site_is_refused_by_name(self):
        """The known gap (#887): recipes spell the MLP hidden activation's input `mlp.out.hook_in`, which no point
        names yet, so it is refused rather than guessed at."""
        recipe = _recipe(
            site="mlp.out.hook_in", ops=[{"op": "add", "target": {"set": "chess"}, "value": {"dose": 0.8}}]
        )
        refusals = check_recipe(recipe, _support())
        assert any("mlp.out.hook_in" in r and "not one this consumer reaches" in r for r in refusals)
        with pytest.raises(MisconfigurationException, match="asks for more than interpretune honours"):
            require_supported(recipe, _support())

    def test_operand_ops_at_a_layered_site_are_supported(self):
        recipe = _recipe(
            operands={"concept": CONCEPT},
            site="blocks.3.mlp.hook_out",
            ops=[
                {"op": "reject", "operand": "concept", "params": {"fraction": 0.5}},
                {"op": "scale", "value": {"dose": 2}},
            ],
        )
        assert check_recipe(recipe, _support()) == []

    def test_untranslatable_ops_refuse_by_name(self):
        unit_replace = _recipe(
            site="mlp.hook_out", ops=[{"op": "replace", "target": {"set": "chess"}, "value": {"dose": 2.0}}]
        )
        assert any(
            "a unit-targeted `replace` has no native translation" in r for r in check_recipe(unit_replace, _support())
        )
        layer_free = _recipe(
            operands={"concept": CONCEPT}, site="mlp.hook_out", ops=[{"op": "reject", "operand": "concept"}]
        )
        assert any("names no layer" in r for r in check_recipe(layer_free, _support()))
        other_basis = _recipe(
            operands={"concept": {**CONCEPT, "basis": "unfolded"}},
            site="blocks.3.mlp.hook_out",
            ops=[{"op": "reject", "operand": "concept"}],
        )
        assert any("basis `unfolded`, which is not supported" in r for r in check_recipe(other_basis, _support()))
        assert check_recipe(other_basis, venngram_support(InterventionSupport.every(), _Capture(), bases=None)) == []

    def test_the_published_record_round_trips_and_conformance_agrees(self, tmp_path):
        import yaml
        from venngram import conformance, store
        from venngram.support import SupportRecord

        support = _support()
        document = support_mapping(support)
        assert document["schema_version"] == 2 and set(document["sections"]) == {"steering"}
        record = SupportRecord.from_mapping(document)
        reread = record.steering
        assert record.name == support.name
        assert (reread.ops, reread.targets, reread.tokens, reread.values, reread.sites, reread.bases) == (
            support.ops,
            support.targets,
            support.tokens,
            support.values,
            support.sites,
            support.bases,
        )

        root = tmp_path / "published"
        root.mkdir()
        recipe = {
            "schema_version": 2,
            "kind": "recipe",
            "name": "dose-the-units",
            "model": {"id": "m", "revision": "r1"},
            "book": BOOK,
            "stages": [
                {"site": "mlp.hook_out", "ops": [{"op": "add", "target": {"set": "chess"}, "value": {"dose": 0.8}}]}
            ],
        }
        (root / "recipe.yaml").write_text(yaml.safe_dump(recipe))
        (root / "support.yaml").write_text(yaml.safe_dump(document))
        store.publish(
            "local", str(root), root, documents=[("recipe", "recipe.yaml"), ("support_record", "support.yaml")]
        )
        report = conformance.check(f"local:{root}", supports=[record])
        assert report.ok, [(d.path, d.error, dict(d.refusals)) for d in report.documents]
