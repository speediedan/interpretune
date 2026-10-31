"""The venngram support declaration (#887): derived from a backend's own records, refusing by name."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends.capabilities import InterventionSupport
from interpretune.analysis.venngram import check_recipe, require_supported, venngram_support
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


def _recipe(**stage):
    from venngram.recipe import Recipe

    doc = {
        "schema_version": 2,
        "kind": "recipe",
        "name": "dose-the-units",
        "model": {"id": "m", "revision": "r1"},
        "book": BOOK,
        "stages": [stage],
    }
    return Recipe.from_mapping(doc)


def _support(capture=True):
    return venngram_support(InterventionSupport.every(), _Capture() if capture else None)


class TestDerivedSupport:
    def test_scopes_and_ops_come_from_the_backend_record(self):
        support = _support()
        assert support.ops == frozenset({"add"})  # only what the apply path implements so far
        assert support.tokens == frozenset({"all", "last"})
        assert support.targets == frozenset({"units"}) and support.values == frozenset({"dose", "doses"})
        assert support.sites == frozenset({"mlp.hook_out", "hook_resid_pre"}) and support.name == "interpretune"

    def test_a_backend_without_add_declares_no_ops(self):
        from interpretune.analysis.backends.capabilities import InterventionMode, PositionScope

        support = venngram_support(
            InterventionSupport(position_scopes={PositionScope.LAST_TOKEN}, modes={InterventionMode.REPLACE})
        )
        assert support.ops == frozenset() and support.tokens == frozenset({"last"})

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

    def test_unmapped_ops_and_operand_ops_refuse_by_name(self):
        recipe = _recipe(
            site="mlp.hook_out",
            ops=[{"op": "scale", "target": {"set": "chess"}, "value": {"dose": 2.0}}],
        )
        assert any("not supported" in r for r in check_recipe(recipe, _support()))
        from venngram.recipe import Op, Stage, Tokens

        with_operand = SimpleNamespace(
            stages=[Stage(site="mlp.hook_out", ops=(Op(op="add", operand="concept"),), tokens=Tokens("all"))],
            book=recipe.book,
        )
        assert any("operand `concept` is not supported yet" in r for r in check_recipe(with_operand, _support()))
