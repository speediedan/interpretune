"""Resolving venngram site addresses into hook names, with the flow each acts on (#938)."""

import pytest

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.venngram import SiteSelection, resolve_sites
from interpretune.utils import MisconfigurationException

POINT = "hook_resid_post"


def test_a_gradient_address_selects_its_layers_and_says_grad():
    selection = resolve_sites({"point": POINT, "layers": "8:12", "of": "grad"}, n_layers=12)
    assert selection == SiteSelection(sites=tuple(f"blocks.{i}.{POINT}" for i in range(8, 12)), flow="grad")


def test_an_address_without_of_acts_on_values_and_counts_negative_layers_from_the_depth():
    selection = resolve_sites(f"point:{POINT} & layers:-2:", n_layers=6)
    assert selection.sites == (f"blocks.4.{POINT}", f"blocks.5.{POINT}") and selection.flow == "value"


def test_set_algebra_orders_sites_by_layer_then_point():
    """A union narrowed by a layer-only term, minus one site: the layer term filters the points the others name."""
    address = f"(point:{POINT} | point:attn.hook_out) & layers:[1,3] - (point:attn.hook_out & layers:3)"
    assert resolve_sites(address, n_layers=4).sites == (
        "blocks.1.attn.hook_out",
        f"blocks.1.{POINT}",
        f"blocks.3.{POINT}",
    )


def test_references_follow_the_books_sets():
    sets = {"top": {"point": POINT, "layers": "-1:"}, "top_grads": "@top & of:grad"}
    assert resolve_sites("@top_grads", n_layers=3, sets=sets) == SiteSelection(
        sites=(f"blocks.2.{POINT}",), flow="grad"
    )


@pytest.mark.parametrize("point", ["unembed.hook_in", "ln_final.hook_normalized", "embed.hook_out", "hook_embed"])
def test_a_global_point_resolves_once_to_its_layer_free_name(point):
    assert resolve_sites(f"point:{point}", n_layers=4) == SiteSelection(sites=(point,), flow="value")
    assert resolve_sites(f"point:{point} & of:grad", n_layers=4).flow == "grad"


def test_global_points_join_block_sites_and_follow_them():
    address = f"point:unembed.hook_in | (point:{POINT} & layers:-1:)"
    assert resolve_sites(address, n_layers=3).sites == (f"blocks.2.{POINT}", "unembed.hook_in")
    assert resolve_sites("@out", n_layers=3, sets={"out": "point:unembed.hook_in"}).sites == ("unembed.hook_in",)


def test_a_layer_term_never_selects_a_global_point():
    with pytest.raises(
        MisconfigurationException, match=r"\['unembed.hook_in'\] are global points, which have no layer"
    ):
        resolve_sites("point:unembed.hook_in & layers:1", n_layers=4)
    assert resolve_sites(f"(point:unembed.hook_in | point:{POINT}) & layers:1", n_layers=4).sites == (
        f"blocks.1.{POINT}",
    )


def test_no_global_component_shares_a_name_with_a_block_component_in_any_bundled_map():
    """The point alone decides whether it is global, so a resolver needs no model to say so."""
    from interpretune.analysis.points import component_map_for, known_architectures

    maps = [component_map_for(name) for name in known_architectures()]
    assert maps
    stacked = {c for m in maps for stack in m.stack_names() for c in m.block_components(stack)}
    assert not stacked & {c for m in maps for c in m.global_components()}


def test_an_intervention_address_at_a_global_point_expands_to_the_models_hook():
    from interpretune.analysis.backends.interventions import expand_intervention_patterns

    hooks = {name: name for name in ("blocks.0.hook_in", "blocks.1.hook_in", "unembed.hook_in")}
    address = "point:unembed.hook_in"
    assert expand_intervention_patterns([address], hooks) == {address: ["unembed.hook_in"]}
    with pytest.raises(ValueError, match="selects ln_final.hook_in, which this model does not expose"):
        expand_intervention_patterns(["point:ln_final.hook_in"], hooks)


@pytest.mark.parametrize(
    "address, sets, match",
    [
        ({"point": "resid_post", "layers": 1}, None, "`point:resid_post` is not an activation point"),
        ({"role": "adapter", "point": POINT}, None, r"\['role'\] have no site meaning here"),
        ({"point": POINT, "units": [3]}, None, r"\['units'\] have no site meaning here"),
        ({"point": POINT, "stack": "vision"}, None, "only the primary block stack is resolved"),
        ("layers:1:", None, "names no `point`"),
        ({"point": POINT, "layers": 9}, None, "Layer 9 is outside a stack of 4 blocks"),
        ("@missing", None, "no set is named `@missing`"),
        ("@a", {"a": "@b", "b": "@a"}, "refers to itself"),
        (f"point:{POINT} & of:grad | point:attn.hook_out & of:value", None, "cannot resolve the site address"),
        ({"point": POINT, "layers": [1], "expect": 2}, None, "selects 1 sites, expected 2"),
        (f"point:{POINT} & layers:1 - point:{POINT}", None, "selects no site of a 4-layer stack"),
    ],
)
def test_what_has_no_site_reading_is_refused_by_name(address, sets, match):
    with pytest.raises(MisconfigurationException, match=match):
        resolve_sites(address, n_layers=4, sets=sets)


class _Backend:
    def capture_support(self, model):
        return type("_Support", (), {"n_layers": 6})()


class _Module:
    """What ``resolve_gradient_sites`` reads of a module: its model backend and the model it is asked about."""

    def __init__(self):
        self._model_backend = _Backend()
        self.model = object()


class TestGradientSitesForTheOp:
    """``grad_capture`` takes hook names, which pass through, or one address, resolved over the model's depth."""

    def test_names_and_patterns_pass_through(self):
        from interpretune.analysis.optools import resolve_gradient_sites

        assert resolve_gradient_sites(_Module(), ["blocks.2.hook_out", "blocks.*.attn.hook_out"]) == [
            "blocks.2.hook_out",
            "blocks.*.attn.hook_out",
        ]
        assert resolve_gradient_sites(_Module(), None) is None

    def test_an_address_resolves_over_the_models_depth(self):
        from interpretune.analysis.optools import resolve_gradient_sites

        assert resolve_gradient_sites(_Module(), f"point:{POINT} & layers:-2: & of:grad") == [
            f"blocks.4.{POINT}",
            f"blocks.5.{POINT}",
        ]
        sets = {"late": {"point": POINT, "layers": "-1:"}}
        assert resolve_gradient_sites(_Module(), "@late & of:grad", sets=sets) == [f"blocks.5.{POINT}"]

    @pytest.mark.parametrize(
        "sites, match",
        [
            (f"point:{POINT} & layers:1", "acts on values, and these sites are read for their gradients"),
            (["blocks.1.hook_out", f"point:{POINT} & of:grad"], "ONE venngram address"),
        ],
    )
    def test_a_value_address_or_a_mixed_list_is_refused(self, sites, match):
        from interpretune.analysis.optools import resolve_gradient_sites

        with pytest.raises(MisconfigurationException, match=match):
            resolve_gradient_sites(_Module(), sites)
