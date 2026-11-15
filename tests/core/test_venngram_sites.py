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
