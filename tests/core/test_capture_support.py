"""The capture declaration: a support record for the one surface every model backend has.

Capture is not an optional capability, so it is not a ``BackendCapability`` member; what varies between backends is
WHICH vocabulary points they can capture on the model they wrap, and that is a typed record beside the intervention
and latent-model records. These tests pin the record's semantics and the inventory it is a fraction of, without a
session; the conformance suite is where a live backend's declaration is checked against what it captures.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.backends import CaptureSupport
from interpretune.analysis.points import component_map_for
from interpretune.analysis.points.inventory import inventory, is_global, named_inventory, spelled_at


class TestInventory:
    def test_gpt2_inventory_is_layer_free_bases_with_globals_once(self):
        bases = inventory(component_map_for("GPT2LMHeadModel"))
        assert "hook_in" in bases and "hook_out" in bases
        # a semantic name for the same tensor as a component spelling files under that spelling's key
        assert "hook_resid_pre" not in bases and "hook_attn_in" not in bases
        # a semantic CONTRIBUTION keeps its own key: on a sandwich-norm architecture it is not `mlp.hook_out`
        assert "hook_mlp_out" in bases and "hook_attn_out" in bases and "mlp.hook_out" in bases
        assert "ln2.hook_out" in bases and "ln2.hook_normalized" in bases and "attn.o.hook_in" in bases
        assert "unembed.hook_in" in bases and "embed.hook_out" in bases
        assert not any(b.startswith(".") for b in bases), "the block row itself must not contribute a bare '.hook_*'"
        assert bases == tuple(sorted(bases))

    def test_named_inventory_expands_block_points_over_layers_and_globals_once(self):
        cmap = component_map_for("GPT2LMHeadModel")
        names = named_inventory(cmap, 2)
        assert "blocks.0.ln2.hook_out" in names and "blocks.1.ln2.hook_out" in names
        assert names.count("unembed.hook_in") == 1
        assert is_global("unembed.hook_in", cmap) and not is_global("ln2.hook_out", cmap)
        assert spelled_at("ln2.hook_out", cmap, 5) == "blocks.5.ln2.hook_out"
        assert spelled_at("unembed.hook_in", cmap) == "unembed.hook_in"

    def test_a_cross_attention_point_is_not_in_a_decoder_inventory(self):
        bases = inventory(component_map_for("GPT2LMHeadModel"))
        assert "hook_cross_attn_out" not in bases, (
            "the inventory is what this architecture can host, not the whole vocabulary"
        )


def _record(**kw) -> CaptureSupport:
    """A declaration over gpt2's real inventory: everything capturable but one declared gap."""
    bases = set(inventory(component_map_for("GPT2LMHeadModel")))
    gap = "mlp.hook_out"
    base = dict(
        capturable=bases - {gap},
        uncapturable={gap: "the legacy grammar spells this hook_mlp_out, a different tensor on a sandwich-norm model"},
        n_layers=12,
        architecture="GPT2LMHeadModel",
    )
    base.update(kw)
    return CaptureSupport(**base)


class TestCaptureSupport:
    def test_capturable_points_are_accepted_at_any_layer_the_model_has(self):
        r = _record()
        assert r.can_capture("blocks.0.hook_in") and r.can_capture("blocks.11.ln2.hook_out")
        assert r.can_capture("unembed.hook_in")
        assert r.refusal("blocks.3.hook_in.hook_sae_acts_post") is None, "an SAE sub-hook is judged by its point"

    def test_a_declared_gap_is_refused_with_its_reason(self):
        why = _record().refusal("blocks.5.mlp.hook_out")
        assert why is not None and "blocks.5.mlp.hook_out" in why and "sandwich-norm" in why

    def test_a_layer_the_model_lacks_is_refused_as_such(self):
        why = _record().refusal("blocks.12.hook_in")
        assert why is not None and "layer 12" in why and "12 blocks" in why

    def test_a_declaration_must_decide_every_inventory_point(self):
        """The understating direction: a record that omits points would shrink its own denominator, so omission
        read as completeness ("captures 20 of 20") until this refusal existed."""
        bases = set(inventory(component_map_for("GPT2LMHeadModel")))
        with pytest.raises(ValueError, match=r"decides 29 of 30 inventory points; undecided .*'unembed.hook_out'"):
            CaptureSupport(
                capturable=bases - {"unembed.hook_out"}, uncapturable={}, n_layers=12, architecture="GPT2LMHeadModel"
            )

    def test_a_declaration_cannot_name_points_outside_the_inventory(self):
        bases = set(inventory(component_map_for("GPT2LMHeadModel")))
        with pytest.raises(ValueError, match="outside the architecture's inventory: \\['vision.hook_in'\\]"):
            CaptureSupport(
                capturable=bases | {"vision.hook_in"}, uncapturable={}, n_layers=12, architecture="GPT2LMHeadModel"
            )

    def test_a_spelling_outside_the_vocabulary_is_refused_as_unknown(self):
        assert _record().refusal("blocks.5.attn.hook_pattern_weird") is not None

    def test_a_point_the_architecture_does_not_define_is_refused_as_the_architectures_fact(self):
        """A cross-attention point on a decoder parses but resolves to no tensor: the record must not present that as
        a gap in the backend's declaration, which decides every inventory point and so has no gaps."""
        why = _record().refusal("blocks.5.hook_cross_attn_out")
        assert why is not None and "not defined on GPT2LMHeadModel" in why and "resolver's refusal" in why
        assert "neither declared" not in why

    def test_a_point_cannot_be_both(self):
        with pytest.raises(ValueError, match="both capturable and uncapturable"):
            _record(uncapturable={"hook_in": "x", "mlp.hook_out": "y"})

    def test_an_empty_declaration_is_refused(self):
        with pytest.raises(ValueError, match="at least one capturable point"):
            _record(capturable=set())

    def test_describe_carries_the_fraction_and_the_gaps_by_name(self):
        text = _record().describe()
        assert "captures 29 of 30 base points on GPT2LMHeadModel (12 blocks)" in text and "mlp.hook_out" in text
