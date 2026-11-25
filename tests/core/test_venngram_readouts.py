"""A treatment's declared readout executed as the scalar a backward pass differentiates (#1024)."""

import copy

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends.capabilities import InterventionSupport, PositionScope
from interpretune.analysis.venngram import (
    evaluate_readout_case,
    executed_readouts,
    load_recipe,
    recipe_readout,
    register_readout_executor,
    venngram_support,
)
from interpretune.analysis.venngram.readouts import READOUT_EXECUTORS
from interpretune.utils import MisconfigurationException

A, B, READ, WIDTH, VOCAB = 7, 11, "blocks.5.hook_in", 6, 20
DIRECTION = [0.5, -1.0, 0.0, 2.0, 0.25, 1.0]


def _doc(readout=None, *, stages=None, operands=None, sets=None, schema=4):
    doc = {
        "schema_version": schema,
        "kind": "recipe",
        "name": "declared-readout",
        "model": {"id": "m", "revision": "r"},
        "book": {
            "schema_version": 3,
            "name": "readout-sets",
            "portability": "checkpoint",
            "applies_to": {"models": [{"id": "m", "revision": "r"}]},
            "sets": {
                "answer": {"point": "unembed.hook_out", "units": [A]},
                "foil": {"point": "unembed.hook_out", "units": [B]},
                "read": {"point": "hook_in", "layers": [5]},
                "read_two": {"point": "hook_in", "layers": [2, 5], "units": [1, 3]},
                "mixed": {"any": [{"point": "unembed.hook_out"}, {"point": "hook_in", "layers": [5]}]},
                "outside": "point:ln_final.hook_normalized",
                "edit_grad": {"point": "hook_out", "layers": [3], "units": [0], "of": "grad"},
                "other_grad": {"point": "hook_out", "layers": [4], "units": [0], "of": "grad"},
                **(sets or {}),
            },
        },
        "operands": operands
        if operands is not None
        else {"direction": {"values": DIRECTION, "shape": [WIDTH], "basis": "jlens_folded"}},
        "stages": stages
        if stages is not None
        else [
            {
                "site": "hook_out",
                "tokens": "last",
                "ops": [{"op": "scale", "target": {"set": "edit_grad"}, "value": {"dose": 0.0}}],
            }
        ],
    }
    if readout is not None:
        doc["readout"] = readout
    return doc


LOGIT_DIFFERENCE = {
    "terms": [
        {"reduce": "sum", "set": "answer", "tokens": "last", "weight": 1.0},
        {"reduce": "sum", "set": "foil", "tokens": "last", "weight": -1.0},
    ]
}


def _readout(readout, **kw):
    return recipe_readout(load_recipe(_doc(readout, **kw)).recipe, n_layers=8)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 4, VOCAB, generator=g)
    acts = {f"blocks.{layer}.hook_in": torch.randn(2, 4, WIDTH, generator=g) for layer in range(8)}
    return logits, acts


class TestConformance:
    def test_the_executors_compute_venngrams_conformance_vectors(self):
        from venngram.readouts import CORE_READOUTS, check_readouts

        assert check_readouts(evaluate_readout_case) == []
        assert executed_readouts() >= CORE_READOUTS

    def test_a_planted_wrong_executor_fails_the_vectors_and_is_not_installed(self):
        before = dict(READOUT_EXECUTORS)
        with pytest.raises(MisconfigurationException, match="fails venngram's conformance vectors: .*norm"):
            register_readout_executor("norm", lambda x, o, p: x.abs().sum(dim=(-2, -1)), replace=True)
        assert READOUT_EXECUTORS == before

    def test_an_executor_for_a_reduction_venngram_has_not_registered_is_refused(self):
        with pytest.raises(MisconfigurationException, match="`max_abs` is not registered in venngram"):
            register_readout_executor("max_abs", lambda x, o, p: x.abs().amax(dim=(-2, -1)))
        with pytest.raises(MisconfigurationException, match="already has an executor"):
            register_readout_executor("sum", READOUT_EXECUTORS["sum"])

    def test_a_registered_reduction_executes_once_its_executor_passes(self):
        from venngram import readouts as vg

        vg.register_readout(
            "max_abs",
            meaning="sum_t max_u |x_t[u]|: the largest magnitude at each token, summed over the tokens",
            cases=[
                {
                    "name": "max_abs_two_tokens",
                    "terms": [{"reduce": "max_abs", "values": [[1, -3], [2, 0]]}],
                    "expect": 5.0,
                }
            ],
        )
        try:
            register_readout_executor("max_abs", lambda x, o, p: x.abs().amax(dim=-1).sum(dim=-1))
            assert "max_abs" in executed_readouts()
            readout = _readout({"terms": [{"reduce": "max_abs", "set": "read", "tokens": "all"}]})
            logits, acts = _inputs()
            torch.testing.assert_close(readout(logits, acts), acts[READ].abs().amax(dim=-1).sum())
        finally:
            READOUT_EXECUTORS.pop("max_abs", None)
            vg.READOUTS.pop("max_abs", None)
            vg._REGISTERED_CASES.pop("max_abs", None)


class TestDeclaredReadout:
    def test_a_logit_difference_reads_the_logits_at_the_last_token(self):
        readout = _readout(LOGIT_DIFFERENCE)
        assert readout.sites == ()
        logits, acts = _inputs()
        torch.testing.assert_close(readout(logits, acts), (logits[:, -1, A] - logits[:, -1, B]).sum())

    def test_a_projection_reads_its_block_site_with_the_operand_in_unit_order(self):
        readout = _readout(
            {"terms": [{"reduce": "project", "set": "read", "tokens": "last", "operand": "direction", "weight": 0.5}]}
        )
        assert readout.sites == (READ,)
        logits, acts = _inputs()
        expected = 0.5 * (acts[READ][:, -1, :] @ torch.tensor(DIRECTION)).sum()
        torch.testing.assert_close(readout(logits, acts), expected)

    def test_terms_sum_with_their_weights_and_scopes(self):
        readout = _readout(
            {
                "terms": [
                    *LOGIT_DIFFERENCE["terms"],
                    {"reduce": "norm", "set": "read", "tokens": "all", "weight": 2.0},
                    {"reduce": "sum", "set": "read_two", "tokens": {"positions": [0, 2]}, "weight": -0.25},
                ]
            }
        )
        assert set(readout.sites) == {"blocks.2.hook_in", READ}
        logits, acts = _inputs(1)
        two = torch.cat([acts["blocks.2.hook_in"][..., [1, 3]], acts[READ][..., [1, 3]]], dim=-1)
        expected = (
            (logits[:, -1, A] - logits[:, -1, B]).sum()
            + 2.0 * acts[READ].norm(dim=-1).sum()
            - 0.25 * two[:, [0, 2], :].sum()
        )
        torch.testing.assert_close(readout(logits, acts), expected)

    def test_its_gradient_is_the_hand_written_readouts(self):
        readout = _readout(
            {"terms": [*LOGIT_DIFFERENCE["terms"], {"reduce": "project", "set": "read", "operand": "direction"}]}
        )
        logits, acts = _inputs(2)
        mine = {"logits": logits.clone().requires_grad_(), READ: acts[READ].clone().requires_grad_()}
        theirs = {k: v.detach().clone().requires_grad_() for k, v in mine.items()}
        readout(mine["logits"], {READ: mine[READ]}).backward()
        hand = (theirs["logits"][:, -1, A] - theirs["logits"][:, -1, B]).sum() + (
            theirs[READ] @ torch.tensor(DIRECTION)
        ).sum()
        hand.backward()
        for k in mine:
            torch.testing.assert_close(mine[k].grad, theirs[k].grad)

    def test_a_stage_readout_replaces_the_treatments(self):
        stage = {
            "site": "hook_out",
            "tokens": "last",
            "readout": {"terms": [{"reduce": "sum", "set": "foil", "tokens": "last"}]},
            "ops": [{"op": "scale", "target": {"set": "edit_grad"}, "value": {"dose": 0.0}}],
        }
        readout = _readout(None, stages=[stage])
        logits, acts = _inputs()
        torch.testing.assert_close(readout(logits, acts), logits[:, -1, B].sum())

    def test_a_recipe_written_before_readouts_leaves_the_default(self):
        doc = _doc(schema=2, operands={})
        del doc["operands"]
        assert recipe_readout(load_recipe(doc).recipe, n_layers=8) is None


class TestRefusals:
    def _stage(self, grad_set, readout=None):
        stage = {
            "site": "hook_out",
            "tokens": "last",
            "ops": [{"op": "scale", "target": {"set": grad_set}, "value": {"dose": 0.0}}],
        }
        if readout is not None:
            stage["readout"] = readout
        return stage

    def test_stages_taking_gradients_of_different_readouts_are_refused(self):
        stages = [self._stage("edit_grad"), self._stage("other_grad", LOGIT_DIFFERENCE)]
        readout = {"terms": [{"reduce": "sum", "set": "answer", "tokens": "last"}]}
        with pytest.raises(MisconfigurationException, match=r"different readouts \(stages \[0, 1\]\)"):
            _readout(readout, stages=stages)

    def test_stages_sharing_one_readout_are_accepted(self):
        stages = [self._stage("edit_grad"), self._stage("other_grad")]
        readout = _readout(LOGIT_DIFFERENCE, stages=stages)
        logits, acts = _inputs()
        torch.testing.assert_close(readout(logits, acts), (logits[:, -1, A] - logits[:, -1, B]).sum())

    def test_a_recipe_with_no_gradient_stage_is_refused(self):
        stage = {"site": "blocks.3.hook_out", "tokens": "last", "ops": [{"op": "scale", "value": {"dose": 0.0}}]}
        with pytest.raises(MisconfigurationException, match="has no stage acting on a gradient"):
            recipe_readout(load_recipe(_doc(schema=2, operands={}, stages=[stage])).recipe, n_layers=8)

    def test_a_term_mixing_the_logits_with_a_block_site_is_refused(self):
        with pytest.raises(MisconfigurationException, match=r"mixes the logits .* with block sites \['hook_in'\]"):
            _readout({"terms": [{"reduce": "sum", "set": "mixed", "tokens": "last"}]})

    def test_a_term_reading_a_global_point_other_than_the_logits_is_refused(self):
        with pytest.raises(MisconfigurationException, match=r"\['ln_final.hook_normalized'\] are global points"):
            _readout({"terms": [{"reduce": "sum", "set": "outside", "tokens": "last"}]})

    @pytest.mark.parametrize(
        "tokens, match",
        [("prompt", "token scope `prompt` has no interpretune spelling"), ({"positions": [1, 1]}, "must be distinct")],
    )
    def test_a_token_scope_without_an_interpretune_spelling_is_refused(self, tokens, match):
        with pytest.raises(MisconfigurationException, match=match):
            _readout({"terms": [{"reduce": "sum", "set": "answer", "tokens": tokens}]})

    def test_a_position_past_the_sequence_is_refused_when_read(self):
        readout = _readout({"terms": [{"reduce": "sum", "set": "answer", "tokens": {"positions": [9]}}]})
        with pytest.raises(MisconfigurationException, match="position 9 is outside a sequence of 4 columns"):
            readout(*_inputs())

    def test_an_operand_outside_the_declared_bases_or_of_another_width_is_refused(self):
        term = {"reduce": "project", "set": "read", "tokens": "last", "operand": "direction"}
        recipe = load_recipe(_doc({"terms": [term]})).recipe
        with pytest.raises(MisconfigurationException, match="basis `jlens_folded`, which is not one of"):
            recipe_readout(recipe, n_layers=8, bases=frozenset({"vocab"}))
        operands = {"direction": {"values": [1.0, 2.0], "shape": [2], "basis": "jlens_folded"}}
        readout = _readout({"terms": [term]}, operands=operands)
        with pytest.raises(MisconfigurationException, match="operand of 2 values for 6 addressed units"):
            readout(*_inputs())

    def test_a_reduction_without_an_executor_is_refused_by_name(self):
        from venngram import readouts as vg

        vg.register_readout(
            "unexecuted",
            meaning="sum_t x_t[0]: the first addressed value at each token",
            cases=[
                {"name": "unexecuted_first", "terms": [{"reduce": "unexecuted", "values": [[1, 2]]}], "expect": 1.0}
            ],
        )
        try:
            with pytest.raises(MisconfigurationException, match="reduction `unexecuted` has no executor here"):
                _readout({"terms": [{"reduce": "unexecuted", "set": "read", "tokens": "last"}]})
        finally:
            vg.READOUTS.pop("unexecuted", None)
            vg._REGISTERED_CASES.pop("unexecuted", None)


def test_the_support_record_declares_the_executed_reductions_only_where_it_acts_on_gradients():
    both = InterventionSupport(position_scopes={PositionScope.ALL_POSITIONS}, ops={"scale"}, of={"value", "grad"})
    values = InterventionSupport(position_scopes={PositionScope.ALL_POSITIONS}, ops={"scale"}, of={"value"})
    assert venngram_support(both).readouts == executed_readouts()
    assert venngram_support(values).readouts == frozenset()


def test_check_recipe_refuses_a_reduction_the_record_does_not_declare():
    from interpretune.analysis.venngram import check_recipe

    recipe = load_recipe(_doc({"terms": [{"reduce": "norm", "set": "read", "tokens": "last"}]})).recipe
    both = InterventionSupport(position_scopes={PositionScope.LAST_TOKEN}, ops={"scale"}, of={"value", "grad"})
    narrowed = copy.replace(venngram_support(both), readouts=frozenset({"sum"}))
    assert any("reduction `norm` is not computed" in r for r in check_recipe(recipe, narrowed, flow="grad"))
    assert check_recipe(recipe, venngram_support(both), flow="grad") == []
