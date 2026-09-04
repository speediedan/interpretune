"""Unit tests for the conformance harness itself: selection, the report, and the vacuity guards.

These never build a session. They pin the logic that decides WHICH cases run, because that logic is what
turns "each adapter validates what it declares" from a sentence into a property.
"""

from __future__ import annotations


from interpretune.analysis.backends import (
    BackendCapability,
    InterventionMode,
    InterventionSupport,
    LatentModelSupport,
    ModuleCapabilities,
    PositionScope,
)
from interpretune.testing.conformance.gates import UNDECLARED, Gate, SelectionReport, conformance_case, gate_of
from interpretune.testing.conformance.plugin import vacuity_problems


def _caps(*, intervention=None, latent=None, analysis=frozenset()):
    model = set()
    if intervention is not None:
        model.add(BackendCapability.INTERVENTION)
    if latent is not None:
        model.add(BackendCapability.LATENT_MODELS)
    return ModuleCapabilities(
        model=frozenset(model), analysis=frozenset(analysis), intervention=intervention, latent_models=latent
    )


ADD_LAST = InterventionSupport(position_scopes={PositionScope.LAST_TOKEN}, modes={InterventionMode.ADD})


class TestGateSelection:
    def test_always_on_selects_everything(self):
        assert Gate().selects(_caps(), family="hf_native")

    def test_capability_gate_follows_the_declaration(self):
        g = Gate(capability=BackendCapability.INTERVENTION)
        assert g.selects(_caps(intervention=ADD_LAST), family="x")
        assert not g.selects(_caps(), family="x")

    def test_scope_and_mode_gates_read_the_support_record(self):
        assert Gate(capability=BackendCapability.INTERVENTION, scope=PositionScope.LAST_TOKEN).selects(
            _caps(intervention=ADD_LAST), family="x"
        )
        assert not Gate(capability=BackendCapability.INTERVENTION, scope=PositionScope.ALL_POSITIONS).selects(
            _caps(intervention=ADD_LAST), family="x"
        )
        assert not Gate(capability=BackendCapability.INTERVENTION, mode=InterventionMode.REPLACE).selects(
            _caps(intervention=ADD_LAST), family="x"
        )

    def test_negative_gate_inverts(self):
        g = Gate(capability=BackendCapability.INTERVENTION, scope=PositionScope.ALL_POSITIONS, negative=True)
        assert g.selects(_caps(intervention=ADD_LAST), family="x")
        assert not g.selects(_caps(intervention=InterventionSupport.every()), family="x")

    def test_family_gate(self):
        assert Gate(family="hf_native").selects(_caps(), family="hf_native")
        assert not Gate(family="hf_native").selects(_caps(), family="weight_converted")

    def test_batched_hooks_gate(self):
        assert Gate(capability=BackendCapability.LATENT_MODELS, batched_hooks=True).selects(
            _caps(latent=LatentModelSupport(True)), family="x"
        )
        assert not Gate(capability=BackendCapability.LATENT_MODELS, batched_hooks=True).selects(
            _caps(latent=LatentModelSupport(False)), family="x"
        )

    def test_scope_compares_by_value(self):
        """A second load of the enum module yields identity-distinct members; the gate must not care."""

        class _Rec:
            position_scopes = frozenset({"last_token"})
            modes = frozenset({"add"})

        caps = ModuleCapabilities(
            model=frozenset({BackendCapability.INTERVENTION}),
            analysis=frozenset(),
            intervention=InterventionSupport(position_scopes={"last_token"}, modes={"add"}),
        )
        assert Gate(capability=BackendCapability.INTERVENTION, scope=PositionScope.LAST_TOKEN).selects(caps, family="x")

    def test_describe_names_every_axis(self):
        g = Gate(
            capability=BackendCapability.INTERVENTION,
            scope=PositionScope.LAST_TOKEN,
            mode=InterventionMode.ADD,
            negative=True,
        )
        assert g.describe() == "NOT INTERVENTION, scope=last_token, mode=add"


class TestDecorator:
    def test_marks_the_function(self):
        @conformance_case(capability=BackendCapability.GRADIENTS)
        def f():
            pass

        assert gate_of(f) == Gate(capability=BackendCapability.GRADIENTS)
        assert gate_of(lambda: None) is None


class TestReportAndVacuity:
    def test_nothing_ran_is_a_problem(self):
        r = SelectionReport()
        r.record("a", "skipped-undeclared")
        assert vacuity_problems(r, strict=False)

    def test_one_ran_is_fine(self):
        r = SelectionReport()
        r.record("a", "ran")
        r.record("b", "skipped-undeclared")
        assert not vacuity_problems(r, strict=False)

    def test_other_skips_fail_only_under_strict(self):
        r = SelectionReport()
        r.record("a", "ran")
        r.record("b", "skipped-other")
        assert not vacuity_problems(r, strict=False)
        assert vacuity_problems(r, strict=True)

    def test_render_prints_all_four_counts(self):
        r = SelectionReport(declared=["INTERVENTION"])
        r.record("a", "ran")
        text = r.render()
        for needle in ("declared", "ran", "undeclared", "other", "failed"):
            assert needle in text

    def test_undeclared_reason_constant_is_stable(self):
        assert UNDECLARED == "undeclared"
