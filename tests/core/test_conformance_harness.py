"""Unit tests for the conformance harness itself: selection, the report, and the vacuity guards.

These never build a session. They pin the logic that decides WHICH cases run, because that logic is what
turns "each adapter validates what it declares" from a sentence into a property.
"""

from __future__ import annotations


import pytest

from interpretune.analysis.backends import (
    ModelBackendCapability,
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
        model.add(ModelBackendCapability.ACTIVATION_INTERVENTION)
    if latent is not None:
        model.add(ModelBackendCapability.LATENT_MODELS)
    return ModuleCapabilities(
        model=frozenset(model), analysis=frozenset(analysis), intervention=intervention, latent_models=latent
    )


ADD_LAST = InterventionSupport(position_scopes={PositionScope.LAST_TOKEN}, modes={InterventionMode.ADD})


class TestGateSelection:
    def test_always_on_selects_everything(self):
        assert Gate().selects(_caps(), family="hf_native")

    def test_capability_gate_follows_the_declaration(self):
        g = Gate(capability=ModelBackendCapability.ACTIVATION_INTERVENTION)
        assert g.selects(_caps(intervention=ADD_LAST), family="x")
        assert not g.selects(_caps(), family="x")

    def test_scope_and_mode_gates_read_the_support_record(self):
        assert Gate(capability=ModelBackendCapability.ACTIVATION_INTERVENTION, scope=PositionScope.LAST_TOKEN).selects(
            _caps(intervention=ADD_LAST), family="x"
        )
        assert not Gate(
            capability=ModelBackendCapability.ACTIVATION_INTERVENTION, scope=PositionScope.ALL_POSITIONS
        ).selects(_caps(intervention=ADD_LAST), family="x")
        assert not Gate(
            capability=ModelBackendCapability.ACTIVATION_INTERVENTION, mode=InterventionMode.REPLACE
        ).selects(_caps(intervention=ADD_LAST), family="x")

    def test_negative_gate_inverts(self):
        g = Gate(
            capability=ModelBackendCapability.ACTIVATION_INTERVENTION, scope=PositionScope.ALL_POSITIONS, negative=True
        )
        assert g.selects(_caps(intervention=ADD_LAST), family="x")
        assert not g.selects(_caps(intervention=InterventionSupport.every()), family="x")

    def test_family_gate(self):
        assert Gate(family="hf_native").selects(_caps(), family="hf_native")
        assert not Gate(family="hf_native").selects(_caps(), family="weight_converted")

    def test_batched_hooks_gate(self):
        assert Gate(capability=ModelBackendCapability.LATENT_MODELS, batched_hooks=True).selects(
            _caps(latent=LatentModelSupport(True)), family="x"
        )
        assert not Gate(capability=ModelBackendCapability.LATENT_MODELS, batched_hooks=True).selects(
            _caps(latent=LatentModelSupport(False)), family="x"
        )

    def test_scope_compares_by_value(self):
        """A second load of the enum module yields identity-distinct members; the gate must not care."""

        class _Rec:
            position_scopes = frozenset({"last_token"})
            modes = frozenset({"add"})

        caps = ModuleCapabilities(
            model=frozenset({ModelBackendCapability.ACTIVATION_INTERVENTION}),
            analysis=frozenset(),
            intervention=InterventionSupport(position_scopes={"last_token"}, modes={"add"}),
        )
        assert Gate(capability=ModelBackendCapability.ACTIVATION_INTERVENTION, scope=PositionScope.LAST_TOKEN).selects(
            caps, family="x"
        )

    def test_a_scopes_gate_needs_every_scope_declared(self):
        """A mixed-scope case selects only when the target declared BOTH scopes; one of two is undeclared."""
        both = InterventionSupport(
            position_scopes={PositionScope.LAST_TOKEN, PositionScope.ALL_POSITIONS}, modes={InterventionMode.ADD}
        )
        g = Gate(
            capability=ModelBackendCapability.ACTIVATION_INTERVENTION,
            scopes=(PositionScope.LAST_TOKEN, PositionScope.ALL_POSITIONS),
        )
        assert g.selects(_caps(intervention=both), family="x")
        assert not g.selects(_caps(intervention=ADD_LAST), family="x")
        assert not g.selects(_caps(), family="x")
        assert g.describe() == "ACTIVATION_INTERVENTION, scopes=last_token+all_positions"

    def test_single_prompt_gate_reads_the_target_not_the_backend(self):
        g = Gate(single_prompt=True)
        assert g.selects(_caps(), family="x", single_prompt=True)
        assert not g.selects(_caps(), family="x", single_prompt=False)
        assert not g.selects(_caps(), family="x")

    def test_describe_names_every_axis(self):
        g = Gate(
            capability=ModelBackendCapability.ACTIVATION_INTERVENTION,
            scope=PositionScope.LAST_TOKEN,
            mode=InterventionMode.ADD,
            negative=True,
        )
        assert g.describe() == "NOT ACTIVATION_INTERVENTION, scope=last_token, mode=add"


class TestDecorator:
    def test_marks_the_function(self):
        @conformance_case(capability=ModelBackendCapability.GRADIENTS)
        def f():
            pass

        assert gate_of(f) == Gate(capability=ModelBackendCapability.GRADIENTS)
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
        r = SelectionReport(declared=["ACTIVATION_INTERVENTION"])
        r.record("a", "ran")
        text = r.render()
        for needle in ("declared", "ran", "undeclared", "other", "failed"):
            assert needle in text

    def test_undeclared_reason_constant_is_stable(self):
        assert UNDECLARED == "undeclared"


class TestTheMarkerIsSilentWhenNoCaseWasInScope:
    """A run that collected no conformance case (an unrelated file, a `-k` on ordinary tests) gets no report and no
    marker: on the first hub adapter the marker printed on every targeted run, was read past for hours, and a real
    conformance failure reached CI that way."""

    def test_no_case_collected_means_no_report(self):
        from interpretune.testing.conformance.plugin import collected_any_case

        assert not collected_any_case(SelectionReport())
        assert collected_any_case(SelectionReport(skipped_undeclared=["a"]))
        assert collected_any_case(SelectionReport(ran=["a"]))

    def test_the_terminal_summary_prints_nothing_for_such_a_run(self):
        from unittest.mock import MagicMock

        from interpretune.testing.conformance.plugin import _REPORT_KEY, pytest_terminal_summary

        config = MagicMock()
        config.stash.get.return_value = SelectionReport()
        reporter = MagicMock()
        pytest_terminal_summary(reporter, 0, config)
        reporter.write_sep.assert_not_called()
        reporter.write_line.assert_not_called()
        config.stash.get.return_value = SelectionReport(skipped_undeclared=["a"])
        pytest_terminal_summary(reporter, 0, config)
        assert any("VACUITY" in str(c) for c in reporter.write_line.call_args_list), "a real vacuity still prints"
        assert _REPORT_KEY is not None


class TestExpectRefusal:
    def test_finds_a_wrapped_refusal(self):
        from interpretune.testing.conformance.oracles import expect_refusal

        with expect_refusal(NotImplementedError, match="mode='replace'"):
            try:
                raise NotImplementedError("backend cannot apply mode='replace'")
            except NotImplementedError as inner:
                raise RuntimeError("An error occurred while generating the dataset") from inner

    def test_direct_refusal_still_matches(self):
        from interpretune.testing.conformance.oracles import expect_refusal

        with expect_refusal(NotImplementedError, match="all_positions"):
            raise NotImplementedError("position_scope='all_positions' is not declared")

    def test_nothing_raised_is_a_failure(self):
        from interpretune.testing.conformance.oracles import expect_refusal

        with pytest.raises(AssertionError, match="nothing was raised"):
            with expect_refusal(NotImplementedError, match="x"):
                pass

    def test_the_wrong_exception_is_a_failure_naming_what_was_got(self):
        from interpretune.testing.conformance.oracles import expect_refusal

        with pytest.raises(AssertionError, match="got RuntimeError"):
            with expect_refusal(NotImplementedError, match="x"):
                raise RuntimeError("unrelated")


class TestCollectionSelection:
    """`OpCollectionConformance` reads a collection off the dispatcher's canonical definitions."""

    @staticmethod
    def _def(name, *, implementation="", source="bundled", collection_name=None, composition=None):
        from interpretune.analysis.ops.base import OpSchema
        from interpretune.analysis.ops.compiler.cache_manager import OpDef

        return OpDef(
            name=name,
            description="",
            implementation=implementation,
            input_schema=OpSchema({}),
            output_schema=OpSchema({}),
            source=source,
            collection_name=collection_name,
            composition=composition,
        )

    def test_a_bundled_family_is_its_leaves_plus_composites_of_them(self):
        from interpretune.testing.conformance.collections import belongs_to_collection

        leaf = self._def("x", implementation="interpretune.analysis.ops.bundled.concept.concept_ops.x_impl")
        other = self._def("y", implementation="interpretune.analysis.ops.bundled.sae.sae_ops.y_impl")
        assert belongs_to_collection(leaf, "concept") and not belongs_to_collection(other, "concept")
        composite = self._def("x_then_y", composition=["x", "y"])
        assert belongs_to_collection(composite, "concept", family_members={"x", "y"})
        assert not belongs_to_collection(composite, "concept", family_members={"x"})
        assert not belongs_to_collection(composite, "concept")  # no members admitted: a composite alone is nothing

    def test_a_hub_collection_is_identified_by_declaration_or_provenance(self):
        from interpretune.testing.conformance.collections import belongs_to_collection

        declared = self._def("org.repo.op", source="hub:org.repo", collection_name="org/repo")
        by_provenance = self._def("org.repo.op2", source="hub:org.repo")
        bundled = self._def("op", implementation="interpretune.analysis.ops.bundled.concept.concept_ops.op_impl")
        assert belongs_to_collection(declared, "org/repo") and belongs_to_collection(by_provenance, "org/repo")
        assert not belongs_to_collection(bundled, "org/repo")

    def test_the_bundled_concept_family_resolves_to_its_intervention_ops(self):
        from interpretune.testing.conformance.collections import ops_in_collection

        ops = ops_in_collection("concept")
        assert "model_fwd_intervention" in ops and "concept_direction" in ops
        assert all(d.name == n for n, d in ops.items()), "aliases must not appear beside their canonical entry"
        # a composite that mixes families belongs to none of them: `intervention_from_concept` composes concept
        # ops with circuit-tracer ops, so validating it "as the concept collection" would judge the wrong thing
        assert "intervention_from_concept" not in ops and "attribution_from_concept" not in ops


class TestSuppliedSettingsSurviveComposition:
    """The case is a plain method carrying its gate as an attribute, so it runs on a stub suite."""

    @staticmethod
    def _run(extras, cfg):
        from types import SimpleNamespace

        from interpretune.testing.conformance import ModelBackendConformance

        suite = SimpleNamespace(inputs=SimpleNamespace(supplied_extras=extras), module=SimpleNamespace(it_cfg=cfg))
        ModelBackendConformance.test_supplied_settings_survive_composition(ModelBackendConformance(), suite)

    def test_a_declared_field_holding_the_supplied_object_passes(self):
        from dataclasses import dataclass

        @dataclass
        class _Cfg:
            my_adapter_cfg: object = None

        value = object()
        self._run({"my_adapter_cfg": value}, _Cfg(my_adapter_cfg=value))

    def test_a_stray_attribute_fails_by_name(self):
        from dataclasses import dataclass

        @dataclass
        class _Cfg:
            other: int = 0

        cfg = _Cfg()
        value = object()
        cfg.my_adapter_cfg = value  # type: ignore[attr-defined]  # the seam this case exists to catch
        with pytest.raises(AssertionError, match="'my_adapter_cfg' was supplied .* _Cfg declares no such field"):
            self._run({"my_adapter_cfg": value}, cfg)

    def test_a_copied_object_fails_by_identity(self):
        from dataclasses import dataclass

        @dataclass
        class _Cfg:
            my_adapter_cfg: object = None

        with pytest.raises(AssertionError, match="reached the composed config as a different object"):
            self._run({"my_adapter_cfg": object()}, _Cfg(my_adapter_cfg=object()))

    def test_no_extras_skips_rather_than_passing(self):
        with pytest.raises(pytest.skip.Exception):
            self._run({}, object())
