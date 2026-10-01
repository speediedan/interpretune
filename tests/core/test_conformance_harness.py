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
from interpretune.testing.conformance.inputs import ConformanceInputs
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

    def test_a_locally_staged_collection_is_identified_by_its_declared_name(self):
        """A collection repository validates its own working tree, loaded from an op path, before publishing."""
        from interpretune.testing.conformance.collections import belongs_to_collection

        declared = self._def("op", source="local", collection_name="my_ops")
        other = self._def("op2", source="local", collection_name="other_ops")
        undeclared = self._def("op3", source="local")
        assert belongs_to_collection(declared, "my_ops")
        assert not belongs_to_collection(other, "my_ops") and not belongs_to_collection(undeclared, "my_ops")
        composite = self._def("op_then_op2", source="local", composition=["op", "op2"])
        assert belongs_to_collection(composite, "my_ops", family_members={"op", "op2"})
        assert not belongs_to_collection(composite, "my_ops", family_members={"op"})
        # A local op is never admitted by the bundled implementation-path rule, whatever its implementation.
        impostor = self._def("op4", source="local", implementation="interpretune.analysis.ops.bundled.my_ops.x_impl")
        assert not belongs_to_collection(impostor, "my_ops")

    def test_stage_local_collection_loads_a_working_tree_the_cases_can_read(self, tmp_path, monkeypatch):
        """The ``load`` hook a collection repository uses: the ops appear in the dispatcher without a re-import."""
        from interpretune.analysis.ops import dispatcher as dispatcher_module
        from interpretune.analysis.ops.base import OpWrapper
        from interpretune.testing.conformance.collections import ops_in_collection, stage_local_collection

        op_dir = tmp_path / "collection"
        op_dir.mkdir()
        (op_dir / "ops.yaml").write_text(
            "collection:\n  name: staged_ops\n  version: 0.1.0\n\n"
            "staged_op:\n  description: fixture op\n"
            "  implementation: interpretune.analysis.ops.bundled.core.core_ops.model_fwd_impl\n"
            "  input_schema: {}\n  output_schema: {}\n"
        )
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        fresh = dispatcher_module.AnalysisOpDispatcher(enable_hub_ops=False)
        fresh._cache_manager.cache_dir = cache_dir
        monkeypatch.setattr(dispatcher_module, "DISPATCHER", fresh)
        monkeypatch.setattr(OpWrapper, "_target_module", None)  # keep the fixture op off the real `it` namespace

        fresh.load_definitions()
        assert not ops_in_collection("staged_ops"), "the collection must be absent before it is staged"
        stage_local_collection(op_dir)()
        assert set(ops_in_collection("staged_ops")) == {"staged_op"}
        stage_local_collection(op_dir)()
        assert [p for p in fresh.yaml_paths if p.resolve() == op_dir.resolve()] == [op_dir], "staging twice duplicated"

    def test_an_override_cannot_stand_in_for_an_undeclared_sample(self):
        """``run_input_overrides`` fills a declared ``conformance.run_inputs`` sample; naming an op that declares
        none is refused by name rather than silently running (or silently skipping) an undeclared sample."""
        from interpretune.testing.conformance.collections import OpCollectionConformance

        class _Probe(OpCollectionConformance):
            collection = "my_ops"
            run_input_overrides = {"undeclared_op": {"x": 1}}

        ops = {"undeclared_op": self._def("undeclared_op", source="local", collection_name="my_ops")}
        with pytest.raises(AssertionError, match=r"run_input_overrides name \['undeclared_op'\]"):
            _Probe().test_each_op_runs_on_its_declared_sample(suite=None, collection_ops=ops)

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


class TestWorkingDirectoryLifetime:
    """An inputs object creates no directory until a session needs one, and removes it on cleanup.

    Measured before the fix: one ``it_conformance_*`` directory per run, created at import for every class that
    declares its inputs, never removed, tens of gigabytes each, 451 GB after five days on a shared host.
    """

    def test_construction_creates_nothing_and_first_use_creates_one(self, tmp_path, monkeypatch):
        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        inputs = ConformanceInputs()
        assert inputs.workdir is None and not list(tmp_path.glob("it_conformance_*"))
        created = inputs._ensure_workdir()
        assert created.is_dir() and created.parent == tmp_path and created is inputs._ensure_workdir()

    def test_cleanup_removes_the_directory_and_a_later_use_creates_a_fresh_one(self, tmp_path, monkeypatch):
        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        inputs = ConformanceInputs()
        first = inputs._ensure_workdir()
        (first / "cache").mkdir()
        inputs.cleanup()
        assert not first.exists() and inputs.workdir is None
        inputs.cleanup()  # idempotent
        second = inputs._ensure_workdir()
        assert second.is_dir() and second != first


class TestReportArtifact:
    """The selection report as data: per-target declarations rendered field by field, with provenance."""

    def _caps(self):
        from interpretune.analysis.backends import (
            AnalysisBackendCapability,
            FeatureInterventionSupport,
            InterventionMode,
            InterventionSupport,
            ModelBackendCapability,
            ModuleCapabilities,
            PositionScope,
        )

        return ModuleCapabilities(
            model=frozenset({ModelBackendCapability.ACTIVATION_INTERVENTION}),
            analysis=frozenset({AnalysisBackendCapability.FEATURE_INTERVENTION}),
            intervention=InterventionSupport(
                modes=frozenset({InterventionMode.ADD}), position_scopes=frozenset({PositionScope.LAST_TOKEN})
            ),
            attribution_graph=None,
            feature_intervention=FeatureInterventionSupport(value_sources=frozenset({"constant"})),
        )

    def test_describe_capabilities_is_plain_data_with_absent_surfaces_absent(self):
        import json

        from interpretune.testing.conformance.gates import describe_capabilities

        data = describe_capabilities(self._caps(), composition=("core", "x"), model_id="gpt2")
        json.dumps(data)  # serializable
        assert data["model_capabilities"] == ["activation_intervention"]
        assert data["analysis_capabilities"] == ["feature_intervention"]
        assert data["intervention"]["modes"] == ["add"] and data["intervention"]["position_scopes"] == ["last_token"]
        assert data["feature_intervention"]["value_sources"] == ["constant"]
        assert "latent_models" not in data and "capture" not in data and "attribution_graph" not in data
        assert data["composition"] == ["core", "x"] and data["model_id"] == "gpt2"

    def test_the_artifact_carries_targets_outcomes_and_provenance(self, tmp_path):
        import json

        from interpretune.testing.conformance.gates import SelectionReport, describe_capabilities
        from interpretune.testing.conformance.plugin import write_report_artifact

        report = SelectionReport()
        report.targets["TestX"] = describe_capabilities(self._caps(), composition=("core", "x"), model_id="gpt2")
        report.record("TestX::case_a", "ran")
        report.record("TestX::case_b", "skipped-undeclared")
        path = tmp_path / "out" / "report.json"
        write_report_artifact(report, str(path), exitstatus=0)
        artifact = json.loads(path.read_text())
        assert artifact["format"] == "interpretune.conformance.report/1"
        assert artifact["targets"]["TestX"]["intervention"]["modes"] == ["add"]
        assert artifact["ran"] == ["TestX::case_a"] and artifact["skipped_undeclared"] == ["TestX::case_b"]
        prov = artifact["provenance"]
        assert prov["exit_status"] == 0 and prov["interpretune_version"] and prov["measured_at"].endswith("+00:00")
        assert "git_head" in prov


class TestGenerationsStayPerRun:
    """A conformance session's runner carries no cross-run generator cache key and no shared cache dir.

    The cases on one target run the same op with different ``run_inputs`` (intervention mode, scale, scope,
    vector), so a key built from the target and the suite inputs cannot tell them apart: once such a key took
    effect, cases were served each other's stores. A shared dir without a working key only accumulated files
    nothing read. The runner config the session builder hands over is where both would enter.
    """

    def test_the_runner_config_names_no_generator_cache(self, monkeypatch):
        from types import SimpleNamespace

        import interpretune
        from interpretune.testing.conformance import session as session_mod
        from interpretune.testing.conformance.inputs import ConformanceTarget

        captured: dict = {}

        class _Runner:
            def __init__(self, run_cfg):
                captured.update(run_cfg)

        class _Session:
            def __init__(self, _cfg):
                self.module = object()
                self.datamodule = SimpleNamespace(test_dataloader=lambda: [])

        # Patch the module objects: the builder resolves both names from the package at call time.
        monkeypatch.setattr(interpretune, "AnalysisRunner", _Runner)
        monkeypatch.setattr(interpretune, "ITSession", _Session)
        monkeypatch.setattr(session_mod, "_require_suite_dependencies", lambda: None)
        monkeypatch.setattr(session_mod, "register_conformance_ops", lambda: None)
        monkeypatch.setattr(session_mod, "get_module_capabilities", lambda _module: None)
        target = ConformanceTarget(composition=("core",), session_cfg_factory=lambda _inputs: object())

        session_mod.build_conformance_session(target, ConformanceInputs())

        assert "it_session" in captured, "the builder did not construct its runner through the patched class"
        leaked = sorted(k for k in ("dataset_fingerprint", "generator_cache_dir") if k in captured)
        assert not leaked, (
            f"the conformance runner config carries {leaked}; a key from the target and suite inputs alone "
            "cannot separate cases that differ only in run_inputs"
        )


class TestPayloadRefs:
    """`{fixture: <name>}` sample values resolve to tensors built for the session under test (#450)."""

    def test_a_fixture_ref_resolves_through_the_factory(self):
        from interpretune.testing.conformance.payloads import resolve_payload_refs

        seen = {}
        resolved = resolve_payload_refs(
            {"intervention_tensor": {"fixture": "probe_dir"}, "scale": 1.0},
            {"probe_dir": lambda session: seen.setdefault("built", object())},
            object(),
        )
        assert resolved["intervention_tensor"] is seen["built"]
        assert resolved["scale"] == 1.0

    def test_plain_mappings_pass_through_untouched(self):
        """Only exactly-`{fixture: <name>}` is a reference; every other mapping stays a mapping."""
        from interpretune.testing.conformance.payloads import resolve_payload_refs

        run_inputs = {"nested": {"a": 1}, "lst": [1], "s": "x"}
        assert resolve_payload_refs(run_inputs, {}, object()) == run_inputs

    def test_an_unknown_fixture_is_refused_naming_the_known(self):
        from interpretune.testing.conformance.payloads import resolve_payload_refs

        with pytest.raises(ValueError, match=r"unknown fixture 'nope'.*known: \['yes'\]"):
            resolve_payload_refs({"t": {"fixture": "nope"}}, {"yes": lambda s: 1}, object())

    def test_an_almost_ref_is_refused_rather_than_run_as_a_plain_dict(self):
        """A `fixture` key beside siblings reads as a half-understood reference, not data."""
        from interpretune.testing.conformance.payloads import resolve_payload_refs

        with pytest.raises(ValueError, match="exactly \\{'fixture': <name>\\}"):
            resolve_payload_refs({"t": {"fixture": "x", "other": 1}}, {"x": lambda s: 1}, object())

    def test_a_declared_sample_with_a_ref_runs_resolved_end_to_end(self):
        """The wiring: a staged op's sample carrying a ref reaches `suite.run` with the tensor."""
        from types import SimpleNamespace

        from interpretune.analysis.ops.base import ColCfg, OpSchema
        from interpretune.testing.conformance.collections import OpCollectionConformance
        from interpretune.testing.conformance.inputs import ConformanceInputs

        sentinel = object()
        received = {}

        def fake_run(cfg):
            received.update(cfg.run_inputs)
            return SimpleNamespace(dataset=SimpleNamespace(column_names=["out"]))

        op_def = TestCollectionSelection._def("wired_op")
        import dataclasses

        op_def = dataclasses.replace(
            op_def,
            conformance={"run_inputs": {"intervention_tensor": {"fixture": "probe_dir"}}},
            output_schema=OpSchema({"out": ColCfg(datasets_dtype="float32")}),
        )

        probe = OpCollectionConformance()
        probe.inputs = ConformanceInputs(payload_fixtures={"probe_dir": lambda session: sentinel})
        probe.collection = "wired"
        # A real op name: AnalysisCfg resolves target_op against the dispatcher at construction,
        # and the wiring under test is the ref resolution, not the op. The collection defs stay
        # staged fakes carrying the declared sample.
        real = dataclasses.replace(op_def, name="concept_direction")
        probe.test_each_op_runs_on_its_declared_sample(
            suite=SimpleNamespace(run=fake_run), collection_ops={"concept_direction": real}
        )
        assert received["intervention_tensor"] is sentinel
