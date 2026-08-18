"""Tests for the declared cross-batch op-state seam (the ``op_state`` trait and its lifecycle).

Covers the three pieces separately: the declaration (``OpStateSpec`` + YAML/OpDef compilation), the
container (``OpStateStore``), and the lifecycle owner (``AnalysisCfg``, driven by the analysis
runner). The concept family's use of the seam is exercised in ``test_bundled_concept_streaming.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interpretune.analysis.inputs import AnalysisInputs, OpStateSpec, OpStateStore
from interpretune.config.analysis import AnalysisCfg

SPEC = OpStateSpec(fields=("alpha", "beta"))


def _stub_op(name: str = "stub_op", spec: OpStateSpec | None = SPEC) -> SimpleNamespace:
    """A minimal stand-in for an AnalysisOp; ``op_state_for`` is duck-typed on name + op_state."""
    return SimpleNamespace(name=name, op_state=spec)


class TestOpStateSpec:
    def test_defaults(self):
        spec = OpStateSpec(fields=("alpha",))
        assert spec.scope == "run"
        assert spec.reset_each_epoch is False

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"fields": ("a",), "scope": "epoch"}, "Unsupported op_state scope"),
            ({"fields": ()}, "at least one field"),
            ({"fields": ("a", "a")}, "duplicate fields"),
            ({"fields": ("not an identifier",)}, "valid identifiers"),
        ],
    )
    def test_invalid_declarations_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            OpStateSpec(**kwargs)

    def test_from_raw_round_trips_through_to_dict(self):
        raw = {"fields": ["alpha", "beta"], "scope": "run", "reset_each_epoch": True}
        spec = OpStateSpec.from_raw(raw)
        assert spec is not None
        assert spec.fields == ("alpha", "beta") and spec.reset_each_epoch is True
        assert OpStateSpec.from_raw(spec.to_dict()) == spec

    def test_from_raw_passthrough_and_none(self):
        assert OpStateSpec.from_raw(None) is None
        assert OpStateSpec.from_raw(SPEC) is SPEC

    def test_from_raw_rejects_unrecognized_keys(self):
        with pytest.raises(ValueError, match="Unrecognized op_state keys"):
            OpStateSpec.from_raw({"fields": ["a"], "persist": True})

    def test_from_raw_rejects_non_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            OpStateSpec.from_raw(["alpha"])

    @pytest.mark.parametrize("fields", ["my_field", b"my_field"])
    def test_from_raw_rejects_a_bare_string_for_fields(self, fields):
        """`fields: my_field` is a natural YAML slip and `tuple()` would shred it per character.

        It previously loaded clean as eight single-character fields and only failed later, at the
        impl's first `state.get("my_field")`, with a message listing ['m','y','_',...]. Worse, it was
        inconsistent: `fields: concept` DID fail at load, but as "duplicate fields: ['c']". Declaring
        exists so this fails at load, every time (umbrella review finding, 2026-08-17).
        """
        with pytest.raises(ValueError, match="must be a sequence of names"):
            OpStateSpec.from_raw({"fields": fields})

    def test_a_single_field_list_is_still_accepted(self):
        assert OpStateSpec.from_raw({"fields": ["my_field"]}).fields == ("my_field",)


class TestOpStateStore:
    def test_set_get_and_clear(self):
        store = OpStateStore(SPEC)
        assert store.get("alpha") is None
        store.set("alpha", 1)
        store["beta"] = 2
        assert (store.get("alpha"), store["beta"]) == (1, 2)
        assert store.as_dict() == {"alpha": 1, "beta": 2}
        assert len(store) == 2 and "alpha" in store
        store.clear()
        assert store.as_dict() == {} and "alpha" not in store
        # the declared namespace survives a clear
        assert store.declared == ("alpha", "beta")

    def test_update_sets_several(self):
        store = OpStateStore(SPEC)
        store.update(alpha=1, beta=2)
        assert store.as_dict() == {"alpha": 1, "beta": 2}

    @pytest.mark.parametrize("operation", ["get", "set", "getitem", "setitem"])
    def test_undeclared_names_raise(self, operation):
        store = OpStateStore(SPEC)
        with pytest.raises(KeyError, match="not a declared op_state field"):
            if operation == "get":
                store.get("gamma")
            elif operation == "set":
                store.set("gamma", 1)
            elif operation == "getitem":
                store["gamma"]
            else:
                store["gamma"] = 1

    def test_repr_names_declared_and_set_fields(self):
        store = OpStateStore(SPEC)
        store.set("alpha", 1)
        text = repr(store)
        assert "alpha" in text and "beta" in text and "run" in text


class TestAnalysisInputsCarriesOpState:
    def test_op_state_is_not_a_resolution_scope(self):
        """`op_state` must not participate in scoped input resolution."""
        from interpretune.analysis.inputs import DEFAULT_ANALYSIS_SCOPES

        assert "op_state" not in DEFAULT_ANALYSIS_SCOPES
        with pytest.raises(ValueError, match="Unsupported analysis input scope"):
            AnalysisInputs().resolve_scope("op_state", "alpha")

    def test_merge_prefers_override_then_falls_back(self):
        mine, theirs = OpStateStore(SPEC), OpStateStore(SPEC)
        assert AnalysisInputs(op_state=mine).merged(AnalysisInputs(op_state=theirs)).op_state is theirs
        assert AnalysisInputs(op_state=mine).merged(AnalysisInputs()).op_state is mine
        assert AnalysisInputs(op_state=mine).merged(None).op_state is mine


class TestAnalysisCfgOwnsLifecycle:
    def test_ops_without_declared_state_get_nothing(self):
        cfg = AnalysisCfg()
        assert cfg.op_state_for(_stub_op(spec=None)) is None
        assert cfg.op_state_for(None) is None

    def test_container_is_stable_per_op_name(self):
        cfg = AnalysisCfg()
        first = cfg.op_state_for(_stub_op())
        assert first is cfg.op_state_for(_stub_op())

    def test_member_ops_of_a_composite_get_separate_namespaces(self):
        """Keyed by op name because a composite's MEMBERS are the state declarers."""
        cfg = AnalysisCfg()
        one = cfg.op_state_for(_stub_op("op_one"))
        two = cfg.op_state_for(_stub_op("op_two"))
        assert one is not two
        one.set("alpha", 1)
        assert two.get("alpha") is None

    def test_container_is_rebuilt_when_the_declaration_changes(self):
        cfg = AnalysisCfg()
        first = cfg.op_state_for(_stub_op())
        second = cfg.op_state_for(_stub_op(spec=OpStateSpec(fields=("gamma",))))
        assert second is not first
        assert second.declared == ("gamma",)

    def test_reset_clears_everything_at_a_run_boundary(self):
        cfg = AnalysisCfg()
        state = cfg.op_state_for(_stub_op())
        state.set("alpha", 1)
        cfg.reset_op_state()
        assert state.as_dict() == {}

    def test_epoch_boundary_reset_honors_the_declaration(self):
        cfg = AnalysisCfg()
        accumulating = cfg.op_state_for(_stub_op("accumulates"))
        per_epoch = cfg.op_state_for(_stub_op("per_epoch", OpStateSpec(fields=("alpha",), reset_each_epoch=True)))
        accumulating.set("alpha", 1)
        per_epoch.set("alpha", 1)
        cfg.reset_op_state(epoch_boundary=True)
        assert accumulating.get("alpha") == 1, "default reset_each_epoch=False must accumulate across epochs"
        assert per_epoch.get("alpha") is None

    def test_finalize_releases_state(self):
        cfg = AnalysisCfg()
        state = cfg.op_state_for(_stub_op())
        state.set("alpha", 1)
        cfg.finalize_op_state()
        assert state.as_dict() == {}


class TestOpBindsDeclaredState:
    def test_bind_is_a_noop_for_ops_without_declared_state(self):
        from interpretune.analysis.ops.base import AnalysisOp, OpSchema

        op = AnalysisOp(name="plain", description="", output_schema=OpSchema({}))
        kwargs: dict = {}
        op._bind_op_state(SimpleNamespace(analysis_cfg=AnalysisCfg()), kwargs)
        assert kwargs == {}

    def test_bind_is_a_noop_without_an_owner(self):
        """A bare op call has no run-scoped owner, so inputs reach the impl untouched."""
        from interpretune.analysis.ops.base import AnalysisOp, OpSchema

        op = AnalysisOp(name="stateful", description="", output_schema=OpSchema({}), op_state=SPEC)
        kwargs: dict = {}
        op._bind_op_state(SimpleNamespace(), kwargs)
        assert kwargs == {}

    def test_bind_attaches_the_cfg_owned_container(self):
        from interpretune.analysis.ops.base import AnalysisOp, OpSchema

        cfg = AnalysisCfg()
        op = AnalysisOp(name="stateful", description="", output_schema=OpSchema({}), op_state=SPEC)
        kwargs: dict = {}
        op._bind_op_state(SimpleNamespace(analysis_cfg=cfg), kwargs)
        assert kwargs["analysis_inputs"].op_state is cfg.op_state_for(op)

    def test_bind_preserves_a_caller_supplied_container(self):
        from interpretune.analysis.ops.base import AnalysisOp, OpSchema

        cfg = AnalysisCfg()
        caller_state = OpStateStore(SPEC)
        op = AnalysisOp(name="stateful", description="", output_schema=OpSchema({}), op_state=SPEC)
        kwargs = {"analysis_inputs": AnalysisInputs(op_state=caller_state)}
        op._bind_op_state(SimpleNamespace(analysis_cfg=cfg), kwargs)
        assert kwargs["analysis_inputs"].op_state is caller_state


class TestDeclarationCompilation:
    def test_bundled_concept_ops_declare_state_and_others_do_not(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        assert DISPATCHER._op_definitions["concept_direction"].op_state is not None
        assert DISPATCHER._op_definitions["extract_concept_latent_examples"].op_state is not None
        assert DISPATCHER._op_definitions["model_fwd"].op_state is None

    def test_malformed_declaration_warns_and_drops_the_trait(self):
        """Fail-soft, matching the surrounding YAML/compile paths so one bad hub op is survivable."""
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        with pytest.warns(UserWarning, match="Ignoring invalid op_state declaration"):
            assert AnalysisOpDispatcher._resolve_op_state_spec("bad_op", {"fields": []}) is None

    def test_cache_serialization_round_trips_the_trait(self):
        from interpretune.analysis.ops.compiler.cache_manager import OpDef, OpDefinitionsCacheManager

        spec = OpStateSpec(fields=("alpha", "beta"), reset_each_epoch=True)
        op_def = OpDef(name="x", description="", implementation="m.f", input_schema={}, output_schema={}, op_state=spec)
        assert op_def.to_dict()["op_state"] == spec.to_dict()
        serialized = OpDefinitionsCacheManager.__dict__["_serialize_op_state"](None, spec)
        assert eval(serialized, {"OpStateSpec": OpStateSpec}) == spec  # - generated cache is Python


class TestStreamingRequiresAnOwner:
    def test_concept_direction_streaming_without_state_says_what_is_missing(self):
        """Previously the writes were swallowed and this failed several frames later as a data error."""
        import torch

        from interpretune.analysis.ops.base import AnalysisBatch
        from interpretune.analysis.ops.bundled.concept.concept_ops import concept_direction_impl

        batch = AnalysisBatch(
            concept_latent_state=torch.ones(1, 2),
            concept_group_id=torch.tensor([0]),
            concept_aggregate_output_mode="streaming",
        )
        with pytest.raises(ValueError, match="requires this op's declared op_state"):
            concept_direction_impl(SimpleNamespace(analysis_cfg=None), batch, None, 0)


class TestBehavioralTraits:
    """`uses_default_hooks` / `requires_grad` / `per_latent_preds` replace op-name special cases."""

    def test_bundled_composites_declare_the_traits_they_need(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        grad = DISPATCHER._op_definitions["logit_diffs_attr_grad"]
        ablation = DISPATCHER._op_definitions["logit_diffs_attr_ablation"]
        base = DISPATCHER._op_definitions["logit_diffs_base"]
        assert (grad.requires_grad, grad.uses_default_hooks) == (True, True)
        assert ablation.per_latent_preds is True
        assert (base.requires_grad, base.uses_default_hooks, base.per_latent_preds) == (False, False, False)

    def test_traits_survive_composite_instantiation(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        op = DISPATCHER.get_op("logit_diffs_attr_grad")
        assert op.requires_grad and op.uses_default_hooks

    def test_composite_compilation_preserves_declared_keys(self):
        """Composite definitions used to be rebuilt from scratch, discarding every declared key.

        The visible symptom was authored descriptions being replaced by "Compiled composition: ...";
        the same drop is what prevented a composite from declaring a trait at all.
        """
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        described = DISPATCHER._op_definitions["attribution_from_concept"]
        assert described.description == "Concept direction through graph attribution and top-feature extraction"
        # A composite with no authored description still gets the generated one.
        assert DISPATCHER._op_definitions["logit_diffs_base"].description.startswith("Compiled composition:")

    @staticmethod
    def _op(name: str, **traits):
        from interpretune.analysis.ops.base import AnalysisOp, OpSchema

        return AnalysisOp(name=name, description="", output_schema=OpSchema({}), **traits)

    def test_default_hooks_are_driven_by_the_trait(self):
        cfg = AnalysisCfg()
        cfg.op = self._op("hooky", uses_default_hooks=True)
        installed = {}
        cfg.add_default_cache_hooks = lambda: installed.setdefault("called", True)  # type: ignore[method-assign]
        assert cfg.check_add_default_hooks() is None
        assert installed == {"called": True}

    def test_ops_without_the_trait_get_no_default_hooks(self):
        cfg = AnalysisCfg()
        cfg.op = self._op("plain")

        def _unexpected() -> None:
            pytest.fail("default hooks installed for an op that did not ask")

        cfg.add_default_cache_hooks = _unexpected  # type: ignore[method-assign]
        assert cfg.check_add_default_hooks() == ([], [])

    def test_no_op_clears_hooks(self):
        cfg = AnalysisCfg()
        cfg.fwd_hooks, cfg.bwd_hooks = ["stale"], ["stale"]
        assert cfg.check_add_default_hooks() is None
        assert (cfg.fwd_hooks, cfg.bwd_hooks) == ([], [])


class TestOpProvenance:
    """`OpDef.source` replaces dot-counting as the "is this a hub op" test."""

    def test_bundled_definitions_are_classified_bundled(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        sources = {d.source for d in DISPATCHER._op_definitions.values()}
        assert sources == {"bundled"}

    def test_source_classification_by_yaml_location(self, tmp_path):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        bundled_yaml = DISPATCHER._bundled_ops_dir / "core" / "core_ops.yaml"
        assert DISPATCHER._op_source_for(bundled_yaml) == "bundled"
        assert DISPATCHER._op_source_for(tmp_path / "my_ops.yaml") == "local"

    def test_hub_source_is_derived_from_the_cache_namespace(self, monkeypatch, tmp_path):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        monkeypatch.setattr(DISPATCHER._cache_manager, "get_hub_namespace", lambda _p: "someuser.some_repo")
        assert DISPATCHER._op_source_for(tmp_path / "hub_ops.yaml") == "hub:someuser.some_repo"

    def test_classification_never_breaks_loading(self, monkeypatch):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        def _boom(_path):
            raise RuntimeError("cache unavailable")

        monkeypatch.setattr(DISPATCHER._cache_manager, "get_hub_namespace", _boom)
        assert DISPATCHER._op_source_for("/nonexistent/ops.yaml") == "local"


class TestDottedNameDispatch:
    """A dotted name that resolves to one op is that op, not a composition of its dot-parts."""

    def test_resolvable_dotted_names_are_not_split(self, monkeypatch):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        monkeypatch.setitem(DISPATCHER._op_definitions, "someuser.repo.some_op", object())
        assert DISPATCHER._is_resolvable_op_name("someuser.repo.some_op")
        assert not DISPATCHER._is_resolvable_op_name("labels_to_ids.model_fwd")

    def test_unresolvable_dotted_name_still_composes(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        composite = DISPATCHER.compile_ops("labels_to_ids.model_fwd")
        assert [op.name for op in composite.composition] == ["labels_to_ids", "model_fwd"]


class TestImplSignatureCache:
    def test_signature_is_cached_per_callable(self):
        from interpretune.analysis.ops.base import _impl_signature

        def impl(module, analysis_batch, batch, batch_idx):
            return None

        assert _impl_signature(impl) is _impl_signature(impl)
        assert list(_impl_signature(impl).parameters) == ["module", "analysis_batch", "batch", "batch_idx"]

    def test_uninspectable_callables_degrade_to_none(self):
        from interpretune.analysis.ops.base import _impl_signature

        class Uninspectable:
            __signature__ = property(lambda self: (_ for _ in ()).throw(ValueError("no signature")))

            def __call__(self, *args, **kwargs):
                return None

        assert _impl_signature(Uninspectable()) is None


class TestStrictOpLoad:
    """`IT_STRICT_OP_LOAD=1` turns the fail-soft op-loading warn paths into errors."""

    def test_warns_by_default(self, monkeypatch):
        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, op_load_failure

        monkeypatch.delenv(IT_STRICT_OP_LOAD_ENV_VAR, raising=False)
        with pytest.warns(UserWarning, match="something went wrong"):
            op_load_failure("something went wrong")

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_raises_when_enabled(self, monkeypatch, value):
        from interpretune.analysis.ops.compiler.load_policy import (
            IT_STRICT_OP_LOAD_ENV_VAR,
            OpLoadError,
            op_load_failure,
            strict_op_load,
        )

        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, value)
        assert strict_op_load()
        with pytest.raises(OpLoadError, match="something went wrong"):
            op_load_failure("something went wrong")

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_non_affirmative_values_stay_fail_soft(self, monkeypatch, value):
        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, strict_op_load

        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, value)
        assert not strict_op_load()

    def test_invalid_op_state_raises_under_strict_load(self, monkeypatch):
        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        with pytest.raises(OpLoadError, match="invalid op_state declaration"):
            AnalysisOpDispatcher._resolve_op_state_spec("bad_op", {"fields": []})


class TestHubImportableParamRestriction:
    """A hub op may bind importable_params only to its own repo modules or the optools namespace."""

    @staticmethod
    def _dispatcher():
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        return AnalysisOpDispatcher(enable_hub_ops=True)

    @pytest.mark.parametrize(
        "param_path",
        [
            "interpretune.analysis.optools",
            "interpretune.analysis.optools.last_token_logits",
            "my_repo_module.helper",
            "torch.nn.functional.relu",
        ],
    )
    def test_sanctioned_targets_are_allowed(self, param_path):
        assert self._dispatcher()._hub_param_target_is_sanctioned("u.r.op", "fn", param_path)

    @pytest.mark.parametrize(
        "param_path",
        ["interpretune.analysis.core.some_internal", "interpretune.config.analysis.AnalysisCfg"],
    )
    def test_internal_targets_are_rejected_with_a_warning(self, param_path):
        with pytest.warns(UserWarning, match="interpretune-internal"):
            assert not self._dispatcher()._hub_param_target_is_sanctioned("u.r.op", "fn", param_path)

    def test_internal_targets_raise_under_strict_load(self, monkeypatch):
        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError

        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        with pytest.raises(OpLoadError, match="interpretune-internal"):
            self._dispatcher()._hub_param_target_is_sanctioned("u.r.op", "fn", "interpretune.analysis.core.thing")


class TestOpDefCacheRoundTrip:
    """The generated cache module is Python source, so every new OpDef field needs serializing."""

    def test_all_new_fields_survive_serialization(self):
        from interpretune.analysis.ops.base import ColCfg, OpSchema
        from interpretune.analysis.ops.compiler.cache_manager import (
            CACHE_FORMAT_VERSION,
            OpDef,
            OpDefinitionsCacheManager,
        )

        spec = OpStateSpec(fields=("alpha",), reset_each_epoch=True)
        op_def = OpDef(
            name="x",
            description="",
            implementation="m.f",
            input_schema=OpSchema({}),
            output_schema=OpSchema({}),
            op_state=spec,
            source="hub:user.repo",
            collection_name="user/repo",
            collection_version="2.1.0",
            uses_default_hooks=True,
            requires_grad=True,
            per_latent_preds=True,
        )
        serialized = OpDefinitionsCacheManager.__dict__["_serialize_op_def"](
            OpDefinitionsCacheManager.__new__(OpDefinitionsCacheManager), op_def
        )
        restored = eval(  # - the cache module is generated Python by design
            serialized,
            {"OpDef": OpDef, "OpSchema": OpSchema, "ColCfg": ColCfg, "OpStateSpec": OpStateSpec},
        )
        assert restored.op_state == spec
        assert restored.source == "hub:user.repo"
        # Collection identity must survive: unlike the declaration-site path it is wanted at RUNTIME
        # (op_info reports it), which is what earned it a place on OpDef and a cache-format bump.
        assert (restored.collection_name, restored.collection_version) == ("user/repo", "2.1.0")
        assert (restored.uses_default_hooks, restored.requires_grad, restored.per_latent_preds) == (True, True, True)
        # Defaults stay out of the serialized form so the cache does not grow for every new trait.
        plain = OpDef(
            name="y", description="", implementation="m.f", input_schema=OpSchema({}), output_schema=OpSchema({})
        )
        plain_serialized = OpDefinitionsCacheManager.__dict__["_serialize_op_def"](
            OpDefinitionsCacheManager.__new__(OpDefinitionsCacheManager), plain
        )
        assert "source=" not in plain_serialized and "requires_grad" not in plain_serialized
        assert "collection_name" not in plain_serialized
        # Adding fields to OpDef without bumping this makes stale caches deserialize silently wrong.
        assert CACHE_FORMAT_VERSION == "4"


class TestOpStateIsNotSerialized:
    """`AnalysisCfg` gets pickled and yaml-dumped; run-scoped accumulator state must not ride along."""

    def test_pickling_keeps_the_declaration_and_drops_the_values(self):
        import pickle

        store = OpStateStore(SPEC)
        store.set("alpha", list(range(1000)))
        restored = pickle.loads(pickle.dumps(store))
        assert restored.spec == SPEC
        assert restored.as_dict() == {}

    def test_yaml_dumping_a_cfg_does_not_embed_state(self):
        import yaml

        cfg = AnalysisCfg(name="serializable")
        cfg.op_state_for(_stub_op()).set("alpha", "SENTINEL_STATE_VALUE")
        assert "SENTINEL_STATE_VALUE" not in yaml.dump(cfg)


class TestBundledDefinitionsLoadStrictly:
    def test_bundled_ops_compile_with_no_fail_soft_paths_taken(self, monkeypatch, tmp_path):
        """The CI guarantee behind strict loading: no bundled op relies on a warn-and-drop path.

        Asserted with a dedicated dispatcher rather than by exporting IT_STRICT_OP_LOAD for the whole
        suite, because several hub tests deliberately exercise the fail-soft paths.
        """
        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        monkeypatch.setenv("IT_ANALYSIS_CACHE", str(tmp_path))  # force a compile rather than a cache hit
        dispatcher = AnalysisOpDispatcher(enable_hub_ops=False)
        dispatcher._cache_manager.cache_dir = tmp_path
        # Guard the guarantee: a cache hit would skip compilation entirely and prove nothing.
        assert dispatcher._cache_manager.load_cache() is None
        dispatcher.load_definitions()

        assert dispatcher._op_definitions, "no bundled definitions loaded"
        # Every bundled op must instantiate, which is where importable_params resolve.
        for op_name, op_def in list(dispatcher._op_definitions.items()):
            if op_def.name != op_name:
                continue  # alias entry
            dispatcher.get_op(op_name)


class TestNonBundledDeclarations:
    """Declaring `op_state`/traits must work from a NON-bundled YAML, demonstrated rather than inferred.

    Every other compilation test here drives bundled definitions or calls the compiler entry points directly. That
    infers hub/local declarability from a shared code path, which is the same shape of argument Phase 1's guide made
    about liftability and had to retract. These load a real local collection instead (umbrella review finding,
    2026-08-17).
    """

    OP_YAML = """
my_local_streaming_op:
  description: Local op that declares state and traits
  implementation: interpretune.analysis.ops.bundled.core.core_ops.model_fwd_impl
  requires_grad: true
  uses_default_hooks: true
  per_latent_preds: true
  op_state:
    scope: run
    reset_each_epoch: true
    fields: [running_total]
  input_schema: {}
  output_schema: {}
"""

    @staticmethod
    def _dispatcher_over(op_dir, cache_dir):
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        dispatcher = AnalysisOpDispatcher(yaml_paths=[op_dir], enable_hub_ops=False)
        dispatcher._cache_manager.cache_dir = cache_dir
        dispatcher.load_definitions()
        return dispatcher

    def _load(self, tmp_path, yaml_text: str):
        op_dir = tmp_path / "my_ops"
        op_dir.mkdir()
        (op_dir / "my_ops.yaml").write_text(yaml_text)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return self._dispatcher_over(op_dir, cache_dir)

    def test_local_yaml_can_declare_op_state_and_traits(self, tmp_path):
        dispatcher = self._load(tmp_path, self.OP_YAML)
        op_def = dispatcher._op_definitions["my_local_streaming_op"]
        assert op_def.op_state is not None
        assert op_def.op_state.fields == ("running_total",)
        assert op_def.op_state.reset_each_epoch is True
        assert (op_def.requires_grad, op_def.uses_default_hooks, op_def.per_latent_preds) == (True, True, True)

    def test_local_yaml_is_classified_local_not_bundled(self, tmp_path):
        dispatcher = self._load(tmp_path, self.OP_YAML)
        assert dispatcher._op_definitions["my_local_streaming_op"].source == "local"

    def test_declared_source_cannot_be_spoofed(self, tmp_path):
        """Provenance is computed from where the YAML lives, never read from the YAML.

        `{**value, "source": source}` puts the computed value last, so an author-declared `source:` is overwritten. That
        key ordering is load-bearing for the Phase 3 precedence work and is one reordering away from silent breakage, so
        it is pinned here.
        """
        spoofed = self.OP_YAML.replace("  description:", "  source: bundled\n  description:")
        dispatcher = self._load(tmp_path, spoofed)
        assert dispatcher._op_definitions["my_local_streaming_op"].source == "local"

    def test_declared_state_container_is_usable_for_a_local_op(self, tmp_path):
        from interpretune.config.analysis import AnalysisCfg

        dispatcher = self._load(tmp_path, self.OP_YAML)
        op = dispatcher.get_op("my_local_streaming_op")
        state = AnalysisCfg().op_state_for(op)
        assert state is not None
        state.set("running_total", 3)
        assert state.get("running_total") == 3
        with pytest.raises(KeyError, match="not a declared op_state field"):
            state.set("undeclared", 1)


class TestRequiresGradTraitDrivesTheAnalysisLoop:
    """The `requires_grad` trait's True branch had no end-to-end coverage.

    `logit_diffs_attr_grad` (the only op declaring it) is exercised by an attribute test but never run
    through the analysis loop, so the previous op-name check was untested here too. Pin the hook
    itself rather than leaving the trait covered only by declaration tests.
    """

    @staticmethod
    def _module_with(op):
        import torch

        from interpretune.base.components.mixins import AnalysisStepMixin
        from interpretune.config.analysis import AnalysisCfg

        cfg = AnalysisCfg(name="grad_probe")
        cfg.op = op

        class _Probe(AnalysisStepMixin):
            def __init__(self):
                self.it_cfg = SimpleNamespace(analysis_cfg=cfg)

        del torch
        return _Probe()

    @staticmethod
    def _op(name: str, **traits):
        from interpretune.analysis.ops.base import AnalysisOp, OpSchema

        return AnalysisOp(name=name, description="", output_schema=OpSchema({}), **traits)

    def test_grad_enabled_for_an_op_that_declares_it(self):
        import torch

        previous = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            self._module_with(self._op("wants_grad", requires_grad=True)).on_analysis_start()
            assert torch.is_grad_enabled()
        finally:
            torch.set_grad_enabled(previous)

    def test_grad_disabled_for_an_op_that_does_not(self):
        import torch

        previous = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(True)
            self._module_with(self._op("no_grad")).on_analysis_start()
            assert not torch.is_grad_enabled()
        finally:
            torch.set_grad_enabled(previous)

    def test_the_bundled_grad_composite_still_declares_it(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        assert DISPATCHER.get_op("logit_diffs_attr_grad").requires_grad is True
