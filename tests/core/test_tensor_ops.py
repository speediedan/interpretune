"""Ops, not modes (#952): an op is a name in venngram's registry, applied through an executor registered by name,
and a tensor op an op collection defines registers both when it loads, held to the tensor-op contract on a probe.

Each refusal is paired with the case that passes, so a refusal cannot pass because loading refused everything.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends import (
    Capability,
    InterventionSpec,
    InterventionSupport,
    PositionScope,
    apply_intervention,
    apply_op,
    get_module_capabilities,
    require_intervention_support,
)
from interpretune.analysis.backends import ops as op_registry
from interpretune.analysis.ops import dispatcher as dispatcher_module
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

CORE_OPS = {"replace", "add", "scale", "project", "reject", "patch", "clamp", "replace_with_error"}


@pytest.fixture
def scratch_op():
    """Register a throwaway op with venngram and an executor here, and remove both afterwards."""
    from venngram.ops import OPS, OpSpec, register_op

    registered: list[str] = []

    def _register(name: str, executor, operand=("vector",)) -> str:
        register_op(OpSpec(name, operand))
        op_registry.register_executor(name, executor)
        registered.append(name)
        return name

    yield _register
    for name in registered:
        OPS.pop(name, None)
        op_registry._EXECUTORS.pop(name, None)


class TestTheExecutorRegistry:
    def test_the_core_ops_are_registered_from_the_shared_op_math(self):
        assert CORE_OPS <= op_registry.registered_ops()
        assert all(op_registry.executor_record(name).source == "core" for name in CORE_OPS)

    def test_a_spec_dispatches_through_the_registry(self, scratch_op):
        """A user op registered beside the core ones runs through the same hook path, with no branch to edit."""
        scratch_op("double_then_add", lambda region, operand, spec: 2.0 * region + operand * spec.scale_factor)
        value = torch.ones(1, 3, 4)
        out = apply_intervention(
            value.clone(), InterventionSpec(torch.full((4,), 0.5), op="double_then_add", scale_factor=2.0), last_pos=2
        )
        assert torch.equal(out[:, 2], torch.full((1, 4), 3.0)) and torch.equal(out[:, :2], value[:, :2])

    def test_an_op_venngram_does_not_know_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"Unknown op 'steer'; venngram's registry holds"):
            op_registry.normalize_op("steer")
        with pytest.raises(ValueError, match="Unknown op 'steer'"):
            apply_intervention(torch.ones(1, 2, 4), InterventionSpec(torch.ones(4), op="steer"), last_pos=1)

    def test_an_op_without_an_executor_is_refused_when_applied(self):
        from venngram.ops import OPS, OpSpec, register_op

        register_op(OpSpec("declared_only", ("vector",)))
        try:
            assert "declared_only" not in op_registry.registered_ops()
            with pytest.raises(ValueError, match="op 'declared_only' has no executor in interpretune"):
                apply_op(
                    torch.ones(2, 4),
                    InterventionSpec(torch.ones(4), op="declared_only", position_scope="all_positions"),
                )
        finally:
            OPS.pop("declared_only")

    def test_a_second_executor_for_one_name_is_refused(self):
        with pytest.raises(ValueError, match=r"op 'add' already has an executor \(add \[core\]\)"):
            op_registry.register_executor("add", lambda region, operand, spec: region)

    def test_replace_with_error_has_the_value_of_the_activation_and_routes_the_gradient_to_the_operand(self):
        x = torch.randn(3, 4, requires_grad=True)
        o = torch.randn(4, requires_grad=True)
        out = apply_op(x, InterventionSpec(o, op="replace_with_error", position_scope="all_positions"))
        torch.testing.assert_close(out, x)
        gx, go = torch.autograd.grad(out.sum(), (x, o), allow_unused=True)
        assert gx is None and torch.equal(go, torch.full((4,), 3.0))


class TestBackendsDeclareEveryRegisteredOp:
    def test_a_hook_backend_declares_an_op_registered_after_it_was_built(self, scratch_op):
        from interpretune.adapters.transformer_lens.backends import TLModelBackend

        backend = TLModelBackend()
        record = backend.intervention_support
        assert record.every_registered_op and CORE_OPS <= record.declared_ops
        scratch_op("late_op", lambda region, operand, spec: region + operand)
        assert "late_op" in record.declared_ops
        require_intervention_support(
            backend, {"blocks.0.hook_resid_post": InterventionSpec(torch.ones(4), op="late_op")}, backend_name="tl"
        )

    def test_a_backend_listing_its_ops_refuses_the_rest_by_name(self, scratch_op):
        class _AddOnly:
            capabilities = frozenset({Capability.INTERVENTION})
            intervention_support = InterventionSupport(ops={"add"}, position_scopes={PositionScope.LAST_TOKEN})

        scratch_op("late_op", lambda region, operand, spec: region + operand)
        payload = {"blocks.0.hook_resid_post": InterventionSpec(torch.ones(4), op="late_op")}
        with pytest.raises(NotImplementedError, match=r"cannot apply op='late_op'; it declares \['add'\]"):
            require_intervention_support(_AddOnly(), payload, backend_name="add_only")

    def test_a_record_declares_a_list_or_every_registered_op_not_both(self):
        with pytest.raises(ValueError, match="either lists its ops or declares every registered op"):
            InterventionSupport(ops={"add"}, every_registered_op=True, position_scopes={"last_token"})
        with pytest.raises(ValueError, match="Unknown op 'steer'"):
            InterventionSupport(ops={"steer"}, position_scopes={"last_token"})

    def test_the_module_record_resolves_the_declared_ops(self):
        class _Module:
            def __init__(self):
                from interpretune.adapters.transformer_lens.backends import TLModelBackend

                self._model_backend = TLModelBackend()

        caps = get_module_capabilities(_Module())
        assert CORE_OPS <= caps.intervention_for("activation").declared_ops


class TestThePayloadSpellsOps:
    def test_a_mapping_payload_carrying_the_retired_mode_key_is_refused(self):
        from interpretune.analysis.backends import iter_intervention_axes

        with pytest.raises(ValueError, match="`mode` is now `op`"):
            list(
                iter_intervention_axes(
                    {"blocks.0.hook_resid_post": {"intervention_tensor": torch.ones(4), "mode": "add"}}
                )
            )
        ((_, spec),) = iter_intervention_axes(
            {"blocks.0.hook_resid_post": {"intervention_tensor": torch.ones(4), "op": "add"}}
        )
        assert spec.op == "add"


# ---------------------------------------------------------------------------------------------------------------------
# Tensor ops in an op collection
# ---------------------------------------------------------------------------------------------------------------------

TENSOR_OPS_YAML = """collection:
  name: tensor_op_collection
  version: 0.1.0

soft_add:
  description: add half the operand
  implementation: tensor_impl.soft_add
  op_spec:
    operand: [vector]
    composition: sum

needs_soft_add:
  description: a composition op requiring the tensor op beside a composition op
  implementation: tensor_impl.consumer
  required_ops: [soft_add, get_answer_indices]
  input_schema: {}
  output_schema: {}
"""

TENSOR_IMPL = '''"""Tensor-op implementations for the fixture collection."""


def soft_add(region, operand, spec):
    """Out of place, shape-preserving and differentiable in both."""
    return region + 0.5 * operand * spec.scale_factor


def in_place_add(region, operand, spec):
    """Writes the region: refused."""
    region.add_(operand)
    return region


def reduces(region, operand, spec):
    """Changes the shape: refused."""
    return (region + operand).sum(-1)


def detaches_the_operand(region, operand, spec):
    """Not differentiable in the operand: refused."""
    return region + operand.detach()


def consumer(module, analysis_batch, batch, batch_idx):
    """Stand in for a real op body."""
    return analysis_batch
'''


def _collection(tmp_path, monkeypatch, ops_yaml: str, repo_id: str = "tensorer/ops") -> AnalysisOpDispatcher:
    from tests.hub_op_fixtures import write_cached_op_collection

    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    hub_cache = tmp_path / "hub"
    write_cached_op_collection(
        hub_cache, repo_id=repo_id, op_files={"ops.yaml": ops_yaml, "tensor_impl.py": TENSOR_IMPL}
    )
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr(dispatcher_module, "IT_ANALYSIS_HUB_CACHE", hub_cache)
    monkeypatch.setattr("interpretune.analysis.IT_ANALYSIS_OP_PATHS", [])
    dispatcher = AnalysisOpDispatcher(enable_hub_ops=True)
    (tmp_path / "cache").mkdir()
    dispatcher._cache_manager.cache_dir = tmp_path / "cache"
    return dispatcher


@pytest.fixture
def forget_tensor_ops():
    """Remove what a test's collection registered, so registrations do not leak between tests."""
    from venngram.ops import OPS

    before = set(op_registry._EXECUTORS)
    yield
    for name in set(op_registry._EXECUTORS) - before:
        op_registry._EXECUTORS.pop(name)
        OPS.pop(name, None)


@pytest.mark.usefixtures("forget_tensor_ops")
class TestTensorOpsInACollection:
    def test_loading_registers_the_op_with_venngram_and_its_executor_with_provenance(self, tmp_path, monkeypatch):
        from venngram.ops import OPS

        _collection(tmp_path, monkeypatch, TENSOR_OPS_YAML).load_definitions()
        assert OPS["soft_add"].composition == "sum" and OPS["soft_add"].operand == ("vector",)
        record = op_registry.executor_record("soft_add")
        assert record.source == "hub:tensorer.ops" and record.collection == "tensor_op_collection"
        assert record.version == "0.1.0" and record.revision == "abc123"
        out = apply_op(
            torch.zeros(2, 4), InterventionSpec(torch.ones(4), op="soft_add", position_scope="all_positions")
        )
        assert torch.equal(out, torch.full((2, 4), 0.5))

    def test_a_tensor_op_is_not_a_composition_op(self, tmp_path, monkeypatch):
        dispatcher = _collection(tmp_path, monkeypatch, TENSOR_OPS_YAML)
        dispatcher.load_definitions()
        assert not any(name.split(".")[-1] == "soft_add" for name in dispatcher._op_definitions)

    def test_required_ops_may_name_a_tensor_op_beside_a_composition_op(self, tmp_path, monkeypatch):
        dispatcher = _collection(tmp_path, monkeypatch, TENSOR_OPS_YAML)
        dispatcher.load_definitions()
        op_def = dispatcher._op_definitions["tensorer.ops.needs_soft_add"]
        assert op_def.required_ops == ["get_answer_indices"] and op_def.required_tensor_ops == ["soft_add"]

    def test_an_op_requiring_a_tensor_op_is_refused_on_a_backend_that_does_not_declare_it(self, tmp_path, monkeypatch):
        from interpretune.analysis.ops.base import AnalysisBatch, AnalysisOp, OpSchema

        _collection(tmp_path, monkeypatch, TENSOR_OPS_YAML).load_definitions()

        class _Narrow:
            capabilities = frozenset({Capability.INTERVENTION})
            intervention_support = InterventionSupport(ops={"add"}, position_scopes={"last_token"})

        class _Module(torch.nn.Module):
            def __init__(self, backend):
                super().__init__()
                self._model_backend = backend

        op = AnalysisOp(name="probe", description="", output_schema=OpSchema({}), required_tensor_ops=["soft_add"])
        op._impl = lambda module, analysis_batch, batch, batch_idx, **kw: analysis_batch
        with pytest.raises(ValueError, match=r"requires intervention ops \['soft_add'\] .* missing \['soft_add'\]"):
            op(module=_Module(_Narrow()), analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)
        from interpretune.adapters.transformer_lens.backends import TLModelBackend

        assert op(module=_Module(TLModelBackend()), analysis_batch=AnalysisBatch(), batch=None, batch_idx=0) is not None

    @pytest.mark.parametrize(
        "implementation, clause",
        [
            pytest.param("in_place_add", "is not out of place: it wrote its region", id="in-place"),
            pytest.param("reduces", "is not shape-preserving", id="shape"),
            pytest.param("detaches_the_operand", "is not differentiable in the operand", id="differentiable"),
        ],
    )
    def test_an_implementation_breaking_the_contract_is_refused_on_the_probe(
        self, tmp_path, monkeypatch, implementation, clause
    ):
        from venngram.ops import OPS

        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError

        broken = TENSOR_OPS_YAML.replace("tensor_impl.soft_add", f"tensor_impl.{implementation}")
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        with pytest.raises(OpLoadError, match=f"tensor op 'soft_add' {clause}"):
            _collection(tmp_path, monkeypatch, broken).load_definitions()
        assert "soft_add" not in OPS and "soft_add" not in op_registry._EXECUTORS, "nothing registers on a refusal"

    def test_a_malformed_op_spec_is_refused_by_name(self, tmp_path, monkeypatch):
        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError

        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        malformed = TENSOR_OPS_YAML.replace("composition: sum", "compose: sum")
        with pytest.raises(OpLoadError, match=r"unknown keys \['compose'\]"):
            _collection(tmp_path, monkeypatch, malformed).load_definitions()

    def test_a_name_another_source_holds_is_refused(self, tmp_path, monkeypatch):
        from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError

        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        shadowing = TENSOR_OPS_YAML.replace("soft_add:\n  description", "add:\n  description")
        with pytest.raises(OpLoadError, match=r"tensor op 'add' is already registered \(add \[core\]\); rename it"):
            _collection(tmp_path, monkeypatch, shadowing).load_definitions()

    def test_reloading_the_same_collection_replaces_its_own_registration(self, tmp_path, monkeypatch):
        dispatcher = _collection(tmp_path, monkeypatch, TENSOR_OPS_YAML)
        dispatcher.load_definitions()
        dispatcher.reload_definitions()
        assert op_registry.executor_record("soft_add").source == "hub:tensorer.ops"

    def test_without_the_trust_opt_in_no_tensor_op_loads(self, tmp_path, monkeypatch):
        dispatcher = _collection(tmp_path, monkeypatch, TENSOR_OPS_YAML)
        monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "0")
        dispatcher.load_definitions()
        assert "soft_add" not in op_registry._EXECUTORS


class TestOpSpecFromData:
    def test_every_field_is_read(self):
        spec = op_registry.op_spec_from_mapping(
            "bounded",
            {
                "operand": ["vector", "subspace"],
                "params": {"fraction": {"type": "float", "range": [0.0, 1.0]}, "label": {"type": "str"}},
                "requires": ["basis"],
                "targets": ["activation"],
                "value": "none",
                "token_override": True,
                "composition": "ordered",
                "at_least_one_of": ["fraction"],
            },
        )
        assert spec.operand == ("vector", "subspace") and spec.requires == ("basis",) and spec.value == "none"
        assert spec.params["fraction"].range == (0.0, 1.0) and spec.params["label"].type is str
        assert spec.token_override and spec.at_least_one_of == ("fraction",)

    def test_of_is_read_and_a_name_is_refused(self):
        assert op_registry.op_spec_from_mapping("g", {"of": ["grad"]}).of == ("grad",)
        with pytest.raises(ValueError, match="takes its name from the op definition"):
            op_registry.op_spec_from_mapping("g", {"name": "other"})

    def test_an_unknown_parameter_type_is_refused(self):
        with pytest.raises(ValueError, match="parameter `k` has type 'complex'"):
            op_registry.op_spec_from_mapping("x", {"params": {"k": {"type": "complex"}}})
