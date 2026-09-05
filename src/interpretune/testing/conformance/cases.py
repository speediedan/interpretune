"""The model-backend conformance cases. A repository subclasses ``ModelBackendConformance`` and sets ``target``.

Every case goes through the runner with inputs constructed the way a caller constructs them, and asserts the declared
return type before reading a value. A case that must reach the backend directly says so in its docstring and is the
exception.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import torch

from interpretune.analysis.backends import (
    BackendCapability,
    InterventionMode,
    PositionScope,
    SupportsGradients,
    SupportsIntervention,
    SupportsLatentModels,
)
from interpretune.analysis.optools import require_backend_capability

from .gates import UNDECLARED, conformance_case, gate_of
from .plugin import _REPORT_KEY
from .inputs import ConformanceInputs, ConformanceTarget
from .oracles import (
    expect_refusal,
    CONVERGENCE_ATOL,
    STEER_SCALE,
    assert_non_degenerate,
    changed_positions,
    expected_positions,
    steering_vector,
)
from .ops import captured_points
from .reference import HFReference
from .session import ConformanceSession, build_conformance_session, tokenized_prompts

#: Tolerances for the real positions of a left-padded dataset batch, looser than the unpadded calibration
#: (1e-4) and RELATIVE as well as absolute. Measured: on Linux the bridge and a plain HF forward agree to 4.6e-4
#: absolute on gpt2's logits; on the Windows CPU runner the deepest points (the unembed input after twelve
#: blocks, and the logits) exceeded 1e-3 absolute while the shallow points passed, which is accumulated
#: platform drift rather than a wrong tensor. Logits sit at magnitude ~100, where 1e-3 absolute is 1e-5
#: relative. Pad positions are undefined and never compared.
PADDED_RTOL = 1e-3
PADDED_ATOL = 1e-3

_PROTOCOL_FOR = {
    BackendCapability.LATENT_MODELS: SupportsLatentModels,
    BackendCapability.GRADIENTS: SupportsGradients,
    BackendCapability.INTERVENTION: SupportsIntervention,
}


def _require_discriminating_length(attention_mask: torch.Tensor | None, seq_len: int, *, what: str) -> None:
    """The two scopes are indistinguishable below two real positions (both edit `{0}`), so a short input would pass
    a backend that ignores scope entirely.

    The discriminating power lives in the fixture, so the fixture is asserted before any set is compared. Measured on
    gpt2: n=1 gives `{0}` for both scopes.
    """
    real = seq_len if attention_mask is None else int(attention_mask.sum(dim=-1).min())
    assert real >= 2, (
        f"{what}: only {real} real position(s); the position scopes are indistinguishable below 2, so the exact-set "
        "cases would pass a backend that ignores scope entirely"
    )


def _assert_close_padded(got: torch.Tensor, ref: torch.Tensor, *, what: str) -> None:
    """Padded-batch convergence, with the measured differences IN the message so a platform failure reports
    numbers."""
    got = got.to(torch.float32)
    ref = ref.to(torch.float32)
    diff = (got - ref).abs()
    ok = diff <= PADDED_ATOL + PADDED_RTOL * ref.abs()
    if bool(ok.all()):
        return
    rel = (diff / ref.abs().clamp_min(1e-12)).max().item()
    raise AssertionError(
        f"{what} diverged from the HF forward (rtol={PADDED_RTOL}, atol={PADDED_ATOL}): greatest absolute difference "
        f"{diff.max().item():.3e}, greatest relative difference {rel:.3e}, {int((~ok).sum())} of {ok.numel()} elements"
    )


def _real_effect(effect: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero the effect at pad positions: they are undefined by contract, and a padded forward may not even be
    deterministic there, so a discriminator that counted them would flag noise as movement."""
    if attention_mask is None:
        return effect
    return effect * attention_mask.to(effect.dtype)


def _real_positions(attention_mask: torch.Tensor | None, tensor: torch.Tensor) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(tensor.shape[:2], dtype=torch.bool)
    return attention_mask.bool()


class ModelBackendConformance:
    """Subclass, set ``target``, and pytest does the rest.

    Fixtures are class-scoped: one composed session and one HF reference per target class, and every case
    reads an extracted store rather than holding the live session.
    """

    target: ClassVar[ConformanceTarget]
    inputs: ClassVar[ConformanceInputs | None] = None

    # -- fixtures ----------------------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def suite(self, request) -> ConformanceSession:
        """One composed session and runner per target class."""
        cls = request.cls
        inputs = cls.inputs or ConformanceInputs()
        return build_conformance_session(cls.target, inputs)

    @pytest.fixture(scope="class")
    def hf(self, suite) -> HFReference:
        """The library-independent HF reference, loaded once per class."""
        return HFReference(suite.inputs.model_id)

    @pytest.fixture(scope="class")
    def prompt_ids(self, hf, suite) -> list[torch.Tensor]:
        """The suite's prompts as unpadded id tensors."""
        return tokenized_prompts(hf.tokenizer, suite.inputs.prompts)

    @pytest.fixture(autouse=True)
    def _gate(self, request, suite):
        """Skip a case whose gate the target has not declared, with the reason the report keys on."""
        gate = gate_of(request.function)
        if gate is None:
            return
        report = request.config.stash.get(_REPORT_KEY, None)
        if report is not None and not report.declared:
            report.declared = sorted(c.name for c in suite.capabilities.model) + sorted(
                c.name for c in suite.capabilities.analysis
            )
        if not gate.selects(suite.capabilities, family=suite.family, single_prompt=suite.target.single_prompt):
            pytest.skip(f"{UNDECLARED}: needs {gate.describe()}")

    # -- always-on ---------------------------------------------------------------------------------

    @conformance_case()
    def test_session_composes_and_declarations_are_coherent(self, suite):
        """A backend is attached, each declared surface satisfies its protocol, and records match declarations."""
        backend = suite.backend
        assert backend is not None, "the composed module attaches no model backend"
        assert isinstance(backend.capabilities, frozenset)
        for cap in backend.capabilities:
            assert isinstance(cap, BackendCapability), f"{cap!r} is not a BackendCapability member"
            protocol = _PROTOCOL_FOR[cap]
            assert isinstance(backend, protocol), (
                f"{suite.backend_name} declares {cap.name} but is not a {protocol.__name__}"
            )
        # the present-iff-declared invariant, from the live objects rather than the enum
        assert (suite.capabilities.intervention is not None) == (BackendCapability.INTERVENTION in backend.capabilities)
        assert (suite.capabilities.latent_models is not None) == (
            BackendCapability.LATENT_MODELS in backend.capabilities
        )

    @conformance_case()
    def test_undeclared_capabilities_are_refused_by_name(self, suite):
        """Every surface the backend does NOT claim must be refused by the shared gate, naming the backend and what
        it does claim.

        Checked at the gate rather than through an op, because the ops that need LATENT_MODELS and GRADIENTS also need
        latent models configured, which would fail first.
        """
        backend = suite.backend
        undeclared = [c for c in BackendCapability if c not in backend.capabilities]
        # A backend declaring every surface has nothing to refuse; that is a pass, not a skip, because the
        # claim "every surface is declared" is itself checked above and a skip here would count as "other".
        for cap in undeclared:
            with pytest.raises(ValueError, match=f"requires a model backend with {cap.name}"):
                require_backend_capability(backend, cap, "conformance")

    @conformance_case()
    def test_runner_produces_the_store_schema(self, suite):
        """`logit_diffs_base` yields the declared columns, one row per batch, with the declared dtypes."""
        import interpretune as it
        from interpretune import AnalysisCfg

        store = suite.run(AnalysisCfg(target_op=it.logit_diffs_base, save_tokens=True))
        n = suite.inputs.limit_batches
        for column in ("logit_diffs", "answer_logits", "preds", "orig_labels"):
            values = getattr(store, column)
            assert values is not None, f"{column} is absent from the store"
            assert len(values) == n, f"{column}: {len(values)} rows for {n} batches"
        for al in store.answer_logits:
            assert isinstance(al, torch.Tensor) and al.dtype == torch.float32, "answer_logits must be float32 tensors"
        for ld in store.logit_diffs:
            assert isinstance(ld, torch.Tensor), "logit_diffs must be tensors"

    @conformance_case()
    def test_cache_op_stores_logits_and_every_requested_point(self, suite):
        """The cache path returns logits and every requested point reaches the store as a tensor."""
        from interpretune import AnalysisCfg

        points = list(suite.inputs.capture_points)
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=points, save_tokens=True))
        columns = list(store.dataset.column_names)
        for column in ("captured_values", "captured_shape", "captured_point_names"):
            assert column in columns, f"{column} is absent from the store; columns present: {columns}"
        assert store.answer_logits is not None and all(t is not None for t in store.answer_logits), (
            "answer_logits is None from the cache path; the backend's fwd_w_cache returned no logits"
        )
        for i in range(len(store["captured_values"])):
            captured = captured_points(store, i)
            missing = [p for p in points if p not in captured]
            assert not missing, f"cache lacks requested points {missing}"
            for p in points:
                assert isinstance(captured[p], torch.Tensor), f"{p} is not a tensor in the cache"

    @conformance_case()
    def test_block_output_is_the_next_block_input(self, suite):
        """`blocks.L.hook_out` and `blocks.L+1.hook_in` are the SAME residual tensor.

        Proves the layer index and the in/out slot mean what they say, with no reference at all.
        """
        from interpretune import AnalysisCfg

        layer = suite.inputs.capture_layer
        a, b = f"blocks.{layer}.hook_out", f"blocks.{layer + 1}.hook_in"
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=[a, b, f"blocks.{layer}.hook_in"]))
        captured = captured_points(store, 0)
        torch.testing.assert_close(captured[a], captured[b], rtol=0, atol=0)
        assert not torch.allclose(captured[a], captured[f"blocks.{layer}.hook_in"]), "a block changed nothing"

    @conformance_case(family="hf_native")
    def test_capture_converges_on_the_forward(self, suite, hf):
        """Every captured point matches the HF module's tensor on the real positions of each batch."""
        from interpretune import AnalysisCfg

        points = list(suite.inputs.capture_points)
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=points, save_tokens=True))
        for i in range(len(store["captured_values"])):
            captured = captured_points(store, i)
            ids, mask = suite.batch_inputs(i)
            ref = hf.capture(ids, points, attention_mask=mask)
            real = _real_positions(mask, ref[points[0]])
            for p in points:
                assert_non_degenerate(captured[p], what=p)
                _assert_close_padded(captured[p][real], ref[p][real], what=f"{p} on the real positions of batch {i}")

    @conformance_case(family="hf_native")
    def test_answer_logits_converge_on_the_forward(self, suite, hf):
        """The store's logits match the HF forward on real positions."""
        from interpretune import AnalysisCfg

        store = suite.run(
            AnalysisCfg(
                target_op="store_capture_points", names_filter=[suite.inputs.capture_points[0]], save_tokens=True
            )
        )
        for i, al in enumerate(store.answer_logits):
            ids, mask = suite.batch_inputs(i)
            ref = hf.logits(ids, attention_mask=mask)
            real = _real_positions(mask, ref)
            _assert_close_padded(al[real], ref[real], what=f"logits on the real positions of batch {i}")

    # -- INTERVENTION ------------------------------------------------------------------------------

    def _intervene(self, suite, *, scope: str, mode: str = "add", scale: float = STEER_SCALE, vector=None):
        """The caller's path: a raw payload in run_inputs, through model_fwd_intervention."""
        import interpretune as it
        from interpretune import AnalysisCfg

        if vector is None:
            vector = self._vector(suite)
        payload = {
            suite.inputs.intervention_point: {
                "intervention_tensor": vector,
                "mode": mode,
                "scale_factor": scale,
                "position_scope": scope,
            }
        }
        return suite.run(
            AnalysisCfg(target_op=it.model_fwd_intervention, run_inputs={"interventions": payload}, save_tokens=True)
        )

    def _vector(self, suite) -> torch.Tensor:
        from interpretune import AnalysisCfg

        point = suite.inputs.intervention_point
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=[point]))
        return steering_vector(captured_points(store, 0)[point])

    @conformance_case(
        capability=BackendCapability.INTERVENTION, scope=PositionScope.LAST_TOKEN, mode=InterventionMode.ADD
    )
    def test_last_token_scope_moves_exactly_the_final_position(self, suite):
        """The changed-position SET under `last_token` is exactly the final position."""
        store = self._intervene(suite, scope="last_token")
        for i, effect in enumerate(store["intervention_position_effect"]):
            _ids, mask = suite.batch_inputs(i)
            _require_discriminating_length(mask, effect.shape[-1], what=f"batch {i}")
            effect = _real_effect(effect, mask)
            seq = effect.shape[-1]
            got = changed_positions(torch.zeros_like(effect).unsqueeze(-1), effect.unsqueeze(-1))
            assert got == expected_positions("last_token", seq), f"batch {i}: changed {sorted(got)}"

    @conformance_case(
        capability=BackendCapability.INTERVENTION, scope=PositionScope.ALL_POSITIONS, mode=InterventionMode.ADD
    )
    def test_all_positions_scope_moves_every_real_position(self, suite):
        """Every real position is in the changed set under `all_positions`."""
        store = self._intervene(suite, scope="all_positions")
        for i, effect in enumerate(store["intervention_position_effect"]):
            _ids, mask = suite.batch_inputs(i)
            _require_discriminating_length(mask, effect.shape[-1], what=f"batch {i}")
            effect = _real_effect(effect, mask)
            got = changed_positions(torch.zeros_like(effect).unsqueeze(-1), effect.unsqueeze(-1))
            real = _real_positions(mask, effect.unsqueeze(-1))
            expected = {int(p) for p in torch.nonzero(real.any(dim=0)).flatten().tolist()}
            assert got == expected, f"batch {i}: changed {sorted(got)} but the real positions are {sorted(expected)}"

    @conformance_case(capability=BackendCapability.INTERVENTION, scope=PositionScope.ALL_POSITIONS, negative=True)
    def test_undeclared_all_positions_is_refused(self, suite):
        """A scope the backend did not declare is refused by name, never narrowed."""
        with expect_refusal(NotImplementedError, match="position_scope='all_positions'"):
            self._intervene(suite, scope="all_positions")

    @conformance_case(capability=BackendCapability.INTERVENTION, scope=PositionScope.LAST_TOKEN, negative=True)
    def test_undeclared_last_token_is_refused(self, suite):
        """A scope the backend did not declare is refused by name, never widened."""
        with expect_refusal(NotImplementedError, match="position_scope='last_token'"):
            self._intervene(suite, scope="last_token")

    @conformance_case(capability=BackendCapability.INTERVENTION)
    def test_undeclared_modes_are_refused_on_the_mode_axis(self, suite):
        """Each undeclared mode is refused naming the mode axis, never applied as another mode."""
        declared = {m.value for m in suite.capabilities.intervention.modes}
        undeclared = [m for m in InterventionMode if m.value not in declared]
        # Every mode declared means nothing to refuse: a pass, since the declaration itself is checked elsewhere.
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        for mode in undeclared:
            with expect_refusal(NotImplementedError, match=f"mode='{mode.value}'"):
                self._intervene(suite, scope=scope, mode=mode.value)

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.ADD)
    def test_zero_intervention_is_identity(self, suite):
        """A zero-scale intervention leaves the logits unchanged."""
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        store = self._intervene(suite, scope=scope, scale=0.0)
        for pre, post in zip(store["pre_intervention_logits"], store["post_intervention_logits"]):
            torch.testing.assert_close(
                post, pre, rtol=0, atol=CONVERGENCE_ATOL, msg="a zero-scale intervention changed the logits"
            )

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.ADD, family="hf_native")
    def test_baseline_is_an_unsteered_forward(self, suite, hf):
        """The pre-intervention half equals the plain forward."""
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        store = self._intervene(suite, scope=scope)
        for i, pre in enumerate(store["pre_intervention_logits"]):
            ids, mask = suite.batch_inputs(i)
            # the op reports the FIRST ROW's last-token logits (`last_token_logits` takes `logits[0, -1]`)
            ref = hf.logits(ids, attention_mask=mask)[0, -1, :]
            _assert_close_padded(pre, ref, what=f"batch {i}: the baseline half against the plain forward")

    @conformance_case(
        capability=BackendCapability.INTERVENTION,
        scope=PositionScope.LAST_TOKEN,
        mode=InterventionMode.ADD,
        family="hf_native",
    )
    def test_steered_logits_converge_on_the_forward(self, suite, hf):
        """Adding a vector at the last token matches an HF hook doing the same."""
        vector = self._vector(suite)
        store = self._intervene(suite, scope="last_token", vector=vector)
        for i, post in enumerate(store["post_intervention_logits"]):
            ids, mask = suite.batch_inputs(i)
            ref = hf.steered(
                ids,
                suite.inputs.intervention_point,
                lambda t: t + vector * STEER_SCALE,
                scope="last_token",
                attention_mask=mask,
            )
            _assert_close_padded(
                post,
                ref["logits"][0, -1, :],
                what=f"batch {i}: steered last-token logits against the HF reference edit",
            )

    # -- single-prompt backends ----------------------------------------------------------------------

    @conformance_case(single_prompt=True)
    def test_a_batch_above_the_declared_limit_is_refused_by_name(self, suite):
        """A target that declared one prompt at a time must REFUSE a larger batch, naming the limit.

        Reaches the backend directly, by design: the runner never builds a batch above the declared size for this
        target, so the refusal is only observable by asking. Silently processing row 0, or looping and re-padding
        without saying so, are the substitutions this case exists to catch.
        """
        ids, mask = suite.batch_inputs(0)
        assert ids.shape[0] == 1, f"a single-prompt target must run batches of one row, got {ids.shape[0]}"
        batch: dict[str, Any] = {"input": ids.repeat(2, 1)}
        if mask is not None:
            batch["attention_mask"] = mask.repeat(2, 1)
        with pytest.raises(Exception, match=r"(?i)one prompt|batch|single"):
            suite.backend.fwd(model=suite.module.model, batch=batch)

    # -- calibration: the discriminator, on the reference alone ------------------------------------

    @conformance_case(family="hf_native")
    def test_the_scope_discriminator_tells_the_scopes_apart(self, suite, hf, prompt_ids):
        """Positive control.

        Without it a scope case could pass because the instrument always says yes.
        """
        ids = prompt_ids[0]
        _require_discriminating_length(None, ids.shape[1], what="the calibration prompt")
        base = hf.capture(ids, [suite.inputs.observe_point])[suite.inputs.observe_point]
        vec = steering_vector(hf.capture(ids, [suite.inputs.intervention_point])[suite.inputs.intervention_point])
        for scope in ("last_token", "all_positions"):
            out = hf.steered(
                ids,
                suite.inputs.intervention_point,
                lambda t: t + vec * STEER_SCALE,
                scope=scope,
                observe=[suite.inputs.observe_point],
            )
            assert changed_positions(base, out[suite.inputs.observe_point]) == expected_positions(scope, ids.shape[1])
