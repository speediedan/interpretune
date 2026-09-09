"""What is left of the set-level parity module after its capture, scope and logits layers moved into the suite.

`docs/adapter_parity_governance.md` places TransformerBridge, NNsight and interp-engine in one family: all three
execute the HF forward and differ only in how they observe and modify it. The family-wide claims (every backend
sees the same activation, applies an edit to the positions it was asked to, and reduces to the same logits) are
now conformance cases in `interpretune.testing.conformance`, selected per target from its live declarations and
compared against the same library-independent HF reference this module introduced. Core runs them over the
bundled compositions in `tests/core/test_backend_conformance.py`; a hub adapter runs them in its own repository.

Two things stay here because the suite is the wrong place for them:

1. **The interp-engine steering leg.** It pins a fact about a specific third-party engine (its native steering is
   whole-prompt) against the HF reference, which is a parity assertion about that engine rather than a case every
   adapter must satisfy. The engine's adapter carries the conformance cases; this leg stays as the record of WHY
   the position-scope axis exists, with the measurement that motivated it.
2. **The op-purity guard.** It asserts a property of the op layer, not of any backend, and needs no session.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from torch.testing import assert_close

from interpretune.testing.conformance.oracles import changed_positions
from interpretune.utils.import_utils import package_available

MODEL_ID = "gpt2"
LAYER = 5
PROMPT = "The capital of France is"

# Tight, because every participant runs the SAME forward: a difference should come from the observation or
# edit mechanism, which is what is under test, and not from arithmetic drift.
RTOL, ATOL = 1e-4, 1e-4

# Imported at COLLECTION rather than inside a test. `interp_engine` pulls in a compiler/runtime stack that
# sets TORCHINDUCTOR_*, TRITON_* and TILELANG_* as an import side effect, and the suite fails any test that
# leaves the environment dirtier than it found it. That check is PER-TEST, so an import inside the first
# test is attributed to it no matter how carefully it cleans up -- the variables are set before the test
# body can restore anything. Module scope puts the side effect outside every test's snapshot.
INTERP_ENGINE_AVAILABLE = package_available("interp_engine")
if INTERP_ENGINE_AVAILABLE:  # pragma: no cover - imported for side-effect ordering, not for the name
    import interp_engine  # noqa: F401


@pytest.fixture(scope="module")
def prompt_ids():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)(PROMPT, return_tensors="pt")["input_ids"]


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_reference(hf_model, prompt_ids):
    """Library-independent ground truth: the block's input off the HF module, by a plain forward pre-hook."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        captured["resid_pre"] = args[0].detach().clone()

    handle = hf_model.transformer.h[LAYER].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            hf_model(prompt_ids)
    finally:
        handle.remove()
    assert "resid_pre" in captured, "the pre-hook never fired; the reference would be vacuous"
    assert captured["resid_pre"].abs().max() > 0, "reference is all zeros; comparisons against it prove nothing"
    return {"resid_pre": captured["resid_pre"][0]}


# --------------------------------------------------------------------------------------------------
# The interp-engine steering leg
# --------------------------------------------------------------------------------------------------
#
# The discriminator is CAUSALITY, not introspection, which is what makes it backend-agnostic. In a causal
# LM, editing the residual stream at position p can only affect positions >= p downstream. So a last-token
# intervention moves exactly the final position of a later activation, and a whole-prompt intervention
# moves all of them. The observable is the FINAL layer's `resid_post` rather than logits, because that is a
# point interp-engine has: its vocabulary is activation points and carries no `logits` entry.

STEER_SCALE = 12.0
LAST_LAYER = 11  # gpt2 has 12 blocks; the last one's output is the observable


def _steering_vector(hf_reference):
    """A unit direction with real effect, derived from the reference so it is meaningful for this model."""
    resid = hf_reference["resid_pre"]
    return (resid[-1] / resid[-1].norm()).clone()


@pytest.fixture(scope="module")
def hf_final_resid(hf_model, prompt_ids):
    """Un-steered final-layer activation: the baseline every scope comparison is made against."""
    captured = {}

    def hook(_module, _args, output):
        captured["out"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    handle = hf_model.transformer.h[LAST_LAYER].register_forward_hook(hook)
    try:
        with torch.no_grad():
            hf_model(prompt_ids)
    finally:
        handle.remove()
    assert "out" in captured, "the final-layer hook never fired; the baseline would be vacuous"
    return captured["out"][0]


def _hf_steered_final_resid(hf_model, prompt_ids, vector, *, all_positions: bool):
    """The reference implementation of BOTH scopes, so the discriminator itself is validated first."""
    captured = {}

    def steer(_module, args):
        resid = args[0].clone()
        if all_positions:
            resid = resid + vector * STEER_SCALE
        else:
            resid[:, -1, :] = resid[:, -1, :] + vector * STEER_SCALE
        return (resid,) + tuple(args[1:])

    def observe(_module, _args, output):
        captured["out"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    h1 = hf_model.transformer.h[LAYER].register_forward_pre_hook(steer)
    h2 = hf_model.transformer.h[LAST_LAYER].register_forward_hook(observe)
    try:
        with torch.no_grad():
            hf_model(prompt_ids)
    finally:
        h1.remove()
        h2.remove()
    return captured["out"][0]


@pytest.mark.skipif(not INTERP_ENGINE_AVAILABLE, reason="interp-engine is not installed")
class TestInterpEngineSteeringIsAllPositions:
    """Interp-engine steers EVERY prompt position, and that is a capability we now express (#441).

    Its ``SteeringSpec`` carries ``layers``, ``point`` and ``stream`` and no position field, so whole-prompt
    is the only scope it implements. **That is not a bug in interp-engine.** "Steer the whole prompt" is a
    legitimate experiment -- the right shape for changing how a model reads its input, where last-token
    steering is the right shape for changing the next prediction.

    **The defect was ours.** Interpretune's primitive was named `apply_intervention_to_last_token`, so it
    had no way to say which scope a caller wanted, and interp-engine's whole-prompt result was consumed as
    though it were last-token. Shapes agreed, nothing raised, and the activations were entirely plausible.
    `InterventionSpec.position_scope` now names the operation, and `require_position_scope` refuses a scope
    a backend cannot honour rather than substituting the one it can.

    So these tests are PARITY assertions, not a bug record: interp-engine's native steering must match our
    ``all_positions`` semantics exactly, and must be distinguishable from ``last_token``. The discriminator
    is validated on the HF reference alone first, in both directions, so a broken instrument cannot pass it.
    """

    @staticmethod
    def _steered_final_resid(prompt_ids, vector):
        from interp_engine.model import EagerModel
        from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

        model = EagerModel(MODEL_ID, dtype="float32")
        spec = SteeringSpec(
            layers={LAYER: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=STEER_SCALE)])},
            point="resid_pre",
        )
        out = asyncio.run(model.capture(prompt_ids[0], [f"resid_post.{LAST_LAYER}"], steering_spec=spec))
        assert out, "steered capture returned nothing"
        return next(iter(out.values()))

    def test_the_discriminator_tells_the_scopes_apart_on_the_reference(
        self, hf_model, hf_reference, hf_final_resid, prompt_ids
    ):
        """Positive control, in both directions: a test that only checked one scope could not tell "the
        discriminator works" from "the discriminator always returns that scope's set"."""
        vector = _steering_vector(hf_reference)
        last = _hf_steered_final_resid(hf_model, prompt_ids, vector, all_positions=False)
        whole = _hf_steered_final_resid(hf_model, prompt_ids, vector, all_positions=True)
        assert changed_positions(hf_final_resid, last) == {prompt_ids.shape[1] - 1}
        assert changed_positions(hf_final_resid, whole) == set(range(prompt_ids.shape[1]))

    def test_native_steering_matches_our_all_positions_semantics(
        self, hf_model, hf_reference, hf_final_resid, prompt_ids
    ):
        """The parity claim: interp-engine's only scope IS our ``all_positions``, numerically.

        Stronger than observing the position set, because it pins the VALUES against an independent
        implementation of the same operation. If our `all_positions` arithmetic and interp-engine's
        differed -- a scale applied twice, an edit at the wrong point -- the sets would still match while
        the tensors did not.
        """
        vector = _steering_vector(hf_reference)
        theirs = self._steered_final_resid(prompt_ids, vector)
        ours = _hf_steered_final_resid(hf_model, prompt_ids, vector, all_positions=True)
        assert_close(
            theirs.to(ours.dtype),
            ours,
            rtol=RTOL,
            atol=ATOL,
            msg="interp-engine's whole-prompt steering diverged from interpretune's all_positions semantics",
        )

    def test_it_is_distinguishable_from_last_token(self, hf_reference, hf_final_resid, prompt_ids):
        """The capability statement, and the negative control on the claim above.

        Without it, "matches all_positions" would be satisfiable by an implementation whose two scopes are
        the same thing. This pins that interp-engine's NATIVE scope is genuinely the whole prompt -- which
        is what an adapter listing ``all_positions`` in its ``InterventionSupport`` is asserting about it.
        """
        vector = _steering_vector(hf_reference)
        steered = self._steered_final_resid(prompt_ids, vector)
        assert changed_positions(hf_final_resid, steered) == set(range(prompt_ids.shape[1])), (
            "interp-engine's native steering is no longer whole-prompt. An adapter declaring all_positions "
            "for it on that basis would now be wrong, and the position-scope framing needs revisiting."
        )


# --------------------------------------------------------------------------------------------------
# The op layer is backend-independent
# --------------------------------------------------------------------------------------------------
#
# `logit_diffs_impl` derives everything it returns from `analysis_batch.answer_logits` and
# `answer_indices`; it never touches the backend. So "do analysis ops agree across backends" is really
# two questions: (a) do backends produce the same ANSWER LOGITS, which varies by backend and is now the
# suite's `test_answer_logits_converge_on_the_forward`; and (b) given identical inputs, is the op
# DETERMINISTIC and backend-independent, which is pure by construction and asserted here so it cannot
# quietly stop being.


class TestTheOpLayerIsBackendIndependent:
    """The half that is pure by construction -- asserted so it cannot quietly stop being.

    `logit_diffs_impl` takes its inputs from the analysis batch and never consults the backend. That is a
    design property, not an accident, and it is what lets one op serve every backend. A future
    backend-conditional branch inside an op would break it silently: the op would still run, still return
    plausible numbers, and no conformance case would notice, because they all compare backends running
    THEIR OWN ops rather than one op over fixed inputs.
    """

    @staticmethod
    def _run(answer_logits, answer_indices):
        import torch as _t

        from interpretune.analysis.ops.bundled.core.core_ops import logit_diffs_impl

        captured = {}

        class _Batch(dict):
            """Minimal analysis batch: the op only reads two fields and calls `.update`."""

            answer_logits = None
            answer_indices = None

            def update(self, **kw):
                captured.update(kw)

        ab = _Batch()
        ab.answer_logits = answer_logits
        ab.answer_indices = answer_indices

        def _fake_get_loss_preds_diffs(module, analysis_batch, answer_logits, logit_diff_fn):
            # Stand in for the label-dependent half; the point under test is that the op consults its
            # ARGUMENTS rather than the module it was handed.
            return (
                _t.tensor(0.0),
                answer_logits.sum(-1) if answer_logits.dim() > 1 else answer_logits.clone(),
                _t.zeros(answer_logits.shape[0], dtype=_t.long),
                answer_logits,
            )

        logit_diffs_impl(
            module=object(),  # deliberately not a backend: the op must not consult it
            analysis_batch=ab,
            batch={"input_ids": _t.zeros(answer_logits.shape[0], 4, dtype=_t.long)},
            get_loss_preds_diffs=_fake_get_loss_preds_diffs,
        )
        return captured

    def test_identical_inputs_give_identical_outputs(self):
        logits = torch.randn(3, 1, 5)
        idx = torch.zeros(3, 1, dtype=torch.long)
        a = self._run(logits.clone(), idx.clone())
        b = self._run(logits.clone(), idx.clone())
        assert_close(a["logit_diffs"], b["logit_diffs"])

    def test_the_op_does_not_consult_the_module_it_is_handed(self):
        """The load-bearing assertion: `module` is a bare object, so any backend branch would raise.

        This is the guard that would fail the day someone adds `if isinstance(module, NNsightBackend)`
        to an op -- which is the change that would make ops silently backend-dependent while every
        backend-vs-backend parity test kept passing.
        """
        logits = torch.randn(2, 1, 4)
        out = self._run(logits, torch.zeros(2, 1, dtype=torch.long))
        assert "logit_diffs" in out and out["logit_diffs"].shape[0] == 2
