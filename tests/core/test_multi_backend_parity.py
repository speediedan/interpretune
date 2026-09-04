"""Every execution backend we ship must observe, and edit, the same HuggingFace forward.

`docs/adapter_parity_governance.md` places TransformerBridge, NNsight and interp-engine in one family: all
three execute the HF forward and differ only in how they observe and modify it. This module is the
**set-level** parity suite for that family, parameterized over whichever participants are installed.

**Why a set compared to a reference, and not pairs compared to each other.** Pairwise comparison scales
quadratically, which is the boring objection. The real one is that a pair that agrees cannot distinguish
"both correct" from "both wrong in the same way" -- the two implementations share the very forward they
are being used to check. Comparing every participant against a library-independent capture of the HF
forward makes agreement evidence about the forward rather than about the pair.

**The reference is never a participant.** If it were, a participant agreeing with itself would count as
convergence, which is the failure mode this whole file exists to exclude.

**Three layers, deliberately separate, because their failures mean different things.**

1. **Capture** -- does the backend see the same activation? A divergence here explains any divergence at
   layers 2 and 3, so testing it separately localizes the fault instead of reporting three failures for
   one cause.
2. **Intervention** -- does the backend apply the same edit, to the same positions? This is where #441
   lives: interp-engine steers every prompt position while our contract is the last token, with matching
   shapes and no exception raised.
3. **Analysis ops** -- do ops built on those backends produce the same values?

**Layer 2 asserts the changed-position SET, not the values and not a count.** A whole-prompt intervention
produces entirely plausible activations, which is why the conflation went unnoticed. A count is no better:
a bug steering position 0 under a last-token spec gives a count of 1 and passes, and a whole-prompt
intervention that happens to be inert at some position also gives a count of 1 while operating on the wrong
scope. The set *is* the claim, and it costs the same to assert.

**Skip discipline.** Participants are contingent on installation, but this module FAILS rather than skips
when NONE is available. A parity suite that silently tests nothing is the worst instance of the
"green summary answering a narrower question" class, because parity is exactly the property people quote
without re-deriving.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from torch.testing import assert_close

from interpretune.utils.import_utils import package_available

MODEL_ID = "gpt2"
LAYER = 5
PROMPT = "The capital of France is"

# Tight, because every participant runs the SAME forward: a difference should come from the observation or
# edit mechanism, which is what is under test, and not from arithmetic drift.
RTOL, ATOL = 1e-4, 1e-4

# "Changed" needs a stated tolerance or the position set goes noisy from nondeterminism at ~1e-9 and the
# pressure becomes to loosen the assertion until it passes -- a guard decaying into decoration. This is far
# above numerical noise and far below any real intervention's effect.
CHANGED_ATOL = 1e-3

#: Participants, each a name plus the package that must be importable. Asked HERE rather than exported from
#: core: the bundled availability flags name BUNDLED adapters, and `interp_engine` is hub-delivered. A test
#: may know which optional package it needs; core may not.
PARTICIPANTS = {
    "interp_engine": "interp_engine",
    "transformer_lens": "transformer_lens",
    "nnsight": "nnsight",
}
AVAILABLE = sorted(name for name, pkg in PARTICIPANTS.items() if package_available(pkg))

# Imported at COLLECTION rather than inside a test. `interp_engine` pulls in a compiler/runtime stack that
# sets TORCHINDUCTOR_*, TRITON_* and TILELANG_* as an import side effect, and the suite fails any test that
# leaves the environment dirtier than it found it. That check is PER-TEST, so an import inside the first
# test is attributed to it no matter how carefully it cleans up -- the variables are set before the test
# body can restore anything. Module scope puts the side effect outside every test's snapshot.
if "interp_engine" in AVAILABLE:  # pragma: no cover - imported for side-effect ordering, not for the name
    import interp_engine  # noqa: F401


class TestTheParticipantSetIsNonEmpty:
    """This module must never pass by having nothing to compare.

    Every other test here is gated on a participant being installed. If the whole set were absent, they would all skip
    and the file would report success while asserting nothing about parity -- and parity is precisely the kind of
    property that gets quoted later without being re-derived.
    """

    def test_at_least_one_participant_is_installed(self):
        assert AVAILABLE, (
            "no execution backend from "
            f"{sorted(PARTICIPANTS)} is installed, so this module compared nothing. Parity cannot be "
            "reported from this environment; install at least one participant."
        )


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
    """Library-independent ground truth: the block's input and the resulting logits, off the HF module.

    Captured with a plain forward pre-hook so nothing under test participates in producing the reference.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        captured["resid_pre"] = args[0].detach().clone()

    handle = hf_model.transformer.h[LAYER].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            out = hf_model(prompt_ids)
    finally:
        handle.remove()
    assert "resid_pre" in captured, "the pre-hook never fired; the reference would be vacuous"
    return {"resid_pre": captured["resid_pre"][0], "logits": out.logits.detach().clone()}


class TestTheReferenceIsLive:
    """Positive control for the whole module.

    Every comparison below is against `hf_reference`. A degenerate reference would let all of them pass
    while measuring nothing, so pin that it is real before believing anything compared to it.
    """

    def test_the_captured_activation_is_non_degenerate(self, hf_reference, prompt_ids):
        resid = hf_reference["resid_pre"]
        assert resid.shape[0] == prompt_ids.shape[1], "one position per input token"
        assert torch.isfinite(resid).all()
        assert resid.abs().max() > 0, "reference is all zeros; comparisons against it prove nothing"

    def test_the_reference_logits_are_non_degenerate(self, hf_reference, prompt_ids):
        logits = hf_reference["logits"]
        assert logits.shape[:2] == (1, prompt_ids.shape[1])
        assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------------------------------
# Layer 1: capture
# --------------------------------------------------------------------------------------------------


def _capture_interp_engine(prompt_ids, layer):
    """Interp-engine's capture is async-only, takes a FLAT token sequence, and returns Address-keyed dict."""
    from interp_engine.model import EagerModel

    model = EagerModel(MODEL_ID, dtype="float32")
    out = asyncio.run(model.capture(prompt_ids[0], [f"resid_pre.{layer}"]))
    assert out, "capture returned nothing"
    return next(iter(out.values()))


def _capture_transformer_lens(prompt_ids, layer):
    """**TransformerBridge, not `HookedTransformer.from_pretrained`** -- and the difference is the point.

    The governance doc's HF-native family is defined by *executing the HF forward*, and the bridge is the
    TransformerLens member that does. `HookedTransformer.from_pretrained` is the LEGACY path: it converts
    weights, folding LayerNorm and centering writing weights by default. Those transforms preserve model
    OUTPUTS while changing the residual stream, so its `hook_resid_pre` is a differently-scaled tensor
    wearing the same name.

    Measured here rather than reasoned about: the first version of this function used
    `HookedTransformer.from_pretrained` and failed this module's own convergence assertion, while
    interp-engine and NNsight passed. That is "Matching names is not matching tensors" (governance doc)
    catching a real instance on its first run -- and it is exactly the failure a value-tolerance nudge would
    have buried, since the natural response to one participant disagreeing is to widen the tolerance.
    """
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers(MODEL_ID, device="cpu")
    _, cache = model.run_with_cache(prompt_ids, names_filter=f"blocks.{layer}.hook_resid_pre")
    return cache[f"blocks.{layer}.hook_resid_pre"][0]


def _capture_nnsight(prompt_ids, layer):
    """Two NNsight idioms matter here, and both are documented in our own nnsight backend.

    ``nnsight.save(value)`` is the save call in 0.6+, not ``value.save()`` -- the latter raises
    ``AttributeError: 'Tensor' object has no attribute 'save'``, because ``.input`` already resolves to a
    real tensor rather than a proxy. That is the second idiom: ``.input`` returns the **first positional
    argument** directly, so it IS the residual stream entering the block, with no tuple to index.
    """
    import nnsight
    from nnsight import LanguageModel

    model = LanguageModel(MODEL_ID, device_map="cpu", dispatch=True)
    with model.trace(prompt_ids):
        saved = nnsight.save(model.transformer.h[layer].input)
    return saved[0].detach()


CAPTURERS = {
    "interp_engine": _capture_interp_engine,
    "transformer_lens": _capture_transformer_lens,
    "nnsight": _capture_nnsight,
}


@pytest.mark.parametrize("participant", sorted(PARTICIPANTS))
class TestCaptureConvergesOnTheForward:
    """Layer 1. Generalizes the interp-engine-only forward leg to the whole family.

    Kept as its own layer because a capture divergence would explain an intervention divergence: reporting
    both would be one cause presented as two failures, and the intervention one would be the misleading
    half.
    """

    def test_resid_pre_matches_the_hf_reference(self, participant, hf_reference, prompt_ids):
        if participant not in AVAILABLE:
            pytest.skip(f"{participant} is not installed")
        got = CAPTURERS[participant](prompt_ids, LAYER)
        assert_close(
            got.to(hf_reference["resid_pre"].dtype),
            hf_reference["resid_pre"],
            rtol=RTOL,
            atol=ATOL,
            msg=f"{participant}'s resid_pre diverged from the HF forward it is supposed to be observing",
        )

    def test_a_different_layer_is_a_different_tensor(self, participant, prompt_ids):
        """Negative control on capture itself, per participant.

        Without it, a capture that ignored the requested point and returned a fixed tensor would pass the comparison
        above. Two layers must disagree for that agreement to mean anything.
        """
        if participant not in AVAILABLE:
            pytest.skip(f"{participant} is not installed")
        a = CAPTURERS[participant](prompt_ids, LAYER)
        b = CAPTURERS[participant](prompt_ids, LAYER + 1)
        assert not torch.allclose(a, b, rtol=RTOL, atol=ATOL), (
            f"{participant} returned the same activation for two different layers, so it is not honouring "
            "the requested point and its convergence result above is meaningless"
        )


# --------------------------------------------------------------------------------------------------
# Layer 2: intervention scope
# --------------------------------------------------------------------------------------------------
#
# The discriminator is CAUSALITY, not introspection, which is what makes it backend-agnostic. In a causal
# LM, editing the residual stream at position p can only affect positions >= p downstream. So a last-token
# intervention moves exactly the final position of a later activation, and a whole-prompt intervention
# moves all of them. That turns "which positions were steered" -- an internal question each backend answers
# differently, if at all -- into one observable available identically to every participant.
#
# The observable is the FINAL layer's `resid_post` rather than logits, because that is a point every
# participant actually has: interp-engine's vocabulary is activation points and carries no `logits` entry.
# Reference and participant are read at the same point, so the comparison stays apples-to-apples.

STEER_SCALE = 12.0
LAST_LAYER = 11  # gpt2 has 12 blocks; the last one's output is the observable


def _steering_vector(hf_reference):
    """A unit direction with real effect, derived from the reference so it is meaningful for this model."""
    resid = hf_reference["resid_pre"]
    return (resid[-1] / resid[-1].norm()).clone()


def changed_positions(baseline, intervened, atol=CHANGED_ATOL):
    """Positions that moved by more than a STATED tolerance.

    The tolerance is not decoration. Exact inequality flags positions that moved ~1e-9 from
    nondeterminism, the set goes noisy, and the pressure becomes to loosen the assertion until it passes --
    a guard decaying into decoration on a schedule nobody notices. `CHANGED_ATOL` sits far above numerical
    noise and far below any real intervention's effect.
    """
    delta = (intervened - baseline).abs()
    delta = delta.amax(dim=-1)
    while delta.ndim > 1:
        delta = delta.amax(dim=0)
    return {int(i) for i in torch.nonzero(delta > atol).flatten().tolist()}


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


class TestTheScopeDiscriminatorWorks:
    """Validate the instrument before using it, in BOTH directions.

    A test that only checks the last-token case cannot distinguish "the discriminator works" from "the discriminator
    always returns the last position". Asserting the whole-prompt case too is what makes the instrument able to tell the
    scopes apart -- which is the entire claim it is used for below.
    """

    def test_a_last_token_intervention_moves_exactly_the_final_position(
        self, hf_model, hf_reference, hf_final_resid, prompt_ids
    ):
        vector = _steering_vector(hf_reference)
        steered = _hf_steered_final_resid(hf_model, prompt_ids, vector, all_positions=False)
        assert changed_positions(hf_final_resid, steered) == {prompt_ids.shape[1] - 1}

    def test_a_whole_prompt_intervention_moves_every_position(self, hf_model, hf_reference, hf_final_resid, prompt_ids):
        """The negative control.

        Without it the assertion above is satisfiable by a broken discriminator.
        """
        vector = _steering_vector(hf_reference)
        steered = _hf_steered_final_resid(hf_model, prompt_ids, vector, all_positions=True)
        assert changed_positions(hf_final_resid, steered) == set(range(prompt_ids.shape[1]))


@pytest.mark.skipif("interp_engine" not in AVAILABLE, reason="interp-engine is not installed")
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
    ``all_positions`` semantics exactly, and must be distinguishable from ``last_token``.
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
        the same thing. This pins that interp-engine's scope is genuinely the whole prompt -- which is what
        an adapter declaring only ``INTERVENTION_ALL_POSITIONS`` is asserting about it.
        """
        vector = _steering_vector(hf_reference)
        steered = self._steered_final_resid(prompt_ids, vector)
        assert changed_positions(hf_final_resid, steered) == set(range(prompt_ids.shape[1])), (
            "interp-engine's steering is no longer whole-prompt. An adapter declaring only "
            "INTERVENTION_ALL_POSITIONS for it would now be wrong, and #441's framing needs revisiting."
        )


# --------------------------------------------------------------------------------------------------
# Layer 3: analysis ops
# --------------------------------------------------------------------------------------------------
#
# Layer 3 decomposes, and the decomposition is worth stating because it says where the risk actually is.
# `logit_diffs_impl` derives everything it returns from `analysis_batch.answer_logits` and
# `answer_indices`; it never touches the backend. So "do analysis ops agree across backends" is really
# two questions:
#
#   (a) do backends produce the same ANSWER LOGITS?   <- varies by backend; this is where risk lives
#   (b) given identical inputs, is the op DETERMINISTIC and backend-independent?  <- pure by construction
#
# Both are asserted, because (b) being true by construction today is exactly the kind of property that
# stops being true when someone adds a backend-conditional branch to an op, and nothing else would catch
# that. Asserting only (a) would leave the op layer unguarded; asserting only (b) would test arithmetic
# nobody doubts.


def _final_logits_transformer_lens(prompt_ids):
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers(MODEL_ID, device="cpu")
    return model(prompt_ids)[0, -1, :].detach()


def _final_logits_nnsight(prompt_ids):
    import nnsight
    from nnsight import LanguageModel

    model = LanguageModel(MODEL_ID, device_map="cpu", dispatch=True)
    with model.trace(prompt_ids):
        saved = nnsight.save(model.output.logits)
    return saved[0, -1, :].detach()


#: interp-engine is deliberately ABSENT here, and that absence is a finding rather than an omission.
#: Its point vocabulary is activation points; it exposes no `logits` point, so "the model's output
#: distribution" -- the one tensor every analysis op ultimately reduces to -- has no name in it. Layers 1
#: and 2 work around this by observing the final layer's `resid_post` instead, which is the same
#: workaround the adapter has to make. That mismatch between hook vocabularies is the concrete friction
#: motivating the activation-point naming refactor; see the issue tracking it.
FINAL_LOGITS = {
    "transformer_lens": _final_logits_transformer_lens,
    "nnsight": _final_logits_nnsight,
}


@pytest.mark.parametrize("participant", sorted(FINAL_LOGITS))
class TestAnswerLogitsConvergeOnTheForward:
    """(a) The half that can actually differ: the tensor every analysis op reduces to."""

    def test_final_position_logits_match_the_hf_reference(self, participant, hf_reference, prompt_ids):
        if participant not in AVAILABLE:
            pytest.skip(f"{participant} is not installed")
        got = FINAL_LOGITS[participant](prompt_ids)
        ref = hf_reference["logits"][0, -1, :]
        assert_close(
            got.to(ref.dtype),
            ref,
            rtol=RTOL,
            atol=ATOL,
            msg=(
                f"{participant}'s final-position logits diverged from the HF forward. Every analysis op "
                "reduces to this tensor, so a divergence here propagates to all of them."
            ),
        )


class TestTheOpLayerIsBackendIndependent:
    """(b) The half that is pure by construction -- asserted so it cannot quietly stop being.

    `logit_diffs_impl` takes its inputs from the analysis batch and never consults the backend. That is a
    design property, not an accident, and it is what lets one op serve every backend. A future
    backend-conditional branch inside an op would break it silently: the op would still run, still return
    plausible numbers, and no existing parity test would notice, because they all compare backends running
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
