"""#329: expected-vs-actual validation for J-lens patch interventions at fixed magnitude.

THE FREEZE-SET DECISION (the deliverable this issue names). The circuit-tracer validation
(`_verify_feature_edges_direct`) freezes components because its EXPECTED values come from attribution
edge weights -- a linear surrogate that only matches a finite intervention when the nonlinearities
between the perturbation and the readout are pinned. The J-lens patch case decomposes differently:

1. **Machinery validation needs NO freezing.** The claim under test is "the traced intervention path
   applies exactly ``V(s*sigma(c) - c)`` at the declared hook and position". An EAGER REFERENCE -- the
   same model run under a plain ``register_forward_hook`` applying the identical spec -- produces exact
   expected values through every nonlinearity, because both runs traverse the same ones. Tolerances are
   float-noise tight. This is the level that catches machinery defects: the transformers-5.x
   ``envoy.output[0]`` write path would have overwritten batch row 0 and failed here by a wide margin.
2. **Surrogate validation is a different claim with a different owner.** "The J-lens transport
   predicts the logit effect" is a statement about lens quality (an AVERAGED Jacobian standing in for
   the per-prompt one); its finite-magnitude form is exactly where a freeze set would re-enter, and it
   belongs to the folding investigation (#330) and future J-lens subspace attribution work. What this
   module pins instead is the bridge both must satisfy: for small perturbations, the measured logit
   delta converges to the TRUE per-prompt Jacobian-vector product -- no freezing, no surrogate, just
   calculus. Any frozen linearization that disagrees with this limit is wrong by construction.

Mechanics constants are per-comparison and deliberately named:
- ``EAGER_ATOL``: trace-vs-eager on identical weights/dtype differs only by op scheduling; observed
  drift is <1e-6 on float32, so 1e-4 leaves margin without admitting real defects.
- ``JVP_CONVERGENCE_FACTOR``: first-order error is O(eps); a 10x eps reduction should shrink the
  residual ~10x. Asserting >=3x tolerates curvature while still failing on any systematic mismatch.
"""

from __future__ import annotations

import pytest
import torch

from tests.runif import RunIf

from interpretune.analysis.backends.impls.nnsight import (
    HookNameResolver,
    NNsightModelBackend,
    get_default_configs_per_pass,
)
from interpretune.analysis.backends.interventions import (
    InterventionDict,
    InterventionSpec,
    apply_intervention_to_last_token,
    _validate_intervention_spec,
)

D_MODEL = 32
N_LAYERS = 4
HOOK_LAYER = 2
PROMPT_IDS = [3, 17, 9, 25, 11]

EAGER_ATOL = 1e-4
JVP_CONVERGENCE_FACTOR = 3.0


@pytest.fixture(scope="module")
def tiny_gpt2_dir(tmp_path_factory):
    """A seeded random tiny GPT-2 saved to disk so nnsight can load it by path."""
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    torch.manual_seed(42)
    config = GPT2Config(n_layer=N_LAYERS, n_head=4, n_embd=D_MODEL, vocab_size=64, n_positions=32)
    model = GPT2LMHeadModel(config).eval()
    path = tmp_path_factory.mktemp("tiny_gpt2")
    model.save_pretrained(path)
    # any real tokenizer satisfies nnsight's loader; the tests drive the model with raw ids
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def nnsight_setup(tiny_gpt2_dir):
    from nnsight import LanguageModel

    lm = LanguageModel(str(tiny_gpt2_dir), device_map="cpu", dispatch=True)
    hf_model = NNsightModelBackend._get_hf_model(lm)
    backend = NNsightModelBackend(
        HookNameResolver(hf_model.config.architectures[0]), configs_per_pass=get_default_configs_per_pass()
    )
    backend.register_model_hooks(lm)
    return lm, hf_model, backend


def _patch_pair() -> torch.Tensor:
    torch.manual_seed(7)
    v_s = torch.randn(D_MODEL)
    return torch.stack([v_s, v_s * 0.4 + torch.randn(D_MODEL)])  # deliberately oblique


def _spec(scale: float) -> InterventionSpec:
    return _validate_intervention_spec(
        InterventionSpec(intervention_tensor=_patch_pair(), mode="patch", scale_factor=scale),
        target_shape=(D_MODEL,),
        hook_name=f"blocks.{HOOK_LAYER}.hook_resid_post",
    )


def _eager_reference_logits(hf_model, spec: InterventionSpec | None) -> torch.Tensor:
    """The expected values: a plain eager forward, optionally applying the spec with an HF hook."""
    ids = torch.tensor([PROMPT_IDS])
    handle = None
    if spec is not None:

        def hook(_m, _a, out):
            hidden = out[0] if isinstance(out, tuple) else out
            apply_intervention_to_last_token(hidden, spec, last_pos=hidden.shape[1] - 1)
            return out

        handle = hf_model.transformer.h[HOOK_LAYER].register_forward_hook(hook)
    try:
        with torch.no_grad():
            return hf_model(ids).logits[0, -1].float()
    finally:
        if handle is not None:
            handle.remove()


class TestMechanicsAgainstEagerReference:
    """Level 1: the traced intervention path must match an eager reference exactly-ish."""

    @pytest.mark.parametrize("scale", [1.0, 2.0], ids=["pure-swap", "double-strength"])
    def test_traced_patch_equals_eager_patch(self, nnsight_setup, scale):
        lm, hf_model, backend = nnsight_setup
        hook = f"blocks.{HOOK_LAYER}.hook_resid_post"
        interventions = InterventionDict({hook: (_spec(scale),)})
        with torch.no_grad():
            pre, post = backend.fwd_w_intervention(
                model=lm, batch={"input_ids": torch.tensor([PROMPT_IDS])}, interventions=interventions
            )
        expected_clean = _eager_reference_logits(hf_model, None)
        expected_patched = _eager_reference_logits(hf_model, _spec(scale))
        torch.testing.assert_close(pre[0, -1].float(), expected_clean, atol=EAGER_ATOL, rtol=0)
        torch.testing.assert_close(post[0, -1].float(), expected_patched, atol=EAGER_ATOL, rtol=0)
        # and the intervention genuinely did something, so the agreement above is not vacuous
        assert not torch.allclose(expected_patched, expected_clean, atol=1e-3)

    def test_effect_is_confined_to_the_last_position(self, nnsight_setup):
        """Positions before the intervention point must be byte-identical to the clean run."""
        lm, _hf_model, backend = nnsight_setup
        hook = f"blocks.{HOOK_LAYER}.hook_resid_post"
        with torch.no_grad():
            pre, post = backend.fwd_w_intervention(
                model=lm,
                batch={"input_ids": torch.tensor([PROMPT_IDS])},
                interventions=InterventionDict({hook: (_spec(1.0),)}),
            )
        torch.testing.assert_close(post[0, :-1], pre[0, :-1])


class TestFirstOrderAgainstTrueJacobian:
    """Level 2: small perturbations must converge to the TRUE per-prompt JVP.

    This is the freezing-free bridge to surrogate validation: any linearization (frozen or averaged)
    claiming to predict logit effects must agree with this limit.
    """

    def test_small_patch_delta_converges_to_jvp(self, tiny_gpt2_dir):
        from transformers import GPT2LMHeadModel

        # A FRESH float64 model from the same saved weights. Two deliberate choices: float64 because
        # at eps=1e-2 the measured delta divides float32 forward noise by eps, putting the residual
        # at the noise floor and drowning the convergence signal (measured: the eps=1e-1 residual
        # was already 1.2e-3 while the eps=1e-2 one plateaued); and a fresh load rather than a
        # deepcopy of the fixture model, because copying the nnsight-instrumented module yields a
        # model whose registered forward hooks silently never fire (measured: KeyError on the
        # capture), which is exactly the kind of silence this file exists to distrust.
        hf_model = GPT2LMHeadModel.from_pretrained(tiny_gpt2_dir).double().eval()
        ids = torch.tensor([PROMPT_IDS])
        pair = _patch_pair()
        v_matrix = pair.T.double()
        # the analytic patch displacement at scale 1: Delta = V (sigma(c) - c) for the CLEAN residual c
        captured: dict[str, torch.Tensor] = {}

        def capture(_m, _a, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured["h"] = hidden[0, -1].detach().double()
            return out

        handle = hf_model.transformer.h[HOOK_LAYER].register_forward_hook(capture)
        with torch.no_grad():
            clean_logits = hf_model(ids).logits[0, -1].double()
        handle.remove()
        coords = torch.linalg.pinv(v_matrix) @ captured["h"]
        delta = (v_matrix @ (coords.flip(-1) - coords)).float()

        def perturbed_logits(eps: float) -> torch.Tensor:
            def hook(_m, _a, out):
                hidden = out[0] if isinstance(out, tuple) else out
                hidden[0, -1] = hidden[0, -1] + eps * delta.to(hidden.dtype)
                return out

            h = hf_model.transformer.h[HOOK_LAYER].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    return hf_model(ids).logits[0, -1].double()
            finally:
                h.remove()

        # true JVP via central differences at a much smaller step (the reference derivative)
        ref_eps = 1e-4
        jvp = (perturbed_logits(ref_eps) - perturbed_logits(-ref_eps)) / (2 * ref_eps)

        residuals = {}
        for eps in (1e-1, 1e-2):
            measured = (perturbed_logits(eps) - clean_logits) / eps
            residuals[eps] = torch.linalg.norm(measured - jvp).item()
        assert residuals[1e-2] * JVP_CONVERGENCE_FACTOR < residuals[1e-1], (
            f"first-order convergence failed: residuals {residuals} -- the measured effect of a small "
            "patch displacement is not approaching the true Jacobian-vector product"
        )


class TestCrossBackendAgreement:
    """Level 3: the same patch through both backends' intervention paths must agree.

    The demo measured +4.50 (nnsight) vs +4.25 (TL) on gemma-2-2b's 2304-dim stream -- close but
    only eyeballed. This pins it on gpt2 with named tolerances. Standalone-marked at the METHOD
    level (class-level marks are invisible to the collection filter): real-model loads are too heavy
    for the default CPU lane.
    """

    @RunIf(standalone=True)
    def test_tl_and_nnsight_patch_deltas_agree_on_gpt2(self, tmp_path):
        from nnsight import LanguageModel
        from transformer_lens import HookedTransformer

        from interpretune.analysis.backends.impls.transformer_lens import TLModelBackend

        prompt = "The capital of France is"
        hook = "blocks.8.hook_resid_post"
        torch.manual_seed(11)
        pair = torch.randn(2, 768)

        # --- NNsight path ---
        lm = LanguageModel("gpt2", device_map="cpu", dispatch=True)
        hf_model = NNsightModelBackend._get_hf_model(lm)
        nns_backend = NNsightModelBackend(
            HookNameResolver(hf_model.config.architectures[0]), configs_per_pass=get_default_configs_per_pass()
        )
        nns_backend.register_model_hooks(lm)
        ids = lm.tokenizer(prompt, return_tensors="pt")["input_ids"]
        spec = _validate_intervention_spec(
            InterventionSpec(intervention_tensor=pair, mode="patch", scale_factor=1.0),
            target_shape=(768,),
            hook_name=hook,
        )
        with torch.no_grad():
            nns_pre, nns_post = nns_backend.fwd_w_intervention(
                model=lm, batch={"input_ids": ids}, interventions=InterventionDict({hook: (spec,)})
            )
        nns_delta = (nns_post[0, -1] - nns_pre[0, -1]).float()

        # --- TransformerLens path (same weights by construction: TL converts the same checkpoint) ---
        # NO-processing load, and it is load-bearing: default from_pretrained folds LN and centers
        # weights, which CHANGES the residual basis at hook_resid_post -- the same patch pair then
        # acts in a different geometry and the deltas disagree almost completely (measured: relative
        # gap 0.948 with processing on). The production TL path (circuit-tracer's ReplacementModel)
        # loads unprocessed for SAE compatibility, which is why the demo's backends agreed; this
        # test matches that contract, and documents that patch pairs are defined in the UNPROCESSED
        # residual basis.
        tl_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
        tl_backend = TLModelBackend()
        with torch.no_grad():
            tl_pre, tl_post = tl_backend.fwd_w_intervention(
                model=tl_model, batch={"input": ids}, interventions=InterventionDict({hook: (spec,)})
            )
        tl_delta = (tl_post[0, -1] - tl_pre[0, -1]).float()

        # With processing off, both backends run the same computation in the same basis; the
        # remaining drift is float32 op-ordering. Observed cross-backend delta drift on the gemma
        # demo was ~5% of effect size, so 10% is margin without admitting a geometry mismatch.
        effect = torch.linalg.norm(nns_delta)
        assert effect > 0.1, "patch produced no measurable effect; agreement below would be vacuous"
        rel_gap = torch.linalg.norm(nns_delta - tl_delta) / effect
        assert rel_gap < 0.10, f"backends disagree: relative delta gap {rel_gap:.3f}"
