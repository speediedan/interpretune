"""Interp-engine observes the same HuggingFace forward the other HF-native backends do.

`docs/adapter_parity_governance.md` places TransformerBridge, NNsight and interp-engine in one family: all
three execute the HuggingFace forward and differ only in how they observe it. TransformerBridge ↔ NNsight is
already covered by `test_model_backend_parity.py`; this module adds the leg that was missing, which is the
one that turns an agreeing PAIR into a convergence SET.

**Why the third member is worth more than more coverage.** A pair that agrees cannot distinguish "both
correct" from "both wrong in the same way", because the two implementations share the forward they are being
used to check. A third independent implementation over that same forward makes agreement evidence about the
forward rather than about the pair.

**The reference is the HF model itself, not one of the participants.** If a participant were the reference,
a participant agreeing with itself would count as convergence. Comparing every member against a
library-independent capture keeps the claim about the forward.

**Why this lives in interpretune rather than in the adapter repo.** Forward-level convergence needs only
`interp_engine`. COMPOSITION-level parity — the adapter's capture and steering surface reached through a
registered composition — needs the hub component, its trust gate and its cached snapshot, so it belongs with
the component (N5). Testing the forward claim here keeps core tests free of a dependency on a published
artifact.

**The compared point is `resid_pre`, chosen deliberately.** It is the residual stream entering a block and
means the same tensor in every implementation. The tempting alternatives do not: interp-engine's sublayer
inputs are POST-norm while TransformerLens' block-level hooks of similar name are PRE-norm, differing by a
whole normalization. Comparing those would produce a real divergence that is not the one this test claims to
measure — and the natural response, loosening the tolerance, would bury a genuine defect. See "Matching
names is not matching tensors" in the governance doc.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from torch.testing import assert_close

from interpretune.utils import _IE_AVAILABLE

MODEL_ID = "gpt2"
LAYER = 5
PROMPT = "The capital of France is"

# Tight, because both run the same forward: any difference should come from the observation mechanism.
RTOL, ATOL = 1e-4, 1e-4


# Imported HERE, at collection, rather than inside a test. `interp_engine` pulls in a compiler/runtime
# stack that sets `TORCHINDUCTOR_*`, `TRITON_*` and `TILELANG_*` as a side effect of import, and the suite
# fails any test that leaves the environment dirtier than it found it. That check is PER-TEST, so an import
# occurring inside the first test is attributed to it no matter how carefully that test cleans up after
# itself -- the variables are set before the test body can restore anything. Importing at module scope moves
# the side effect outside every test's snapshot, which is where an import-time side effect belongs.
if _IE_AVAILABLE:  # pragma: no cover - import for its side effect ordering, not for the name
    import interp_engine  # noqa: F401


@pytest.fixture(scope="module")
def prompt_ids():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    return tok(PROMPT, return_tensors="pt")["input_ids"]


@pytest.fixture(scope="module")
def hf_resid_pre(prompt_ids):
    """Library-independent ground truth: the block's input, captured straight off the HF module."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        captured["resid_pre"] = args[0].detach().clone()

    handle = model.transformer.h[LAYER].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(prompt_ids)
    finally:
        handle.remove()
    assert "resid_pre" in captured, "the pre-hook never fired; the reference would otherwise be vacuous"
    return captured["resid_pre"][0]  # drop the batch dim: interp-engine returns (seq, d_model)


class TestReferenceIsLive:
    """Positive control for the module.

    Every comparison below asserts closeness to `hf_resid_pre`. A degenerate reference would let those
    comparisons pass while measuring nothing, so pin that it is a real activation before believing anything
    compared against it.
    """

    def test_reference_activation_is_non_degenerate(self, hf_resid_pre, prompt_ids):
        assert hf_resid_pre.shape[0] == prompt_ids.shape[1], "one position per input token"
        assert torch.isfinite(hf_resid_pre).all(), "reference carries non-finite values"
        assert hf_resid_pre.abs().max() > 0, "reference is all zeros; comparisons against it prove nothing"


@pytest.mark.skipif(not _IE_AVAILABLE, reason="interp-engine is not installed in this environment")
class TestInterpEngineConvergesOnTheForward:
    @staticmethod
    def _capture(prompt_ids, point: str):
        """Interp-engine's capture is async-only and takes a FLAT token sequence, which it batches itself.

        It returns a dict keyed by ``Address`` objects rather than by the requested strings, so the result
        is looked up by matching the address rather than by the string that was asked for.
        """
        from interp_engine.model import EagerModel

        model = EagerModel(MODEL_ID, dtype="float32")
        out = asyncio.run(model.capture(prompt_ids[0], [point]))
        assert out, f"capture returned nothing for {point!r}"
        return next(iter(out.values()))

    def test_resid_pre_matches_the_hf_reference(self, hf_resid_pre, prompt_ids):
        got = self._capture(prompt_ids, f"resid_pre.{LAYER}")
        assert_close(
            got.to(hf_resid_pre.dtype),
            hf_resid_pre,
            rtol=RTOL,
            atol=ATOL,
            msg="interp-engine's resid_pre diverged from the HF forward it is supposed to be observing",
        )

    def test_a_different_layer_is_a_different_tensor(self, prompt_ids):
        """Negative control on the capture itself.

        Without this, a `capture` that ignored the requested point and returned a fixed tensor would pass
        the comparison above. Two layers must disagree for the agreement above to mean anything.
        """
        a = self._capture(prompt_ids, f"resid_pre.{LAYER}")
        b = self._capture(prompt_ids, f"resid_pre.{LAYER + 1}")
        assert not torch.allclose(a, b, rtol=RTOL, atol=ATOL), (
            "two different layers returned the same activation, so capture is not honouring the requested "
            "point and the convergence result above is meaningless"
        )
