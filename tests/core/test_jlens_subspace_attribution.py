"""Tests for the J-lens subspace attribution decomposition (#338).

The decomposition splits the first-order prediction ``gᵀΔh`` per dictionary direction, so its
contract is algebraic before it is empirical: shares plus remainder reconstruct the prediction
exactly. The suite pins that identity (including the two boundary dictionaries: one spanning the
displacement, one orthogonal to it), the refusals, and — standalone, on the real gemma-2-2b pair —
that the pair dictionary explains its own patch displacement with a small remainder.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.ops.bundled.jlens.jlens_ops import subspace_attribution_scores
from tests.runif import RunIf

D = 16


def _report(v: torch.Tensor, g: torch.Tensor, d: torch.Tensor) -> dict[str, object]:
    return subspace_attribution_scores(g, d, v, [0, 1], "jlens_norm_aware")


class TestSubspaceAttributionIdentity:
    def test_shares_plus_remainder_reconstruct_the_prediction(self) -> None:
        torch.manual_seed(3)
        v = torch.randn(2, D)
        g = torch.randn(D)
        d = torch.randn(D)
        out = _report(v, g, d)
        assert out["predicted_delta"] == pytest.approx(float(g @ d))
        assert out["attribution_total"] + out["unexplained_remainder"] == pytest.approx(out["predicted_delta"])
        assert out["basis"] == "jlens_norm_aware"
        assert out["token_ids"] == [0, 1]
        assert len(out["attribution_shares"]) == 2

    def test_dictionary_spanning_the_displacement_leaves_no_remainder(self) -> None:
        torch.manual_seed(4)
        d = torch.randn(D)
        u = d / torch.linalg.vector_norm(d)
        w = torch.randn(D)
        w = w - (w @ u) * u
        v = torch.stack([u, w / torch.linalg.vector_norm(w).clamp_min(1e-12)])
        g = torch.randn(D)
        out = _report(v, g, d)
        assert out["unexplained_remainder"] == pytest.approx(0.0, abs=1e-5)
        assert out["attribution_total"] == pytest.approx(out["predicted_delta"], rel=1e-5)

    def test_dictionary_orthogonal_to_the_displacement_explains_nothing(self) -> None:
        torch.manual_seed(5)
        d = torch.randn(D)
        q, _ = torch.linalg.qr(torch.randn(D, D))
        v = q[:2] - (q[:2] @ d).unsqueeze(-1) * (d / torch.linalg.vector_norm(d) ** 2).unsqueeze(0)
        v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
        g = torch.randn(D)
        out = _report(v, g, d)
        assert out["attribution_shares"] == pytest.approx([0.0, 0.0], abs=1e-5)
        assert out["unexplained_remainder"] == pytest.approx(out["predicted_delta"])

    def test_orthonormal_dictionary_matches_readout_times_coordinate(self) -> None:
        torch.manual_seed(6)
        q, _ = torch.linalg.qr(torch.randn(D, D))
        v, g, d = q[:2], torch.randn(D), torch.randn(D)
        out = _report(v, g, d)
        expected = [(float(v[i] @ g) * float(v[i] @ d)) for i in range(2)]
        assert out["attribution_shares"] == pytest.approx(expected)

    def test_empty_dictionary_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="at least one dictionary token"):
            subspace_attribution_scores(torch.randn(D), torch.randn(D), torch.zeros(0, D), [], "embed")

    def test_mismatched_widths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="matching widths"):
            subspace_attribution_scores(torch.randn(D), torch.randn(D + 1), torch.randn(2, D), [0, 1], "embed")


class TestRealPairSubspaceAttribution:
    @RunIf(standalone=True)
    def test_pair_dictionary_explains_its_own_patch_displacement(self) -> None:
        """Level-3: a patch displacement lives in its pair's span, so the remainder must be small.

        Runs the production pair on gemma-2-2b L24 eagerly (the CPU lane cannot hold the weights):
        the gradient comes from autograd on the gap metric, the displacement from the shared
        apply_intervention math, and the dictionary is the pair itself.
        """
        from types import SimpleNamespace

        from transformers import AutoModelForCausalLM, AutoTokenizer

        from interpretune.analysis.backends.interventions import (
            InterventionSpec,
            _validate_intervention_spec,
            apply_intervention,
        )
        from interpretune.analysis.optools import (
            jlens_direction_rows,
            resolve_jlens_layer,
            resolve_unembed_and_norm_scale,
        )

        model_id, jlens_id, layer = "google/gemma-2-2b", "gemma-2-2b", 24
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        prompt = "Is orange a color or a fruit? Answer with one word: Color or Fruit. orange ->"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
        wrapped = SimpleNamespace(model=model)
        info = resolve_unembed_and_norm_scale(wrapped)
        j, resolved, _artifact = resolve_jlens_layer(
            wrapped,
            {"concept_group_a": ["Fruit"], "concept_group_b": ["Color"]},
            {"jlens_model_id": jlens_id, "jlens_layer": layer},
        )
        assert resolved == layer
        ids_a = [tokenizer.encode("Fruit", add_special_tokens=False)[-1]]
        ids_b = [tokenizer.encode("Color", add_special_tokens=False)[-1]]
        pair = jlens_direction_rows(info, ids_a + ids_b, j, apply_norm=True).detach().float()

        target = model.model.layers[layer]
        captured: dict[str, torch.Tensor] = {}

        def _capture(_module: object, _inputs: object, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden

        handle = target.register_forward_hook(_capture)
        try:
            with torch.enable_grad():
                logits = model(ids).logits
                h = captured["h"]
                metric = logits[0, -1, ids_a[0]] - logits[0, -1, ids_b[0]]
                (grad,) = torch.autograd.grad(metric, h)
        finally:
            handle.remove()

        spec = _validate_intervention_spec(
            InterventionSpec(intervention_tensor=pair, mode="patch", scale_factor=1.0),
            target_shape=(pair.shape[1],),
            hook_name=f"blocks.{layer}.hook_resid_post",
        )
        delta_h = apply_intervention(h.detach().clone(), spec, last_pos=ids.shape[1] - 1) - h.detach()
        out = subspace_attribution_scores(
            grad.detach().float().cpu()[0, -1],
            delta_h.detach().float().cpu()[0, -1],
            pair.cpu(),
            ids_a + ids_b,
            "jlens_norm_aware",
        )
        assert out["predicted_delta"] != pytest.approx(0.0)
        assert abs(out["unexplained_remainder"]) < 0.05 * abs(out["predicted_delta"])
