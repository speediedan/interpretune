"""Direct unit payloads agree with an independent HF edit on both model backends."""

import pytest
import torch

from interpretune.analysis.venngram import unit_interventions


IDS = torch.tensor([[464, 3139, 286, 4881, 318]])
POINT = "mlp.out.hook_in"
UNITS = (1, 7)


@pytest.fixture(scope="module", params=["transformer_lens", "nnsight"])
def unit_backend(request):
    if request.param == "transformer_lens":
        from transformer_lens.model_bridge import TransformerBridge
        from interpretune.adapters.transformer_lens.backends import TLModelBackend

        model = TransformerBridge.boot_transformers("gpt2", device="cpu")
        yield TLModelBackend(), model, model.original_model, {"input": IDS}
    else:
        from nnsight import LanguageModel
        from interpretune.adapters.nnsight.backends import NNsightModelBackend
        from interpretune.analysis.backends.hook_mapping import HookNameResolver

        model = LanguageModel("gpt2", device_map="cpu", dispatch=True)
        yield NNsightModelBackend(HookNameResolver("GPT2LMHeadModel")), model, model._module, {"input_ids": IDS}


@pytest.mark.parametrize("scope", ["last_token", "all_positions"])
@pytest.mark.parametrize("op", ["add", "scale", "replace", "clamp"])
def test_a_unit_address_matches_a_manual_hf_pre_hook(unit_backend, scope, op):
    backend, model, hf, batch = unit_backend
    payload = unit_interventions(
        f"point:{POINT} & layers:0 & units:[1, 7]",
        n_layers=12,
        site_widths={POINT: 3072},
        op=op,
        position_scope=scope,
        **({"clamp_min": -0.01, "clamp_max": 0.01} if op == "clamp" else {"value": [2.0, -3.0]}),
    )
    pre, actual = backend.fwd_w_intervention(model, batch, payload)

    def edit(module, args):
        value = args[0].clone()
        region = value if scope == "all_positions" else value[:, -1, :]
        for unit, dose in zip(UNITS, [2.0, -3.0], strict=True):
            if op == "add":
                region[..., unit] += dose
            elif op == "scale":
                region[..., unit] *= dose
            elif op == "replace":
                region[..., unit] = dose
            else:
                region[..., unit].clamp_(-0.01, 0.01)
        return (value, *args[1:])

    handle = hf.transformer.h[0].mlp.c_proj.register_forward_pre_hook(edit)
    try:
        with torch.no_grad():
            expected = hf(IDS).logits
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert (actual - pre).abs().max().item() > 1e-5, "the intervention must measurably change logits"


def test_a_gradient_unit_address_preserves_every_unselected_gradient(unit_backend):
    backend, model, _hf, batch = unit_backend
    hook = "blocks.0.hook_resid_post"
    payload = unit_interventions(
        "point:hook_resid_post & layers:0 & units:[1, 7] & of:grad",
        n_layers=12,
        site_widths={"hook_resid_post": 768},
        op="replace",
        value=0.5,
        flow="grad",
    )
    pre, post, _ = backend.bwd_w_intervention(
        model, batch, payload, lambda logits: logits[:, -1, 1000].sum(), capture=[hook]
    )
    expected = pre[hook].clone()
    expected[:, -1, list(UNITS)] = 0.5
    torch.testing.assert_close(post[hook], expected)
    assert not torch.equal(pre[hook], post[hook])
