"""The `scale` mode: the activation multiplied elementwise by `intervention_tensor * scale_factor` at the scope."""

from __future__ import annotations

import torch

from interpretune.analysis.backends.capabilities import InterventionMode, InterventionSupport, PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec, apply_intervention, build_intervention_dict


def _scale(value: torch.Tensor, multiplier, *, scale_factor: float = 1.0, scope=PositionScope.ALL_POSITIONS):
    spec = InterventionSpec(
        intervention_tensor=torch.as_tensor(multiplier),
        mode=InterventionMode.SCALE,
        scale_factor=scale_factor,
        position_scope=scope,
    )
    return apply_intervention(value.clone(), spec, last_pos=value.shape[1] - 1)


def test_ones_are_the_identity():
    value = torch.randn(2, 3, 5)
    torch.testing.assert_close(_scale(value, torch.ones(5)), value, rtol=0, atol=0)


def test_a_per_unit_multiplier_moves_only_the_named_units():
    value = torch.randn(1, 2, 4)
    multiplier = torch.tensor([1.0, 3.0, 1.0, 0.0])
    out = _scale(value, multiplier)
    torch.testing.assert_close(out[..., [0, 2]], value[..., [0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(out[..., 1], value[..., 1] * 3.0)
    assert torch.count_nonzero(out[..., 3]) == 0


def test_the_scale_factor_multiplies_the_tensor_and_the_scope_is_honoured():
    value = torch.randn(1, 3, 4)
    out = _scale(value, 2.0, scale_factor=1.5, scope=PositionScope.LAST_TOKEN)
    torch.testing.assert_close(out[:, -1], value[:, -1] * 3.0)
    torch.testing.assert_close(out[:, :-1], value[:, :-1], rtol=0, atol=0)


def test_every_declares_scale_and_a_payload_canonicalizes_it():
    assert InterventionMode.SCALE in InterventionSupport.every().modes
    spec = InterventionSpec(intervention_tensor=torch.ones(4), mode="scale")
    hook = "blocks.0.hook_resid_post"
    canonical = build_intervention_dict({hook: spec}, {hook: [hook]}, {hook: (1, 3, 4)})
    (only,) = canonical[hook]
    assert only.mode is InterventionMode.SCALE
