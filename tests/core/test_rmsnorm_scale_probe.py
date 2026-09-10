"""The final-norm scale is read from the module, not declared from a table of model families.

The table this replaces was a standing liability rather than a one-time cost: `gemma`/`gemma2`/`gemma3`
apply `(1 + weight)` while `gemma3n` and the whole `gemma4` line apply `weight`, so the split runs INSIDE
one family prefix and no prefix test separates them. A wrong entry is silent -- every direction built
from that model is scaled incorrectly and still looks plausible.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from interpretune.analysis.optools import _RMSNORM_PROBE_CONSTANT, _rmsnorm_scale

D = 32
#: (model_type, class prefix, applies the (1 + weight) convention)
FAMILIES = [
    ("gemma", "Gemma", True),
    ("gemma2", "Gemma2", True),
    ("gemma3", "Gemma3", True),
    ("gemma3n", "Gemma3n", False),
    ("gemma4", "Gemma4", False),
    ("llama", "Llama", False),
    ("qwen2", "Qwen2", False),
    ("mistral", "Mistral", False),
]


def _norm(model_type: str, prefix: str, weight: torch.Tensor):
    try:
        module = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        norm = getattr(module, f"{prefix}RMSNorm")(D)
    except (ImportError, AttributeError):
        pytest.skip(f"{prefix}RMSNorm not present in this transformers build")
    with torch.no_grad():
        norm.weight.data = weight.clone().to(norm.weight.dtype)
    return norm


@pytest.fixture
def weight() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(D) * 0.3


class TestTheProbeReadsBehaviourNotTheParameter:
    @pytest.mark.parametrize("model_type, prefix, offset", FAMILIES)
    def test_recovery_is_exact(self, model_type, prefix, offset, weight):
        norm = _norm(model_type, prefix, weight)
        expected = (1.0 + weight) if offset else weight
        assert torch.allclose(_rmsnorm_scale(norm, norm.weight).float(), expected, atol=1e-6)

    def test_it_reads_the_applied_scale_rather_than_the_stored_weight(self, weight):
        """POSITIVE CONTROL for the whole approach. On an offset family the probe must return `1 + w`.

        Returning `w` would mean it is reporting the parameter, which every family stores identically, and
        the exact agreement everywhere else would be an artifact of reading the same tensor back.
        """
        norm = _norm("gemma2", "Gemma2", weight)
        probed = _rmsnorm_scale(norm, norm.weight).float()
        assert torch.allclose(probed, 1.0 + weight, atol=1e-6)
        assert not torch.allclose(probed, weight, atol=1e-3), "the probe returned the stored weight"

    def test_the_probe_amplitude_avoids_the_eps_regime(self):
        """`rms(c*1) = c` for any c, but at c = 1 the eps term is not negligible against it."""
        assert _RMSNORM_PROBE_CONSTANT >= 10.0

    def test_the_result_is_detached(self, weight):
        """Lens directions are consumed as constant bases; an attached scale is a quieter wrong answer."""
        norm = _norm("llama", "Llama", weight)
        assert not _rmsnorm_scale(norm, norm.weight).requires_grad


class TestItWorksThroughTheProxyTheSeamActuallyServes:
    def test_an_nnsight_envoy_is_probed_rather_than_refused(self, weight):
        """`Envoy` is NOT an `nn.Module` subclass, so an isinstance guard would fall back to a declared convention
        on precisely the path this seam exists to serve, while passing on the HF path."""
        envoy = pytest.importorskip("nnsight.intervention.envoy")
        norm = _norm("gemma2", "Gemma2", weight)
        assert not isinstance(envoy.Envoy(norm), torch.nn.Module), "premise gone: revisit the guard"
        assert torch.allclose(_rmsnorm_scale(envoy.Envoy(norm), norm.weight).float(), 1.0 + weight, atol=1e-6)


class TestItRefusesRatherThanGuessing:
    def test_a_norm_that_cannot_be_probed_is_refused_by_name(self, weight):
        class Unprobeable(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("needs an attention mask")

        with pytest.raises(RuntimeError, match=r"could not read the elementwise scale from 'Unprobeable'"):
            _rmsnorm_scale(Unprobeable(), weight)

    def test_a_wrong_shaped_result_is_refused_rather_than_substituted(self, weight):
        class WrongShape(torch.nn.Module):
            def forward(self, x):
                return x[..., :3]

        with pytest.raises(RuntimeError, match=r"returned a scale of shape"):
            _rmsnorm_scale(WrongShape(), weight)

    def test_a_non_finite_result_is_refused(self, weight):
        class NotFinite(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x, float("nan"))

        with pytest.raises(RuntimeError, match=r"non-finite"):
            _rmsnorm_scale(NotFinite(), weight)
