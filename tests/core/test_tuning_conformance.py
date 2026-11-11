"""Core as the first consumer of the tuning conformance cases (#760): finetuning-scheduler's view contract, run
against bundled compositions through the same ``ConformanceTarget`` an adapter repository declares."""

from __future__ import annotations

import pytest

from interpretune.testing.conformance import ConformanceTarget, TuningConformance
from interpretune.utils.import_utils import package_available


class TestBridgeSplicedSAETuning(TuningConformance):
    """gpt2 over the TransformerBridge with an SAE spliced in: a trainable view with a `latent:sae` role."""

    target = ConformanceTarget(
        composition=("core", "sae_lens"), datamodule_flavour="bridge", module_cfg_extras={"add_saes_on_init": True}
    )


@pytest.mark.skipif(not package_available("nnsight"), reason="nnsight is not installed")
class TestNNsightTuning(TuningConformance):
    """A local nnsight composition registers its HF module, so its identity view runs the full cases (#774)."""

    target = ConformanceTarget(
        composition=("core", "nnsight", "sae_lens"), forward_family="hf_native", datamodule_flavour="nnsight"
    )

    def test_view_is_trainable(self, probe):
        """A refusing view would pass the cases above with only the refusal case run, so require the real view."""
        assert probe.support().trainable, probe.support().reason


@pytest.mark.skipif(
    not (package_available("nnsight") and package_available("lightning")), reason="nnsight and lightning are needed"
)
class TestLightningNNsightTuning(TuningConformance):
    """The Lightning nnsight composition registers the same HF module, so its view runs the full cases too
    (#942)."""

    target = ConformanceTarget(
        composition=("lightning", "nnsight", "sae_lens"), forward_family="hf_native", datamodule_flavour="nnsight"
    )

    def test_view_is_trainable(self, probe):
        assert probe.support().trainable, probe.support().reason
