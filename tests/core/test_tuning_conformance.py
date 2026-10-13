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
