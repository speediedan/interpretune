"""Core as the first consumer of the conformance suite: the bundled compositions that attach a model backend.

This is where the contract is proven before any hub repository sees it. Each class is one target; the
suite selects cases from the live declarations, so a bundled backend that stops declaring something turns
its own row red here rather than only in a downstream repository.
"""

from __future__ import annotations

import pytest

from interpretune.testing.conformance import ConformanceTarget, ModelBackendConformance, OpCollectionConformance
from interpretune.utils.import_utils import package_available

pytest_plugins = ["interpretune.testing.conformance.plugin"]


class TestBridgeConformance(ModelBackendConformance):
    """TransformerBridge over gpt2: executes the HF forward in place, so the value cases apply."""

    target = ConformanceTarget(
        composition=("core", "sae_lens"), forward_family="hf_native", datamodule_flavour="bridge"
    )


@pytest.mark.skipif(not package_available("nnsight"), reason="nnsight is not installed")
class TestNNsightConformance(ModelBackendConformance):
    target = ConformanceTarget(
        composition=("core", "nnsight", "sae_lens"), forward_family="hf_native", datamodule_flavour="nnsight"
    )


class TestConceptCollectionOnBridge(OpCollectionConformance):
    """The bundled `concept` family, validated as a collection against the bridge target."""

    target = ConformanceTarget(
        composition=("core", "sae_lens"), forward_family="hf_native", datamodule_flavour="bridge"
    )
    collection = "concept"


def _legacy_hooked_transformer(inputs):
    """The bridge seed's session config with the weight-converted HookedTransformer instead of the bridge."""

    def _no_bridge(_dm_cfg, it_cfg):
        it_cfg.tl_cfg.use_bridge = False

    return inputs.session_cfg(("core", "sae_lens"), flavour="bridge", prepare=_no_bridge)


class TestWeightConvertedConformance(ModelBackendConformance):
    """The legacy HookedTransformer path (weight conversion, not a bridge over the HF module).

    Its forward is a re-implementation, so the `hf_native` reference cases do not apply and skip by family;
    the causal and structural cases run, and capture names are the same vocabulary spellings the bridge takes.
    """

    target = ConformanceTarget(
        composition=("core", "sae_lens"),
        session_cfg_factory=_legacy_hooked_transformer,
        forward_family="weight_converted",
        datamodule_flavour="bridge",
    )
