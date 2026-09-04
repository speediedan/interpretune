"""Core as the first consumer of the conformance suite: the bundled compositions that attach a model backend.

This is where the contract is proven before any hub repository sees it. Each class is one target; the
suite selects cases from the live declarations, so a bundled backend that stops declaring something turns
its own row red here rather than only in a downstream repository.
"""

from __future__ import annotations

import pytest

from interpretune.testing.conformance import ConformanceTarget, ModelBackendConformance
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
