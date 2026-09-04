"""Interpretune's distributed conformance suite for adapter compositions.

A repository that ships an adapter runs this against its own composition, through ``ITSession``,
``AnalysisRunner`` and ``AnalysisStore``, with the cases selected by what the composed backends declare at
runtime. See ``docs/adapter_conformance_contract.md`` for the contract and the oracles behind each case.

Minimal consumer::

    # conftest.py
    pytest_plugins = ["interpretune.testing.conformance.plugin"]

    # test_conformance.py
    from interpretune.testing.conformance import ConformanceTarget, ModelBackendConformance

    class TestMyAdapter(ModelBackendConformance):
        target = ConformanceTarget(composition=("core", "my_adapter"), session_cfg_factory=build_session_cfg)

``pytest`` is imported lazily by the case and plugin modules, so importing this package does not require it.
"""

from __future__ import annotations

from interpretune.testing.conformance.gates import Gate, SelectionReport, conformance_case
from interpretune.testing.conformance.inputs import ConformanceInputs, ConformanceTarget

__all__ = [
    "ConformanceInputs",
    "ConformanceTarget",
    "Gate",
    "SelectionReport",
    "conformance_case",
    "ModelBackendConformance",
    "HFReference",
]


def __getattr__(name: str):
    # The case class imports pytest; resolve it on first use so the package stays import-safe without it.
    if name == "ModelBackendConformance":
        from interpretune.testing.conformance.cases import ModelBackendConformance

        return ModelBackendConformance
    if name == "HFReference":
        from interpretune.testing.conformance.reference import HFReference

        return HFReference
    raise AttributeError(name)
