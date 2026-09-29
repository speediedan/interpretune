"""The circuit-tracer composition with sae_lens latent models: ``(core, nnsight, circuit_tracer, sae_lens)``.

This lives apart from ``adapter.py`` because it imports sae_lens, which the other circuit-tracer compositions do
not need. Kept in ``adapter.py``, it made the whole adapter unimportable without the sae_lens extra, so every
circuit-tracer composition vanished from an environment that lacked it, not only this one.
``CircuitTracerAdapter.register_adapter_ctx`` imports this module only when sae_lens is installed.
"""

from __future__ import annotations

from interpretune.adapters import (
    BaseNNsightModule,
    BaseSAELensModule,
    CompositionRegistry,
    SAELensAdapter,
    SAELensAnalysisMixin,
)
from interpretune.adapters.circuit_tracer.adapter import (
    BaseCircuitTracerModule,
    CircuitTracerAdapter,
    CircuitTracerAnalysisMixin,
    CircuitTracerNNsightModuleMixin,
)
from interpretune.base import CoreHelperAttributes, ITDataModule
from interpretune.config import CircuitTracerConfig, SAELensConfig
from interpretune.protocol import Adapter


class CircuitTracerNNsightSAELensModule(
    CircuitTracerNNsightModuleMixin,
    CircuitTracerAnalysisMixin,
    SAELensAnalysisMixin,
    CircuitTracerAdapter,
    SAELensAdapter,
    CoreHelperAttributes,
    BaseCircuitTracerModule,
    BaseSAELensModule,
    BaseNNsightModule,
):
    """Circuit-tracer attribution over NNsight with attachable sae_lens latent models.

    The circuit-tracer side owns model init (the NNSightReplacementModel wins by MRO, and the
    attach-don't-override seam in ``BaseCircuitTracerModule`` leaves an already-attached model
    backend alone), while the sae_lens side contributes only what the latent-model cases need:
    the ``sae_handles`` properties and ``instantiate_saes``. The SAE weights are model-independent,
    so they load after the replacement model without splicing anything into it.
    """

    def auto_model_init(self) -> None:
        """Init the replacement model, then load the attached latent models."""
        super().auto_model_init()  # the circuit-tracer NNsight init wins by MRO
        self.instantiate_saes()  # type: ignore[attr-defined]  # from BaseSAELensModule


def register_sae_lens_compositions(adapter_ctx_registry: CompositionRegistry) -> None:
    """Register the ``(core, nnsight, circuit_tracer, sae_lens)`` datamodule, module and config compositions."""
    combination = (Adapter.core, Adapter.nnsight, Adapter.circuit_tracer, Adapter.sae_lens)
    adapter_ctx_registry.register(
        Adapter.circuit_tracer,
        component_key="datamodule",
        adapter_combination=combination,  # type: ignore[arg-type]
        composition_classes=(ITDataModule,),
        description="Circuit Tracer adapter with NNsight backend and SAE latent models...",
    )
    adapter_ctx_registry.register(
        Adapter.circuit_tracer,
        component_key="module",
        adapter_combination=combination,  # type: ignore[arg-type]
        composition_classes=(CircuitTracerNNsightSAELensModule,),
        description="Circuit Tracer adapter with NNsight backend and SAE latent models...",
    )
    adapter_ctx_registry.register(
        Adapter.circuit_tracer,
        component_key="module_cfg",
        adapter_combination=combination,  # type: ignore[arg-type]
        composition_classes=(CircuitTracerConfig, SAELensConfig),
        description="Circuit Tracer configuration with NNsight backend and SAE latent models...",
    )
