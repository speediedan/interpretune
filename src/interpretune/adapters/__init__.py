import sys
from interpretune.adapter_registry import ADAPTER_REGISTRY
from interpretune.adapters.registration import _register_adapters, CompositionRegistry, AdapterProtocol
from interpretune.adapters.core import CoreAdapter, ITModule
from interpretune.adapters.lightning import LightningAdapter, LightningDataModule, LightningModule
from interpretune.adapters.transformer_lens import (
    ITLensModule,
    ITDataModule,
    BaseITLensModule,
    TLensAttributeMixin,
    TransformerLensAdapter,
)
from interpretune.adapters.sae_lens import (
    SAELensAdapter,
    SAEAnalysisMixin,
    SAELensModule,
    SAELensAttributeMixin,
    BaseSAELensModule,
    InstantiatedSAE,
)

# TODO: we can remove this logic once a fork of circuit-tracer is available on PyPI and custom import tools are
# no longer needed
# Conditionally import circuit_tracer adapter
try:
    from interpretune.adapters.circuit_tracer import (
        CircuitTracerAdapter,
        CircuitTracerAttributeMixin,
        BaseCircuitTracerModule,
    )

    _circuit_tracer_available = True
except ImportError:
    # circuit_tracer not available, define placeholder classes to avoid import errors
    CircuitTracerAdapter = None
    CircuitTracerAttributeMixin = None
    BaseCircuitTracerModule = None
    _circuit_tracer_available = False
_register_adapters(ADAPTER_REGISTRY, "register_adapter_ctx", sys.modules[__name__], AdapterProtocol)

__all__ = [
    # Registry
    "ADAPTER_REGISTRY",  # from __init__
    "CompositionRegistry",  # from .registration
    "AdapterProtocol",  # from .registration
    "_register_adapters",  # from .registration
    # Core Adapters
    "CoreAdapter",  # from .core
    "ITModule",  # from .core
    # Lightning Adapters
    "LightningAdapter",  # from .lightning
    "LightningDataModule",  # from .lightning
    "LightningModule",  # from .lightning
    # TransformerLens Adapters
    "TransformerLensAdapter",  # from .transformer_lens
    "ITLensModule",  # from .transformer_lens
    "ITDataModule",  # from .transformer_lens
    "TLensAttributeMixin",  # from .transformer_lens
    "BaseITLensModule",  # from .transformer_lens
    # SAE Lens Adapters
    "SAELensAdapter",  # from .sae_lens
    "SAEAnalysisMixin",  # from .sae_lens
    "SAELensModule",  # from .sae_lens
    "SAELensAttributeMixin",  # from .sae_lens
    "BaseSAELensModule",  # from .sae_lens
    "InstantiatedSAE",  # from .sae_lens
]

# Add circuit_tracer adapters only if available
if _circuit_tracer_available:
    __all__.extend(
        [
            "CircuitTracerAdapter",  # from .circuit_tracer
            "CircuitTracerAttributeMixin",  # from .circuit_tracer
            "BaseCircuitTracerModule",  # from .circuit_tracer
        ]
    )
