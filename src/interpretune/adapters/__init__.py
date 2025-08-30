import sys
import os
from interpretune.adapter_registry import ADAPTER_REGISTRY
from interpretune.adapters.registration import _register_adapters, CompositionRegistry, AdapterProtocol
from interpretune.adapters.core import CoreAdapter, ITModule
from interpretune.adapters.lightning import LightningAdapter, LightningDataModule, LightningModule

# Skip heavy imports during pytest collection to improve performance
_in_pytest_collection = (
    "pytest" in sys.modules
    or "PYTEST_CURRENT_TEST" in os.environ
    or any("pytest" in arg for arg in sys.argv)
    or "--collect-only" in sys.argv
)

# Conditionally import transformer_lens adapter to improve import performance
_transformer_lens_available = False
if not _in_pytest_collection:
    try:
        from interpretune.adapters.transformer_lens import (
            ITLensModule,
            ITDataModule,
            BaseITLensModule,
            TLensAttributeMixin,
            TransformerLensAdapter,
        )

        _transformer_lens_available = True
    except ImportError:
        pass

if not _transformer_lens_available:
    # transformer_lens not available or skipped, define placeholder classes
    ITLensModule = None
    ITDataModule = None
    BaseITLensModule = None
    TLensAttributeMixin = None
    TransformerLensAdapter = None

# Conditionally import sae_lens adapter to improve import performance
_sae_lens_available = False
if not _in_pytest_collection:
    try:
        from interpretune.adapters.sae_lens import (
            SAELensAdapter,
            SAEAnalysisMixin,
            SAELensModule,
            SAELensAttributeMixin,
            BaseSAELensModule,
            InstantiatedSAE,
        )

        _sae_lens_available = True
    except ImportError:
        pass

if not _sae_lens_available:
    # sae_lens not available or skipped, define placeholder classes
    SAELensAdapter = None
    SAEAnalysisMixin = None
    SAELensModule = None
    SAELensAttributeMixin = None
    BaseSAELensModule = None
    InstantiatedSAE = None

# TODO: we can remove this logic once a fork of circuit-tracer is available on PyPI and custom import tools are
# no longer needed
# Conditionally import circuit_tracer adapter
_circuit_tracer_available = False
if not _in_pytest_collection and _transformer_lens_available:  # circuit_tracer depends on transformer_lens
    try:
        from interpretune.adapters.circuit_tracer import (
            CircuitTracerAdapter,
            CircuitTracerAttributeMixin,
            BaseCircuitTracerModule,
        )

        _circuit_tracer_available = True
    except ImportError:
        pass

if not _circuit_tracer_available:
    # circuit_tracer not available or skipped, define placeholder classes
    CircuitTracerAdapter = None
    CircuitTracerAttributeMixin = None
    BaseCircuitTracerModule = None
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
]

# Add transformer_lens adapters only if available
if _transformer_lens_available:
    __all__.extend(
        [
            "TransformerLensAdapter",  # from .transformer_lens
            "ITLensModule",  # from .transformer_lens
            "ITDataModule",  # from .transformer_lens
            "TLensAttributeMixin",  # from .transformer_lens
            "BaseITLensModule",  # from .transformer_lens
        ]
    )

# Add sae_lens adapters only if available
if _sae_lens_available:
    __all__.extend(
        [
            "SAELensAdapter",  # from .sae_lens
            "SAEAnalysisMixin",  # from .sae_lens
            "SAELensModule",  # from .sae_lens
            "SAELensAttributeMixin",  # from .sae_lens
            "BaseSAELensModule",  # from .sae_lens
            "InstantiatedSAE",  # from .sae_lens
        ]
    )

# Add circuit_tracer adapters only if available
if _circuit_tracer_available:
    __all__.extend(
        [
            "CircuitTracerAdapter",  # from .circuit_tracer
            "CircuitTracerAttributeMixin",  # from .circuit_tracer
            "BaseCircuitTracerModule",  # from .circuit_tracer
        ]
    )
