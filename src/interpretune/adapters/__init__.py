import sys
from interpretune.adapters.registration import _register_adapters, CompositionRegistry, AdapterProtocol
from interpretune.adapters.core import CoreAdapter, ITModule
from interpretune.adapters.lightning import LightningAdapter, LightningDataModule, LightningModule
from interpretune.adapters.transformer_lens import (ITLensModule, ITDataModule, BaseITLensModule, TLensAttributeMixin,
                                                    TransformerLensAdapter)
from interpretune.adapters.sae_lens import (SAELensAdapter, SAEAnalysisMixin, SAELensModule, SAELensAttributeMixin,
                                            BaseSAELensModule, InstantiatedSAE)

ADAPTER_REGISTRY = CompositionRegistry()
_register_adapters(ADAPTER_REGISTRY, "register_adapter_ctx", sys.modules[__name__], AdapterProtocol)

__all__ = [
    # Registry
    "ADAPTER_REGISTRY",        # from __init__
    "CompositionRegistry",     # from .registration
    "AdapterProtocol",         # from .registration
    "_register_adapters",      # from .registration

    # Core Adapters
    "CoreAdapter",             # from .core
    "ITModule",                # from .core

    # Lightning Adapters
    "LightningAdapter",        # from .lightning
    "LightningDataModule",     # from .lightning
    "LightningModule",         # from .lightning

    # TransformerLens Adapters
    "TransformerLensAdapter",  # from .transformer_lens
    "ITLensModule",            # from .transformer_lens
    "ITDataModule",            # from .transformer_lens
    "TLensAttributeMixin",     # from .transformer_lens
    "BaseITLensModule",        # from .transformer_lens

    # SAE Lens Adapters
    "SAELensAdapter",          # from .sae_lens
    "SAEAnalysisMixin",        # from .sae_lens
    "SAELensModule",           # from .sae_lens
    "SAELensAttributeMixin",   # from .sae_lens
    "BaseSAELensModule",       # from .sae_lens
    "InstantiatedSAE",         # from .sae_lens
]
