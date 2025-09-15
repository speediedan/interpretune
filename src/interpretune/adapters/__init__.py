"""Adapters package lazy exports.

This module exposes adapter classes/registries lazily to avoid importing heavy third-party dependencies (e.g.,
transformer_lens, sae_lens) at package import time.
"""

_LAZY_ADAPTER_ATTRS = {
    "ADAPTER_REGISTRY": "interpretune.adapter_registry.ADAPTER_REGISTRY",
    "CompositionRegistry": "interpretune.adapters.registration.CompositionRegistry",
    "AdapterProtocol": "interpretune.adapters.registration.AdapterProtocol",
    "_register_adapters": "interpretune.adapters.registration._register_adapters",
    # Core
    "CoreAdapter": "interpretune.adapters.core.CoreAdapter",
    "ITModule": "interpretune.adapters.core.ITModule",
    # Lightning
    "LightningAdapter": "interpretune.adapters.lightning.LightningAdapter",
    "LightningDataModule": "interpretune.adapters.lightning.LightningDataModule",
    "LightningModule": "interpretune.adapters.lightning.LightningModule",
    # TransformerLens
    "TransformerLensAdapter": "interpretune.adapters.transformer_lens.TransformerLensAdapter",
    "ITLensModule": "interpretune.adapters.transformer_lens.ITLensModule",
    "ITDataModule": "interpretune.adapters.transformer_lens.ITDataModule",
    "TLensAttributeMixin": "interpretune.adapters.transformer_lens.TLensAttributeMixin",
    "BaseITLensModule": "interpretune.adapters.transformer_lens.BaseITLensModule",
    # SAE Lens
    "SAELensAdapter": "interpretune.adapters.sae_lens.SAELensAdapter",
    "SAEAnalysisMixin": "interpretune.adapters.sae_lens.SAEAnalysisMixin",
    "SAELensModule": "interpretune.adapters.sae_lens.SAELensModule",
    "SAELensAttributeMixin": "interpretune.adapters.sae_lens.SAELensAttributeMixin",
    "BaseSAELensModule": "interpretune.adapters.sae_lens.BaseSAELensModule",
    "InstantiatedSAE": "interpretune.adapters.sae_lens.InstantiatedSAE",
}


def __getattr__(name: str):
    if name in _LAZY_ADAPTER_ATTRS:
        module_path = _LAZY_ADAPTER_ATTRS[name]
        module_name, attr = module_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[attr])
        val = getattr(module, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ADAPTER_ATTRS.keys()))
