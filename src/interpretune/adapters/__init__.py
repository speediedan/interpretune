"""Adapters package lazy exports.

This module exposes adapter classes/registries lazily to avoid importing heavy third-party dependencies (e.g.,
transformer_lens, sae_lens, nnsight) at package import time.
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
    "TransformerLensAdapter": "interpretune.adapters.transformer_lens.adapter.TransformerLensAdapter",
    "ITLensModule": "interpretune.adapters.transformer_lens.adapter.ITLensModule",
    "ITDataModule": "interpretune.base.ITDataModule",
    "TLensAttributeMixin": "interpretune.adapters.transformer_lens.adapter.TLensAttributeMixin",
    "BaseITLensModule": "interpretune.adapters.transformer_lens.adapter.BaseITLensModule",
    # SAE Lens
    "SAELensAdapter": "interpretune.adapters.sae_lens.adapter.SAELensAdapter",
    "SAELensAnalysisMixin": "interpretune.adapters.sae_lens.adapter.SAELensAnalysisMixin",
    "SAELensTLModule": "interpretune.adapters.sae_lens.adapter.SAELensTLModule",
    "SAELensNNsightModule": "interpretune.adapters.sae_lens.adapter.SAELensNNsightModule",
    "SAELensTLModuleMixin": "interpretune.adapters.sae_lens.adapter.SAELensTLModuleMixin",
    "SAELensNNsightModuleMixin": "interpretune.adapters.sae_lens.adapter.SAELensNNsightModuleMixin",
    "SAELensAttributeMixin": "interpretune.adapters.sae_lens.adapter.SAELensAttributeMixin",
    "BaseSAELensModule": "interpretune.adapters.sae_lens.adapter.BaseSAELensModule",
    "InstantiatedSAE": "interpretune.adapters.sae_lens.adapter.InstantiatedSAE",
    # NNsight
    "NNsightAdapter": "interpretune.adapters.nnsight.adapter.NNsightAdapter",
    "NNsightModule": "interpretune.adapters.nnsight.adapter.NNsightModule",
    "NNsightAttributeMixin": "interpretune.adapters.nnsight.adapter.NNsightAttributeMixin",
    "BaseNNsightModule": "interpretune.adapters.nnsight.adapter.BaseNNsightModule",
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
