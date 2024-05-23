import sys
from interpretune.adapters.registration import _register_adapters, CompositionRegistry, AdapterProtocol
from interpretune.adapters.core import CoreAdapter  # noqa: F401
from interpretune.adapters.lightning import LightningAdapter  # noqa: F401
from interpretune.adapters.transformer_lens import TransformerLensAdapter  # noqa: F401
ADAPTER_REGISTRY = CompositionRegistry()
_register_adapters(ADAPTER_REGISTRY, "register_adapter_ctx", sys.modules[__name__], AdapterProtocol)
