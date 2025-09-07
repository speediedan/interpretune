from interpretune.adapters.registration import CompositionRegistry

ADAPTER_REGISTRY = CompositionRegistry()

# Populate registry with known adapter compositions using the lightweight
# registration helper. This performs minimal imports of adapter modules and
# calls their `register_adapter_ctx` methods. Import failures are tolerated
# to avoid pulling heavy optional dependencies at package import time.
try:
    from interpretune.adapters._light_register import register_all_adapters

    register_all_adapters(ADAPTER_REGISTRY)
except Exception:
    # If anything goes wrong during the light registration pass, we don't
    # want to crash import time. Adapters can still register themselves
    # lazily later when their implementation modules are imported.
    pass
