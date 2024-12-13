from interpretune.utils.import_utils import module_available

_SL_AVAILABLE = module_available("sae_lens")

from it_examples.patching.dep_patch_shim import _ACTIVE_PATCHES  # noqa: E402, F401
