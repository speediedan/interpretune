import operator
import sys
import os
from enum import Enum
from typing import NamedTuple, Tuple, Callable
from it_examples.patching._patch_utils import lwt_compare_version


class OSEnvToggle(NamedTuple):
    env_var_name: str
    default: str

class DependencyPatch(NamedTuple):
    """Ephemeral dependency patches to conditionally apply to the environment.

    To activate a given patch, all defined `condition` callables must return truthy and the `env_flag` must be set (or
    must default) to '1'
    """
    condition: Tuple[Callable]  # typically a tuple of `lwt_compare_version` to define version dependency
    env_flag: OSEnvToggle  # a tuple defining the environment variable based condition and its default if not set
    function: Callable
    patched_package: str
    description: str


def _dep_patch_repr(self):
    return f'Patch of {self.patched_package}: {self.description})'

DependencyPatch.__repr__ = _dep_patch_repr

# N.B. One needs to ensure they patch all relevant _calling module_ references to patch targets since we usually patch
# after those calling modules have already secured the original (unpatched) references.

def _patch_sae_from_pretrained():
    from it_examples.patching.patched_sae_from_pretrained import from_pretrained
    target_mod = 'sae_lens.sae'
    sys.modules.get(target_mod).__dict__.get('SAE').from_pretrained = from_pretrained


sae_from_pretrained_patch = DependencyPatch(
    condition=(lwt_compare_version("sae_lens", operator.ge, "4.4.1"),),
    env_flag=OSEnvToggle("ENABLE_IT_SAE_FROM_PRETRAINED_PATCH", default="1"),
    function=_patch_sae_from_pretrained, patched_package='sae_lens',
    description=("SAELens `from_pretrained` patch. Only enabled if the the OS env"
                 " variable `ENABLE_IT_SOME_DEP_PATCH` is set to `1` and `sae_lens` >= 4.4.1.")
)

# when adding patches in the future, ensure they're included in this enum
class ExpPatch(Enum):
    SAE_FROM_PRETRAINED = sae_from_pretrained_patch

_DEFINED_PATCHES = set(ExpPatch)
_ACTIVE_PATCHES = set()

for defined_patch in _DEFINED_PATCHES:
    if all(defined_patch.value.condition) and os.environ.get(defined_patch.value.env_flag.env_var_name,
                                                             defined_patch.value.env_flag.default) == "1":
        defined_patch.value.function()
        _ACTIVE_PATCHES.add(defined_patch)
