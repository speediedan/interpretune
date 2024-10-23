import operator
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

def _patch_some_dep():
    #from it_examples.patching.my_patch import patched_func
    #target_mod = 'some_dep'
    #sys.modules.get(some_dep).__dict__.get('SomeClass').some_func = patched_func
    pass

some_dep_patch = DependencyPatch(
    condition=(lwt_compare_version("torch", operator.le, "2.5.1"), lwt_compare_version("torch", operator.ge, "2.5.0")),
    env_flag=OSEnvToggle("ENABLE_IT_SOME_DEP_PATCH", default="0"),
    function=_patch_some_dep, patched_package='torch',
    description=("Example patch, would apply to `torch` versions `2.5.0` and `2.5.1`. Only enabled if the the OS env"
                 " variable `ENABLE_IT_SOME_DEP_PATCH` is set to `1` and those conditions are met.")
)

# when adding patches in the future, ensure they're included in this enum
class ExpPatch(Enum):
    PLACEHOLDER = some_dep_patch

_DEFINED_PATCHES = set(ExpPatch)
_ACTIVE_PATCHES = set()

for defined_patch in _DEFINED_PATCHES:
    if all(defined_patch.value.condition) and os.environ.get(defined_patch.value.env_flag.env_var_name,
                                                             defined_patch.value.env_flag.default) == "1":
        defined_patch.value.function()
        _ACTIVE_PATCHES.add(defined_patch)
