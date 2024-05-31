import operator
from typing import NamedTuple, Tuple, Callable
from it_examples.patching._patch_utils import lwt_compare_version


class DependencyPatch(NamedTuple):
    """Ephemeral dependency patches to conditionally apply to the environment."""
    condition: Tuple[Callable]
    function: Callable
    patched_package: str
    description: str


def _dep_patch_repr(self):
    return f'Patch of {self.patched_package}: {self.description})'

DependencyPatch.__repr__ = _dep_patch_repr


def _patch_some_dep():
    #from it_examples.patching.my_patch import patched_func
    #target_mod = 'some_dep'
    #sys.modules.get(some_dep).__dict__.get('SomeClass').some_func = patched_func
    pass

some_dep_patch = DependencyPatch((lwt_compare_version("finetuning-scheduler", operator.le, "2.4.0"),),
                                     _patch_some_dep, 'finetuning-scheduler',
                                     'example patch, would apply to the given version finetuning-scheduler package.')

# if adding patches in the future, add them to the DEFINED_PATCHES set here
#   _DEFINED_PATCHES = {some_dep_patch}
_DEFINED_PATCHES = {}
_ACTIVE_PATCHES = set()

for defined_patch in _DEFINED_PATCHES:
    if all(defined_patch.condition):
        defined_patch.function()
        _ACTIVE_PATCHES.add(defined_patch)
