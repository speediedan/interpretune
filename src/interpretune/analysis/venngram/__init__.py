r"""
venngram Integration
^^^^^^^^^^^^^^^^^^^^

Interpretune's side of venngram's consumer-owned integration (interpretune#887): what this package honours of a
venngram recipe, declared as a ``venngram.ops.Support`` record derived from a backend's own support records, and
the refuse-by-name check a recipe runs against it before anything is applied. Applying a recipe (translating its
treatments into intervention specs) builds on this declaration; nothing here executes a treatment.

Everything in this package needs the optional ``venngram`` extra; importing interpretune without it is unaffected.

.. warning:: This is an experimental API and may change in a future release.
"""

from interpretune.analysis.venngram.support import check_recipe, require_supported, venngram_support

__all__ = [
    "check_recipe",
    "require_supported",
    "venngram_support",
]
