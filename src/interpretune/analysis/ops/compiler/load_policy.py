"""Failure policy for op definition loading.

Op loading is deliberately fail-soft: one malformed hub op must not take down every other op in the
process. That resilience has a cost -- a bundled or local op that silently fails to compile looks
identical to one that was never defined -- so the policy is switchable. Set ``IT_STRICT_OP_LOAD=1``
to turn the warn-and-drop paths into errors; CI runs strict so an in-tree definition can never
regress into a warning nobody reads.
"""

from __future__ import annotations

import os

from interpretune.utils.logging import rank_zero_warn

IT_STRICT_OP_LOAD_ENV_VAR = "IT_STRICT_OP_LOAD"
_AFFIRMATIVE = frozenset({"1", "true", "yes", "y", "on"})


class OpLoadError(RuntimeError):
    """Raised instead of warning when strict op loading is enabled."""


def strict_op_load() -> bool:
    """Whether op-loading failures should raise rather than warn."""
    return os.environ.get(IT_STRICT_OP_LOAD_ENV_VAR, "").strip().lower() in _AFFIRMATIVE


def op_load_failure(message: str) -> None:
    """Report an op-loading failure: raise under strict loading, warn otherwise."""
    if strict_op_load():
        raise OpLoadError(f"{message} (raised because {IT_STRICT_OP_LOAD_ENV_VAR} is enabled)")
    rank_zero_warn(message)


__all__ = ["IT_STRICT_OP_LOAD_ENV_VAR", "OpLoadError", "op_load_failure", "strict_op_load"]
