r"""
The venngram Support Declaration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A venngram consumer declares what it honours as a ``venngram.ops.Support`` record and checks a recipe against it
with ``venngram.ops.check_support`` before running anything, so a recipe asking for more is refused by name ahead
of time rather than narrowed silently at run time. Interpretune's declaration is DERIVED from the records a
backend already publishes, never restated by hand:

- **ops** from :class:`~interpretune.analysis.backends.capabilities.InterventionSupport`'s modes, intersected
  with the ops this integration's apply path implements (today: unit-targeted ``add`` only). The mode names are
  shared with venngram's op names for every mode but ``scale``, which interpretune has no mode for.
- **token scopes** from the same record's position scopes: ``last_token`` is venngram's ``last`` and
  ``all_positions`` its ``all``. The remaining scopes (``prompt``, ``generated``, ``positions``, ``slice``) have
  no interpretune spelling yet and are not declared.
- **sites** from :class:`~interpretune.analysis.backends.capabilities.CaptureSupport`'s capturable points, which
  are layer-free base spellings like a recipe's sites. A site the backend cannot capture on this model is a site
  a treatment cannot be verified at, so it is not declared reachable. Notably, recipes name ``mlp.out.hook_in``
  (the MLP hidden activation's input), which interpretune's point vocabulary has no point for yet: it is refused
  by name until that point exists, rather than guessed at.
- **targets and values**: unit targets with ``dose``/``doses`` values, what the apply path implements today.
  Feature targets, evidence-sourced values and transforms are refused by name.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from interpretune.analysis.backends.capabilities import InterventionSupport, PositionScope
from interpretune.utils import MisconfigurationException

if TYPE_CHECKING:
    from interpretune.analysis.backends.capabilities import CaptureSupport

#: Interpretune position scopes by their venngram token-scope names.
_TOKEN_SCOPES = {PositionScope.LAST_TOKEN: "last", PositionScope.ALL_POSITIONS: "all"}

#: The ops this integration's apply path implements. A backend mode outside this set is honoured by the backend
#: through interpretune's own intervention specs, but a venngram recipe naming it is refused until it is mapped.
_APPLIED_OPS = frozenset({"add"})


def _require_venngram():
    try:
        import venngram.ops
    except ImportError as e:
        raise MisconfigurationException(
            "The venngram integration needs the optional `venngram` extra: install `interpretune[venngram]`."
        ) from e
    return venngram.ops


def venngram_support(
    intervention: InterventionSupport,
    capture: CaptureSupport | None = None,
    *,
    ops: frozenset[str] = _APPLIED_OPS,
    name: str = "interpretune",
) -> Any:
    """Derive a ``venngram.ops.Support`` from a backend's own support records (see the module docstring).

    Args:
        intervention: The backend's declared intervention modes and position scopes.
        capture: The backend's capture declaration for the model at hand; its capturable points bound the
            reachable sites. Without one, no site bound is declared (any site passes this check, and site
            resolution is left to the apply path).
        ops: The recipe ops to declare, intersected with what the backend's modes name. Widening this is how a
            newly mapped op enters the declaration; it must only name ops the apply path implements.
        name: The consumer name the record carries.

    Returns:
        A ``venngram.ops.Support``. Typed ``Any`` so importing this module needs no venngram install.

    Raises:
        MisconfigurationException: If venngram is not installed.
    """
    vg_ops = _require_venngram()
    modes = {m.value for m in intervention.modes}
    return vg_ops.Support(
        ops=frozenset(ops & modes & set(vg_ops.OPS)),
        targets=frozenset({"units"}),
        tokens=frozenset(_TOKEN_SCOPES[s] for s in intervention.position_scopes),
        values=frozenset({"dose", "doses"}),
        transforms=frozenset(),
        sites=None if capture is None else frozenset(capture.capturable),
        name=name,
    )


def check_recipe(recipe: Any, support: Any) -> list[str]:
    """Every way ``recipe`` (a ``venngram.recipe.Recipe``) asks for something interpretune does not honour.

    Venngram's own ``check_support`` covers the declared record (ops, targets, token scopes, value sources,
    transforms, bases, reachable sites). This adds the refusals the record cannot express: an op carrying an
    operand (the apply path implements unit doses, not operand arithmetic yet).
    """
    vg_ops = _require_venngram()
    refusals = list(vg_ops.check_support(recipe, support))
    for i, stage in enumerate(recipe.stages):
        for j, op in enumerate(stage.ops):
            if op.operand is not None:
                refusals.append(
                    f"stage {i} ({stage.site}) op {j} (`{op.op}`): operand `{op.operand}` is not supported yet"
                    " (unit doses only)"
                )
    return refusals


def require_supported(recipe: Any, support: Any) -> None:
    """Refuse a recipe this consumer cannot honour in full, by name, before anything runs.

    Raises:
        MisconfigurationException: Listing every refusal :func:`check_recipe` finds.
    """
    if refusals := check_recipe(recipe, support):
        raise MisconfigurationException(
            "This recipe asks for more than interpretune honours:\n" + "\n".join(f"- {r}" for r in refusals)
        )
