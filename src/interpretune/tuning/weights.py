# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Weight-space interventions: venngram ops on parameters, materialized on the tensors they address (#960, #961).

An intervention applies an op to a target. Where the target is an activation (:mod:`interpretune.tuning.interventions`,
the analysis ops), the op runs in a hook on every pass and may be token-scoped. Where it is a **parameter**, the op is
*materialized*: applied to the tensor itself, so it holds on every token and persists into an export. The ops,
operands, addresses and roles are the same in both spaces:

- :func:`apply_weight_patch` applies a venngram weight patch (``kind: weight_patch``, :mod:`venngram.patches`) to the
  tensor its ``target`` addresses, after checking that the tensor's hash is the patch's ``base_sha256``;
- :func:`apply_recipe_weights` applies a recipe's ``weights`` section (venngram recipe schema 3): ``scale``, ``add`` and
  ``replace`` on a tensor or on its units, and ``reject`` and ``project`` of a direction on its output side;
- :func:`weight_edit_roles`, the roles to declare for each edit's operand: ``adapter:<name>`` for a trainable one
  (a LoRA is a trainable low-rank ``add`` on a weight), ``fitted`` for one computed in the run (checkpointed, never
  optimized), or ``instrument`` for one loaded from a published artifact (never checkpointed: it is reloaded from its
  source). Every interpretune module declares them through its cooperative ``tunable_roles()``.

An edit is reversible by default: it is a ``torch.nn.utils.parametrize`` parametrization, so the base tensor is kept
(as ``<module>.parametrizations.<leaf>.original``, which venngram's views keep addressing by its own name), the edit is
a module the gradient reaches, and :meth:`WeightEditHandle.remove` restores the base exactly. With
``reversible=False`` the edited value is written into the tensor instead, for an export. Addresses resolve through the
module's tunable view (or its default view), before anything is parametrized, so they name the tensors as the model
holds them. Addressing a tensor's *units* needs the view's structure to give the tensor's unit axis: venngram's
``MappedView`` over a component map that records each row's ``layout`` (component map schema 5).

At token scope ``all`` several weight edits equal an activation op, because a matrix's input and output are activation
sites. For $y = W x + b$:

- scaling column $j$ of $W$ by $k$ is ``scale`` of unit $j$ of $x$ by $k$;
- the bias patch $b' = b + d W_{:,j}$ is ``add`` of dose $d$ to unit $j$ of $x$;
- $W' = (I - \alpha u u^{\top}) W$ is ``reject`` of $u$ with fraction $\alpha$ from $W x$, at the output.

:func:`weight_form` materializes a recipe's unit ops at ``mlp.out``'s input in these weight forms (:func:`bias_patch`
builds the second), and :func:`check_weight_equivalence` compares the two forms of an edit on a batch. A token-scoped
op has no weight form, and a general low-rank or dense patch has no activation form at one site.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize

#: The roles an edit's operand may take (see the module docstring).
EDIT_ROLES = ("instrument", "fitted", "adapter")

#: The ops :func:`apply_recipe_weights` materializes: every op venngram allows on a parameter target.
WEIGHT_OPS = frozenset({"replace", "add", "scale", "project", "reject"})

_NAME_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _require_venngram() -> None:
    try:
        import venngram  # noqa: F401
    except ImportError as err:  # pragma: no cover - exercised only without the venngram extra
        raise ImportError("Weight-space interventions need venngram, which is not installed.") from err


class WeightEdit(nn.Module):
    """A parametrization editing one tensor: ``forward(W)`` is the edited tensor.

    Its operands are parameters, trainable only when its role is ``adapter``, so a schedule addresses them by role.
    """

    def __init__(self, name: str, role: str) -> None:
        super().__init__()
        self.edit_name = name
        self.role = role

    def operand(self, value: torch.Tensor) -> nn.Parameter:
        """``value`` as one of this edit's operands: trainable only for an ``adapter``."""
        return nn.Parameter(value.detach().clone(), requires_grad=self.role == "adapter")

    def declared_role(self) -> str:
        """The role its operands are declared with (``adapter:<name>`` for a trainable edit)."""
        return f"adapter:{self.edit_name}" if self.role == "adapter" else self.role


class _Delta(WeightEdit):
    """``add`` (:math:`W + \\Delta`) or ``replace`` of a weight patch's delta, in its stored form."""

    indices: torch.Tensor

    def __init__(self, name: str, role: str, op: str, form: str, tensors: Mapping[str, torch.Tensor]) -> None:
        super().__init__(name, role)
        self.op, self.form = op, form
        if form == "sparse":
            self.register_buffer("indices", tensors["indices"].to(torch.long))
            self.values = self.operand(tensors["values"])
        elif form == "low_rank":
            self.u, self.v = self.operand(tensors["u"]), self.operand(tensors["v"])
        else:
            self.delta = self.operand(tensors["delta"])

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.form == "sparse":
            flat = w.reshape(-1)
            values = self.values.to(w.dtype)
            if self.op == "add":
                return flat.index_add(0, self.indices, values).reshape(w.shape)
            return flat.index_put((self.indices,), values).reshape(w.shape)
        delta = (self.u @ self.v.T if self.form == "low_rank" else self.delta).to(w.dtype)
        return w + delta if self.op == "add" else delta.expand_as(w).clone()


class _Units(WeightEdit):
    """``scale``, ``add`` or ``replace`` with a value per unit, along the tensor's unit axis ``dim`` (the whole
    tensor when ``dim`` is ``None``)."""

    units: torch.Tensor

    def __init__(self, name: str, role: str, op: str, dim: int | None, units: Sequence[int], doses: Sequence[float]):
        super().__init__(name, role)
        self.op, self.dim = op, dim
        self.register_buffer("units", torch.tensor(list(units), dtype=torch.long))
        self.doses = self.operand(torch.tensor(list(doses), dtype=torch.float32))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        doses = self.doses.to(w.dtype)
        if self.dim is None:
            (dose,) = doses
            return w * dose if self.op == "scale" else (w + dose if self.op == "add" else torch.full_like(w, 1) * dose)
        shape = [1] * w.ndim
        shape[self.dim] = w.shape[self.dim]
        if self.op == "scale":
            factor = torch.ones(w.shape[self.dim], dtype=w.dtype, device=w.device).index_copy(0, self.units, doses)
            return w * factor.reshape(shape)
        bump = torch.zeros(w.shape[self.dim], dtype=w.dtype, device=w.device).index_copy(0, self.units, doses)
        if self.op == "add":
            return w + bump.reshape(shape)
        keep = torch.ones(w.shape[self.dim], dtype=w.dtype, device=w.device).index_fill(0, self.units, 0.0)
        return w * keep.reshape(shape) + bump.reshape(shape)


class _Vector(WeightEdit):
    """``add`` or ``replace`` of an operand of the tensor's own shape."""

    def __init__(self, name: str, role: str, op: str, value: torch.Tensor) -> None:
        super().__init__(name, role)
        self.op = op
        self.value = self.operand(value)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        value = self.value.to(w.dtype)
        return w + value if self.op == "add" else value.clone()


class _Span(WeightEdit):
    """``reject`` (:math:`W - \\alpha P W`) or ``project`` (:math:`\\alpha P W`) of the span of a basis on the tensor's
    output side ``dim``, with :math:`P = V V^{+}` as the activation ops compute it (the basis need not be
    orthonormal)."""

    def __init__(self, name: str, role: str, op: str, basis: torch.Tensor, fraction: float, dim: int) -> None:
        super().__init__(name, role)
        self.op, self.fraction, self.dim = op, float(fraction), dim
        self.basis = self.operand(basis.reshape(-1, basis.shape[-1]))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        v = self.basis.to(torch.float32).T  # (d, k), the basis vectors as columns
        p = v @ torch.linalg.pinv(v)
        out_first = w.to(torch.float32) if self.dim == 0 else w.to(torch.float32).T
        in_span = p @ out_first
        edited = out_first - self.fraction * in_span if self.op == "reject" else self.fraction * in_span
        return (edited if self.dim == 0 else edited.T).to(w.dtype)


@dataclass(frozen=True)
class WeightEditHandle:
    """One edit applied to one tensor: where it is, and how to take it back."""

    target: str
    """The tensor's name under the root, as the model held it before the edit."""
    edit: WeightEdit
    owner: nn.Module
    leaf: str
    reversible: bool

    def remove(self) -> None:
        """Take the edit back, restoring the tensor to its value before it (edits on one tensor come off in reverse
        order of application; taking an earlier one off first changes what the later ones edit).

        Raises:
            ValueError: The edit was applied with ``reversible=False``, or is no longer applied.
        """
        if not self.reversible:
            raise ValueError(
                f"The edit {self.edit.edit_name!r} on `{self.target}` was written into the tensor "
                "(`reversible=False`) and cannot be taken back"
            )
        held = getattr(getattr(self.owner, "parametrizations", None), self.leaf, None)
        index = next((i for i, m in enumerate(held or ()) if m is self.edit), None)
        if held is None or index is None:
            raise ValueError(f"The edit {self.edit.edit_name!r} is no longer applied to `{self.target}`")
        if len(held) == 1:
            parametrize.remove_parametrizations(self.owner, self.leaf, leave_parametrized=False)
        else:
            del held[index]


def weight_edit_roles(root: nn.Module) -> dict[str, str]:
    """The role of every applied edit's operands, by their names under ``root``."""
    roles: dict[str, str] = {}
    for path, module in root.named_modules():
        if isinstance(module, WeightEdit):
            role = module.declared_role()
            roles.update({f"{path}.{n}": role for n, _ in module.named_parameters(recurse=False)})
    return roles


def _view(root: nn.Module, view: Any | None) -> Any:
    if view is not None:
        return view
    from interpretune.tuning.views import VIEW_ATTR, default_view

    attached = root.__dict__.get(VIEW_ATTR)
    return attached if attached is not None else default_view(root)


_PARAMETRIZED = ".parametrizations."


def _held_names(root: nn.Module) -> list[str]:
    """``root``'s tensors by the names the model holds them under: a parametrized tensor by its own name (not its
    ``parametrizations.<leaf>.original``), and no edit's operands, so an address names the same tensor before and
    after an edit."""
    out = []
    for name, _ in root.named_parameters():
        owner, sep, rest = name.partition(_PARAMETRIZED)
        if not sep:
            out.append(name)
        elif rest.endswith(".original"):
            out.append(f"{owner}.{rest.removesuffix('.original')}")
    return out


def _tensor(root: nn.Module, name: str) -> torch.Tensor:
    """The tensor ``name`` (a held name) as the model computes it: edited, where edits are applied."""
    owner, _, leaf = name.rpartition(".")
    return getattr(root.get_submodule(owner), leaf)


def _space(root: nn.Module, view: Any) -> Any:
    """What addresses resolve against: ``root``'s tensors by physical name (and the view's logical names), their roles
    and the view's structure, as finetuning-scheduler resolves a schedule's addresses."""
    from venngram.resolve import ParamSpace

    names = _held_names(root)
    aliases: dict[str, tuple[str, ...]] = {}
    if view.support().naming_scheme != "physical":
        logical = [n for n, _ in view.logical_named_parameters()]
        for name, physical in view.to_physical(logical).items():
            for p in physical:
                aliases[p] = aliases.get(p, ()) + (name,)
    return ParamSpace(names, roles=dict(view.roles()), aliases=aliases, structure=view.structure())


def _leaf_terms(sets: Mapping[str, Any], name: str) -> list[Any]:
    from venngram.address import references, terms

    out, seen, todo = [], set(), [name]
    while todo:
        n = todo.pop()
        if n in seen or n not in sets:
            continue
        seen.add(n)
        out += terms(sets[n])
        todo += references(sets[n])
    return out


def resolve_weight_targets(
    root: nn.Module, address: Any, *, sets: Mapping[str, Any] | None = None, view: Any | None = None
) -> list[str]:
    """The names of ``root``'s tensors that ``address`` selects: a venngram address (``component:mlp.out &
    layers:13 & leaf:weight``), a reference to one of ``sets`` (``@down``), or a tensor's own name.

    Raises:
        ValueError: If it selects nothing (venngram's refusal, with every reason, is the cause).
    """
    _require_venngram()
    from venngram.errors import VenngramError
    from venngram.resolve import select

    if isinstance(address, str) and address in _held_names(root):
        return [address]
    try:
        return select(address, _space(root, _view(root, view)), sets)
    except VenngramError as err:
        raise ValueError(f"The weight target {address!r} does not resolve on {type(root).__name__}: {err}") from err


def _unit_dim(structure: Any, name: str, ndim: int) -> int:
    unit_dim = getattr(structure, "unit_dim", None)
    if unit_dim is None:
        raise ValueError(
            f"`{name}`'s unit axis is unknown: its view's structure ({type(structure).__name__}) gives "
            "none. Units of a tensor are addressed through a venngram MappedView whose component map "
            "records each row's `layout` (component map schema 5)."
        )
    found = unit_dim(name, ndim)
    if not isinstance(found, int):
        raise ValueError(f"`{name}` has no unit axis: {found}")
    return found


def _out_dim(structure: Any, name: str, w: torch.Tensor, width: int) -> int:
    """The dimension of ``w`` on its output side, for a basis of ``width``: by shape, or for a square matrix from
    the unit axis and the component's unit side."""
    matches = [d for d in range(w.ndim) if w.shape[d] == width]
    if w.ndim != 2 or not matches:
        raise ValueError(f"A basis of width {width} lies on no side of `{name}` (shape {list(w.shape)})")
    if len(matches) == 1:
        return matches[0]
    from venngram.maps.schema import UNIT_SIDES

    loc = structure.locate(name) if structure is not None else None
    side = UNIT_SIDES.get(getattr(loc, "component", None) or "")
    if side is None:
        raise ValueError(
            f"`{name}` is square, so its output side is known only from its component's layout, and it "
            "has none (address it through a MappedView whose map records layouts)"
        )
    dim = _unit_dim(structure, name, 2)
    return dim if side == "out" else 1 - dim


def _register(root: nn.Module, target: str, edit: WeightEdit, reversible: bool) -> WeightEditHandle:
    owner_name, _, leaf = target.rpartition(".")
    owner = root.get_submodule(owner_name)
    base = getattr(owner, leaf)
    edit.to(device=base.device)
    if reversible:
        parametrize.register_parametrization(owner, leaf, edit)
    else:
        if edit.role == "adapter":
            raise ValueError(
                f"A trainable edit ({edit.edit_name!r}) is reversible: written into `{target}` it would "
                "have no operand to train"
            )
        if parametrize.is_parametrized(owner, leaf):
            raise ValueError(
                f"`{target}` carries reversible edits; take them back (or write them in) before writing an edit into it"
            )
        with torch.no_grad():
            base.copy_(edit(base))
    return WeightEditHandle(target=target, edit=edit, owner=owner, leaf=leaf, reversible=reversible)


def _check_role(name: str, role: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"Edit name {name!r} must be letters, digits, '_' and '-' only")
    if role not in EDIT_ROLES:
        raise ValueError(f"An edit's role is one of {list(EDIT_ROLES)}, not {role!r}")


def _warn_if_viewed(root: nn.Module, role: str) -> None:
    from interpretune.tuning.views import VIEW_ATTR

    if role == "adapter" and root.__dict__.get(VIEW_ATTR) is not None:
        warnings.warn(
            f"{type(root).__name__} already has a tunable view, built before this trainable edit: the view's roles do "
            "not include the edit's operand. Apply trainable edits before the view is attached (as interventions are "
            "attached once the model is built).",
            stacklevel=3,
        )


def apply_weight_patch(
    root: nn.Module,
    patch: Any,
    *,
    op: str = "add",
    name: str | None = None,
    role: str = "instrument",
    reversible: bool = True,
    view: Any | None = None,
    sets: Mapping[str, Any] | None = None,
) -> WeightEditHandle:
    """Apply a venngram weight patch to the tensor its ``target`` addresses.

    Args:
        root: The module the target resolves under (an interpretune module is its view's root).
        patch: A ``venngram.patches.WeightPatch``, or the path of its document.
        op: ``add`` (:math:`W + \\Delta`) or ``replace``.
        name: The edit's name (letters, digits, ``_`` and ``-``); the patch's name by default, with other characters
            replaced by ``_``.
        role: The operand's role: ``instrument`` (loaded from an artifact, the default), ``fitted`` (computed in the
            run), or ``adapter`` (trainable, declared ``adapter:<name>``).
        reversible: Parametrize the tensor (the default), or write the edited value into it (for an export).
        view: The view addresses resolve through; ``root``'s attached view, else its default view.
        sets: Named sets a reference in the target resolves against.

    Returns:
        A :class:`WeightEditHandle`.

    Raises:
        ValueError: If the op is not ``add`` or ``replace``, the target does not select exactly one tensor, that
            tensor's shape is not the patch's, or its hash is not the patch's ``base_sha256`` (the patch was made for
            another base).
    """
    _require_venngram()
    from venngram.patches import read_weight_patch, tensor_sha256

    if not hasattr(patch, "tensors"):
        patch = read_weight_patch(patch)
    if op not in ("add", "replace"):
        raise ValueError(f"A weight patch is applied with `add` or `replace`, not `{op}`")
    name = name or re.sub(r"[^A-Za-z0-9_\-]", "_", patch.name)
    _check_role(name, role)
    selected = resolve_weight_targets(root, patch.target, sets=sets, view=view)
    if len(selected) != 1:
        raise ValueError(f"The patch {patch.name!r} targets one tensor; {patch.target!r} selects {selected}")
    (target,) = selected
    base = _tensor(root, target)
    if tuple(base.shape) != tuple(patch.shape):
        raise ValueError(
            f"The patch {patch.name!r} is for a tensor of shape {list(patch.shape)}; `{target}` is {list(base.shape)}"
        )
    if (found := tensor_sha256(base)) != patch.base_sha256:
        raise ValueError(
            f"The patch {patch.name!r} was made for another base: `{target}` hashes to {found}, the patch "
            f"names {patch.base_sha256}"
        )
    tensors = {k: torch.tensor(values).reshape(shape) for k, (values, shape) in patch.tensors().items()}
    _warn_if_viewed(root, role)
    handle = _register(root, target, _Delta(name, role, op, patch.form, tensors), reversible)
    if op == "add" and patch.edited_sha256 is not None:
        edited = tensor_sha256(getattr(handle.owner, handle.leaf))
        if edited != patch.edited_sha256:
            warnings.warn(
                f"`{target}` after the patch {patch.name!r} hashes to {edited}, not its `edited_sha256` "
                f"({patch.edited_sha256}); the difference is in the order the delta was summed if the patch "
                "is otherwise right",
                stacklevel=2,
            )
    return handle


def _tensors_of(root: nn.Module, recipe: Any, op: Any, view: Any) -> list[str]:
    sets = recipe.book.sets()
    if op.target_kind(recipe.book) == "param":
        return resolve_weight_targets(root, f"@{op.target}", sets=sets, view=view)
    # A unit set names tensors through its leaf terms (its algebra composes units, not tensors): each term without its
    # units selects the tensors, and the op's unit doses say which units of each.
    out: list[str] = []
    for term in _leaf_terms(sets, op.target):
        for name in resolve_weight_targets(root, replace(term, units=None), sets=sets, view=view):
            if name not in out:
                out.append(name)
    return out


def _edit_for(root: nn.Module, recipe: Any, op: Any, target: str, structure: Any, name: str, role: str) -> WeightEdit:
    w = _tensor(root, target)
    kind = op.target_kind(recipe.book)
    operand = recipe.operands.get(op.operand) if op.operand is not None else None
    base_dir = recipe.base_dir or "."
    if op.op in ("reject", "project"):
        if operand is None:
            raise ValueError(f"`{op.op}` on `{target}` needs a basis operand")
        values, shape = operand.load(base_dir)
        basis = torch.tensor(values, dtype=torch.float32).reshape(shape)
        dim = _out_dim(structure, target, w, basis.shape[-1])
        return _Span(name, role, op.op, basis, op.params.get("fraction", 1.0), dim)
    if operand is not None and operand.kind == "delta":
        from venngram.patches import tensor_sha256

        patch = operand.weight_patch(base_dir)
        if tuple(w.shape) != tuple(patch.shape):
            raise ValueError(
                f"operand `{op.operand}` is a patch for shape {list(patch.shape)}; `{target}` is {list(w.shape)}"
            )
        if (found := tensor_sha256(w)) != patch.base_sha256:
            raise ValueError(f"operand `{op.operand}` was made for another base: `{target}` hashes to {found}")
        tensors = {k: torch.tensor(v).reshape(s) for k, (v, s) in patch.tensors().items()}
        return _Delta(name, role, op.op, patch.form, tensors)
    if operand is not None:
        values, shape = operand.load(base_dir)
        value = torch.tensor(values, dtype=torch.float32).reshape(shape)
        if tuple(value.shape) != tuple(w.shape):
            raise ValueError(
                f"operand `{op.operand}` has shape {list(value.shape)}; `{target}` is {list(w.shape)}, "
                "and a parameter op's vector operand is of the tensor's own shape"
            )
        return _Vector(name, role, op.op, value)
    if kind == "param":
        if op.value is None or op.value.source != "dose":
            raise ValueError(f"`{op.op}` on the whole of `{target}` takes a `dose`")
        return _Units(name, role, op.op, None, (), (op.value.dose,))
    dim = _unit_dim(structure, target, w.ndim)
    layer = structure.locate(target).layer
    doses = {unit: dose for (lyr, unit), dose in op.unit_doses(recipe.book).items() if lyr == layer or lyr is None}
    for unit in doses:
        if not 0 <= unit < w.shape[dim]:
            raise ValueError(f"unit {unit} is outside `{target}`'s {w.shape[dim]} units")
    return _Units(name, role, op.op, dim, tuple(doses), tuple(doses.values()))


def apply_recipe_weights(
    root: nn.Module,
    recipe: Any,
    *,
    role: str = "instrument",
    name: str | None = None,
    reversible: bool = True,
    view: Any | None = None,
) -> list[WeightEditHandle]:
    """Apply a recipe's ``weights`` section (venngram recipe schema 3) to ``root``'s tensors, in order.

    A recipe applies its weights before any stage, so apply them before translating its stages
    (:func:`~interpretune.analysis.venngram.apply.recipe_interventions`, which refuses a recipe whose weights were not
    applied). venngram has already refused, when the recipe was read, a weights op whose token scope is not ``all`` or
    whose op cannot act on a parameter.

    Args:
        root: The module the targets resolve under.
        recipe: A ``venngram.recipe.Recipe`` (or a document :func:`~interpretune.analysis.venngram.load_recipe` reads).
        role: Every edit's operand role (see :func:`apply_weight_patch`).
        name: The edits' name stem (``<name>-<op index>``); the recipe's name by default.
        reversible: Parametrize the tensors (the default), or write the edited values into them.
        view: The view addresses resolve through; ``root``'s attached view, else its default view.

    Returns:
        One :class:`WeightEditHandle` per edited tensor, in application order (remove them in reverse).

    Raises:
        ValueError: If a target selects nothing, a unit is outside its tensor's units, a tensor's unit axis is unknown
            (its view's structure records no layout), an operand does not fit its tensor, or a patch was made for
            another base. Edits applied before the failing one are taken back.
    """
    if not hasattr(recipe, "weights"):
        from interpretune.analysis.venngram import load_recipe

        recipe = load_recipe(recipe).recipe
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", name or recipe.name)
    bound = _view(root, view)
    structure = bound.structure()
    _warn_if_viewed(root, role)
    handles: list[WeightEditHandle] = []
    try:
        for i, op in enumerate(recipe.weights):
            if op.op not in WEIGHT_OPS:
                raise ValueError(
                    f"weights op {i} (`{op.op}`) is not one interpretune materializes ({sorted(WEIGHT_OPS)})"
                )
            if op.tokens is not None and op.tokens.kind != "all":
                raise ValueError(
                    f"weights op {i} (`{op.op}`) has token scope `{op.tokens.kind}`; a weight edit holds on every token"
                )
            edit_name = f"{stem}-{i}"
            _check_role(edit_name, role)
            for target in _tensors_of(root, recipe, op, bound):
                edit = _edit_for(root, recipe, op, target, structure, edit_name, role)
                handles.append(_register(root, target, edit, reversible))
    except Exception:
        for handle in reversed(handles):
            if handle.reversible:
                handle.remove()
        raise
    return handles


def bias_patch(weight: torch.Tensor, units: Mapping[int, float], *, dim: int = 1) -> torch.Tensor:
    """The bias change :math:`\\sum_j d_j W_{:,j}` that equals adding dose :math:`d_j` to input unit :math:`j` of
    :math:`y = W x + b`, at token scope ``all``.

    Args:
        weight: :math:`W`, as stored.
        units: Dose by input unit.
        dim: ``weight``'s input-unit dimension (1 for ``nn.Linear``'s ``[out, in]``, 0 for GPT-2's ``Conv1D``).
    """
    index = torch.tensor(list(units), dtype=torch.long, device=weight.device)
    doses = torch.tensor(list(units.values()), dtype=weight.dtype, device=weight.device)
    columns = weight.index_select(dim, index)
    return (columns * doses.reshape((1, -1) if dim == 1 else (-1, 1))).sum(dim=dim)


#: The activation site whose unit ops have an exact weight form: the input of ``mlp.out`` (its units are the columns of
#: its weight, in venngram's vocabulary), at token scope ``all``.
WEIGHT_FORM_SITE = "mlp.out.hook_in"


def weight_form(
    root: nn.Module,
    recipe: Any,
    *,
    role: str = "instrument",
    name: str | None = None,
    view: Any | None = None,
) -> list[WeightEditHandle]:
    """Materialize a recipe's activation stages as the weight edits they equal, at token scope ``all``: a unit
    ``scale`` at ``mlp.out``'s input as a scale of those units of its weight, and a unit ``add`` as the bias patch
    :math:`b + \\sum_j d_j W_{:,j}` (:func:`bias_patch`).

    This is the weight form of a per-neuron recipe (neuronary's, for example), so it can be exported or compared with
    the hooked form (:func:`check_weight_equivalence`).

    Args:
        root: The module the targets resolve under.
        recipe: A ``venngram.recipe.Recipe`` (or a document :func:`~interpretune.analysis.venngram.load_recipe` reads).
        role: Every edit's operand role (see :func:`apply_weight_patch`).
        name: The edits' name stem (``<name>-<layer>-scale`` and ``<name>-<layer>-shift``); the recipe's name by
            default.
        view: The view addresses resolve through; ``root``'s attached view, else its default view.

    Returns:
        One :class:`WeightEditHandle` per edited tensor (each layer's weight, then its bias), in application order.

    Raises:
        ValueError: For anything with no exact weight form: a stage at another site or token scope, an op other than a
            unit ``scale`` or ``add`` with a ``dose`` or ``doses`` value, or an ``add`` at a layer whose ``mlp.out`` has
            no bias. Edits applied before the failing one are taken back.
    """
    if not hasattr(recipe, "stages"):
        from interpretune.analysis.venngram import load_recipe

        recipe = load_recipe(recipe).recipe
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", name or recipe.name)
    bound = _view(root, view)
    structure = bound.structure()
    _check_role(f"{stem}-0", role)
    # The ops on mlp.out's input compose, in order, into one affine map per layer, x -> s * x + a (a later scale also
    # scales an earlier add), and W (s * x + a) + b = (W diag(s)) x + (W a + b): a scale of W's units and a bias patch
    # taken from the unedited W.
    affine: dict[int, tuple[dict[int, float], dict[int, float]]] = {}
    for i, stage in enumerate(recipe.stages):
        if stage.site != WEIGHT_FORM_SITE or stage.tokens.kind != "all":
            raise ValueError(
                f"stage {i} ({stage.site}, tokens `{stage.tokens.kind}`) has no exact weight form: only "
                f"unit ops at `{WEIGHT_FORM_SITE}` at token scope `all` do"
            )
        for j, op in enumerate(stage.ops):
            if op.op not in ("scale", "add") or op.target_kind(recipe.book) != "units" or op.operand is not None:
                raise ValueError(
                    f"stage {i} op {j} (`{op.op}`) has no exact weight form: only a unit `scale` or `add` "
                    "with a dose does"
                )
            for (layer, unit), dose in op.unit_doses(recipe.book).items():
                scale, shift = affine.setdefault(layer, ({}, {}))
                if op.op == "scale":
                    scale[unit] = scale.get(unit, 1.0) * dose
                    if unit in shift:
                        shift[unit] *= dose
                else:
                    shift[unit] = shift.get(unit, 0.0) + dose
    handles: list[WeightEditHandle] = []
    try:
        for layer, (scale, shift) in sorted(affine.items()):
            at = {"component": "mlp.out", "layers": [layer]}
            (weight,) = resolve_weight_targets(root, {**at, "leaf": "weight"}, view=bound)
            bias = resolve_weight_targets(root, {**at, "leaf": "bias", "optional": True}, view=bound)
            if shift and not bias:
                raise ValueError(
                    f"`mlp.out` at layer {layer} has no bias, so an `add` to its input has no weight form there"
                )
            w = _tensor(root, weight).detach().clone()
            dim = _unit_dim(structure, weight, w.ndim)
            if scale:
                edit: WeightEdit = _Units(
                    f"{stem}-{layer}-scale", role, "scale", dim, tuple(scale), tuple(scale.values())
                )
                handles.append(_register(root, weight, edit, True))
            if shift:
                patch = bias_patch(w, shift, dim=dim)
                handles.append(_register(root, bias[0], _Vector(f"{stem}-{layer}-shift", role, "add", patch), True))
    except Exception:
        for handle in reversed(handles):
            handle.remove()
        raise
    return handles


@dataclass(frozen=True)
class EquivalenceResult:
    """Two forms of one edit compared on a batch."""

    max_abs_diff: float
    max_abs: float
    rtol: float
    atol: float

    @property
    def equivalent(self) -> bool:
        """Whether the two forms agree within ``atol + rtol * max_abs``."""
        return self.max_abs_diff <= self.atol + self.rtol * self.max_abs


def check_weight_equivalence(
    forward: Callable[[], torch.Tensor],
    apply_weight_form: Callable[[], Sequence[WeightEditHandle] | WeightEditHandle],
    apply_activation_form: Callable[[], Any],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> EquivalenceResult:
    """Compare an edit's weight form with its activation form: ``forward()`` (the model's output on a fixed batch)
    with the weight form applied, then with it taken back and the activation form applied.

    Args:
        forward: Runs the model on the batch and returns the output compared (logits, say).
        apply_weight_form: Applies the weight form, returning its handle or handles (taken back afterwards).
        apply_activation_form: Applies the activation form (a forward hook, say), returning an object with a
            ``remove()`` method (taken back afterwards) or ``None``.
        rtol: Relative tolerance on the largest output magnitude.
        atol: Absolute tolerance.

    Returns:
        An :class:`EquivalenceResult`.
    """
    handles = apply_weight_form()
    handles = [handles] if isinstance(handles, WeightEditHandle) else list(handles)
    try:
        with torch.no_grad():
            by_weight = forward().detach()
    finally:
        for handle in reversed(handles):
            handle.remove()
    hook = apply_activation_form()
    try:
        with torch.no_grad():
            by_activation = forward().detach()
    finally:
        if hook is not None:
            hook.remove()
    diff = (by_weight.float() - by_activation.float()).abs().max().item()
    return EquivalenceResult(max_abs_diff=diff, max_abs=by_activation.float().abs().max().item(), rtol=rtol, atol=atol)


__all__ = [
    "EDIT_ROLES",
    "EquivalenceResult",
    "WEIGHT_OPS",
    "WeightEdit",
    "WeightEditHandle",
    "apply_recipe_weights",
    "apply_weight_patch",
    "bias_patch",
    "check_weight_equivalence",
    "resolve_weight_targets",
    "WEIGHT_FORM_SITE",
    "weight_edit_roles",
    "weight_form",
]
