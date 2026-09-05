"""The activation-point vocabulary: two levels of name, parsed once at the boundary.

A **component point** says WHERE a tensor is in an architecture's module tree, spelled in the
TransformerLens v3 bridge grammar (``blocks.{i}.ln2.hook_out``, ``unembed.hook_out``). A **semantic
point** says what the tensor IS in the forward (``hook_resid_pre``, ``hook_mlp_out``) and resolves to a
component point per architecture. The frictions this vocabulary exists to remove were all cases of one
level read as the other: ``hook_mlp_out`` is the MLP's CONTRIBUTION to the residual, which is the raw
module output on GPT-2 and the post-norm output on a sandwich-norm model, and a table that stored it as a
fixed path had to be wrong on one of them.

Strings stay as the serialization only. Nothing downstream of :func:`parse` matches a string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from interpretune.analysis.backends.hook_mapping_constants import SUBHOOK_SUFFIXES


class Slot(str, Enum):
    """Which tensor of a component a point names."""

    IN = "in"
    """The component's first positional input."""
    OUT = "out"
    """The component's output (element 0 when the module returns a tuple)."""
    NORMALIZED = "normalized"
    """A norm's ``x / scale`` BEFORE the learned gain.

    Derived: no module emits it.
    """
    SCALE = "scale"
    """A norm's per-token denominator, ``[batch, pos, 1]``.

    Derived.
    """


class Contribution(str, Enum):
    """A sublayer's contribution to the residual stream: post-norm output where a post-norm exists, else the raw
    module output.

    The one meaning no single component spelling names across architectures.
    """

    ATTN = "attn"
    MLP = "mlp"


@dataclass(frozen=True)
class ActivationPoint:
    """A parsed point.

    ``component`` is the bridge component path relative to a block (``""`` for the block
    itself, ``"ln2"``, ``"attn.o"``) or a global (``"unembed"``), ``layer`` is ``None`` for globals.
    """

    component: str
    slot: Slot
    layer: int | None = None
    contribution: Contribution | None = None
    """Set when the point is a semantic contribution (``hook_attn_out`` / ``hook_mlp_out``); the component field
    then names the raw module and the resolver picks the post-norm when the architecture has one."""
    subhook: str | None = None
    """An SAE sub-hook suffix (``hook_sae_acts_post``) carried through unchanged."""
    caution: str | None = None
    """A note the resolver surfaces for names that are precise but widely misread."""

    @property
    def is_global(self) -> bool:
        """Whether the point is outside the block stack (no layer)."""
        return self.layer is None

    @property
    def base(self) -> str:
        """The block-relative or global spelling without the layer: ``ln2.hook_out``, ``hook_out``."""
        return f"{self.component}.hook_{self.slot.value}" if self.component else f"hook_{self.slot.value}"

    @property
    def canonical(self) -> str:
        """The bridge-grammar string form, with the layer."""
        head = f"blocks.{self.layer}." if self.layer is not None else ""
        tail = f".{self.subhook}" if self.subhook else ""
        return f"{head}{self.base}{tail}"


#: Semantic and legacy spellings, normalized to (component, slot, contribution, caution).
#: The ``hook_resid_*`` family is the semantic layer, not a legacy one: the names survive an architecture change.
_SEMANTIC: dict[str, tuple[str, Slot, Contribution | None, str | None]] = {
    "hook_resid_pre": ("", Slot.IN, None, None),
    "hook_resid_post": ("", Slot.OUT, None, None),
    "hook_resid_mid": ("ln2", Slot.IN, None, None),
    "hook_attn_out": ("attn", Slot.OUT, Contribution.ATTN, None),
    "hook_mlp_out": ("mlp", Slot.OUT, Contribution.MLP, None),
    "hook_attn_in": (
        "ln1",
        Slot.IN,
        None,
        "TransformerLens fires `hook_attn_in` on the residual BEFORE the block norm; the attention sublayer's "
        "actual argument is `attn.hook_in` (the norm's output). Measured cos 0.088 apart on gemma-3-1b-it layer 5.",
    ),
    "hook_mlp_in": (
        "ln2",
        Slot.IN,
        None,
        "TransformerLens fires `hook_mlp_in` on the residual BEFORE the block norm; the MLP's actual argument is "
        "`mlp.hook_in` (the norm's output). Measured cos 0.088 apart on gemma-3-1b-it layer 5.",
    ),
    "attn.hook_z": ("attn.o", Slot.IN, None, None),
    "hook_embed": ("embed", Slot.OUT, None, None),
    "hook_pos_embed": ("pos_embed", Slot.OUT, None, None),
    # TransformerLens defines `hook_pre` as the projection's OUTPUT (`mlp.hook_pre -> in.hook_out`), the
    # pre-activation, and `hook_post` as the activation's output feeding the down projection.
    "mlp.hook_pre": ("mlp.in", Slot.OUT, None, None),
    "mlp.hook_post": ("mlp.out", Slot.IN, None, None),
    # The sublayer's argument IS the block norm's output (measured cos 1.000000 on gemma-3-1b-it layer 5).
    "mlp.hook_in": ("ln2", Slot.OUT, None, None),
    "attn.hook_in": ("ln1", Slot.OUT, None, None),
}

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_SLOT_RE = re.compile(r"^(?:(?P<component>.+)\.)?hook_(?P<slot>in|out|normalized|scale)$")


class UnknownPointError(ValueError):
    """The name is not in the vocabulary.

    Names the nearest valid spellings.
    """


def parse(name: str) -> ActivationPoint:
    """Parse any accepted spelling (component, semantic, or legacy) into an :class:`ActivationPoint`.

    Total and strict: a name outside the vocabulary raises :class:`UnknownPointError` naming valid forms.
    """
    match = _BLOCK_RE.match(name)
    if match is None:
        if name.startswith("blocks."):
            raise UnknownPointError(f"cannot parse {name!r}: expected 'blocks.<layer>.<point>'")
        layer, rest = None, name
    else:
        layer, rest = int(match.group(1)), match.group(2)

    parts = rest.split(".")
    subhook: str | None = None
    for i, part in enumerate(parts):
        if part in SUBHOOK_SUFFIXES:
            subhook = ".".join(parts[i:])
            parts = parts[:i]
            break
    base = ".".join(parts)
    if not base:
        raise UnknownPointError(f"cannot parse {name!r}: no point name after the layer")

    if base in _SEMANTIC:
        component, slot, contribution, caution = _SEMANTIC[base]
        return ActivationPoint(component, slot, layer, contribution, subhook, caution)

    slot_match = _SLOT_RE.match(base)
    if slot_match is None:
        raise UnknownPointError(
            f"unknown activation point {base!r} in {name!r}; expected '<component>.hook_<in|out|normalized|scale>' "
            f"or a semantic name such as {sorted(_SEMANTIC)[:6]} ..."
        )
    component = slot_match.group("component") or ""
    slot = Slot(slot_match.group("slot"))
    if slot in (Slot.NORMALIZED, Slot.SCALE) and not component:
        raise UnknownPointError(f"{name!r}: hook_{slot.value} needs a norm component (e.g. 'ln2.hook_{slot.value}')")
    return ActivationPoint(component, slot, layer, None, subhook, None)


def semantic_names() -> tuple[str, ...]:
    """Every semantic and legacy spelling the parser accepts."""
    return tuple(sorted(_SEMANTIC))
