"""The activation-point vocabulary: two levels of name, parsed once at the boundary.

A **component point** says WHERE a tensor is in an architecture's module tree, spelled in the
TransformerLens v3 bridge grammar (``blocks.{i}.ln2.hook_out``, ``unembed.hook_out``). A **semantic
point** says what the tensor IS in the forward (``hook_resid_pre``, ``hook_mlp_out``) and resolves to a
component point per architecture. The frictions this vocabulary exists to remove were all cases of one
level read as the other: ``hook_mlp_out`` is the MLP's CONTRIBUTION to the residual, which is the raw
module output on GPT-2 and the post-norm output on a sandwich-norm model, and a table that stored it as a
fixed path had to be wrong on one of them.

Legacy ``HookedTransformer`` spellings that are neither semantic points nor component points (``attn.hook_z``,
``hook_embed``, ``hook_q``) are served through ONE alias table, :data:`ALIASES`, which the parser consults once.
An alias asserts only that a spelling MEANS a point; whether two spellings name the same tensor is answered by
resolution, per architecture, never by the table.

Strings stay as the serialization only. Nothing downstream of :func:`parse` matches a string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

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
    alias: str | None = None
    """The deprecated spelling this point was parsed from, when it came through the alias table, so a caller or a
    linter can report "you wrote X, this means Y"."""
    stack: str | None = None
    """The block stack for an indexed point when it is not the primary one (``"encoder"``, ``"vision"``)."""

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
        stack = f"{self.stack}." if self.stack else ""
        head = f"{stack}blocks.{self.layer}." if self.layer is not None else ""
        tail = f".{self.subhook}" if self.subhook else ""
        return f"{head}{self.base}{tail}"


#: Semantic points: what a tensor IS in the forward. These names survive an architecture change and are what
#: analysis code should ask for; they are not legacy. Normalized to (component, slot, contribution, caution).
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
    # The sublayers' arguments ARE the block norms' outputs (measured cos 1.000000 on gemma-3-1b-it layer 5), and
    # the vocabulary addresses them at the norm: one module boundary, one tensor position every backend can hook
    # without reading a sublayer's argument tuple. Component spellings, not aliases: nothing is deprecated here.
    "mlp.hook_in": ("ln2", Slot.OUT, None, None),
    "attn.hook_in": ("ln1", Slot.OUT, None, None),
}

# An optional leading stack name (`encoder.blocks.3.attn.hook_out`, `vision.blocks.0.mlp.hook_in`) is the
# one grammar extension for architectures with more than one block stack; bare `blocks.` is the primary stack,
# so every existing string keeps its meaning.
_BLOCK_RE = re.compile(r"^(?:(?P<stack>[a-z][a-z0-9_]*)\.)?blocks\.(?P<layer>\d+)\.(?P<rest>.+)$")
_SLOT_RE = re.compile(r"^(?:(?P<component>.+)\.)?hook_(?P<slot>in|out|normalized|scale)$")


class UnknownPointError(ValueError):
    """The name is not in the vocabulary.

    Names the nearest valid spellings.
    """


class DeprecatedPointError(UnknownPointError):
    """A deprecated spelling was used where the caller asked for canonical names only (``strict=True``)."""


class AliasLevel(str, Enum):
    """Which level of the vocabulary an alias spells: a semantic point or a component point."""

    SEMANTIC = "semantic"
    COMPONENT = "component"


@dataclass(frozen=True)
class Alias:
    """One deprecated spelling and the canonical point it means.

    ``canonical`` is a block-relative or global base name in the vocabulary (``attn.o.hook_in``, ``embed.hook_out``,
    ``hook_resid_pre``); ``replacement`` is what a caller should write instead, which is usually the canonical
    spelling but may name a semantic point when that is the better habit.
    """

    alias: str
    canonical: str
    level: AliasLevel
    deprecated_since: str
    replacement: str | None = None
    caution: str | None = None
    source: str = "bundled"

    @property
    def suggested(self) -> str:
        """What to write instead."""
        return self.replacement or self.canonical


class AliasTable:
    """``alias -> Alias``: the legacy ``HookedTransformer`` spellings the parser accepts, served through ONE table.

    The parser consults it once, so an alias never has to be listed by hand beside the point it names and cannot
    disagree with it. Adapters may :meth:`register` aliases for their own vocabulary; a ``strict`` parse refuses
    every entry, for configs that want to be canonical. The former ``HOOK_ALIAS_GROUPS`` in the interventions
    module asserted tensor IDENTITY between spellings, and two of its groups were wrong (measured, #375, #376);
    an alias here asserts only that a spelling MEANS a point, and the point resolves per architecture.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Alias] = {}

    def register(self, entry: Alias, *, replace: bool = False) -> None:
        """Add an alias; refuse to shadow a semantic point, a component spelling, or another alias unless asked."""
        if entry.alias in _SEMANTIC:
            raise ValueError(f"{entry.alias!r} is a semantic point, not an alias; it cannot be redefined")
        if _SLOT_RE.match(entry.alias):
            raise ValueError(f"{entry.alias!r} is a component spelling; component points are canonical by construction")
        existing = self._entries.get(entry.alias)
        if existing is not None and not replace and existing != entry:
            raise ValueError(
                f"alias {entry.alias!r} is already registered (-> {existing.canonical!r}, from {existing.source!r}); "
                "pass replace=True to redefine it deliberately"
            )
        self._entries[entry.alias] = entry

    def get(self, name: str) -> Alias | None:
        """The entry for a deprecated spelling, or ``None``."""
        return self._entries.get(name)

    def aliases_for(self, canonical: str) -> tuple[str, ...]:
        """Every deprecated spelling that means ``canonical`` (a base name), sorted."""
        return tuple(sorted(a for a, e in self._entries.items() if e.canonical == canonical))

    def __iter__(self) -> Iterator[Alias]:
        return iter(sorted(self._entries.values(), key=lambda e: e.alias))

    def __len__(self) -> int:
        return len(self._entries)


def _bundled_aliases() -> list[Alias]:
    c = AliasLevel.COMPONENT
    since = "0.1.0"
    # Not here: `hook_in` / `hook_out` (the block's own slots) and `attn.hook_in` / `mlp.hook_in` (the sublayers'
    # arguments, addressed at the norm output in `_SEMANTIC`). Those are component spellings, not deprecated ones.
    return [
        # attention internals
        Alias("attn.hook_z", "attn.o.hook_in", c, since),
        Alias("hook_q_input", "attn.q.hook_in", c, since),
        Alias("hook_k_input", "attn.k.hook_in", c, since),
        Alias("hook_v_input", "attn.v.hook_in", c, since),
        Alias("hook_q", "attn.q.hook_out", c, since),
        Alias("hook_k", "attn.k.hook_out", c, since),
        Alias("hook_v", "attn.v.hook_out", c, since),
        # MLP internals: TransformerLens defines `hook_pre` as the up-projection's OUTPUT (the pre-activation) and
        # `hook_post` as the activation's output feeding the down projection
        Alias("mlp.hook_pre", "mlp.in.hook_out", c, since),
        Alias("mlp.hook_post", "mlp.out.hook_in", c, since),
        # embeddings
        Alias("hook_embed", "embed.hook_out", c, since),
        Alias("hook_pos_embed", "pos_embed.hook_out", c, since),
    ]


def _bundled_table() -> AliasTable:
    table = AliasTable()
    for entry in _bundled_aliases():
        table.register(entry)
    return table


ALIASES = _bundled_table()


def parse(name: str, *, strict: bool = False) -> ActivationPoint:
    """Parse any accepted spelling (component, semantic, or a deprecated alias) into an :class:`ActivationPoint`.

    Total and strict: a name outside the vocabulary raises :class:`UnknownPointError` naming valid forms. With
    ``strict=True`` a deprecated alias raises :class:`DeprecatedPointError` naming what to write instead; otherwise
    it parses to its canonical point with ``alias`` set to the spelling used.
    """
    match = _BLOCK_RE.match(name)
    stack: str | None = None
    if match is None:
        if name.startswith("blocks.") or ".blocks." in name:
            raise UnknownPointError(f"cannot parse {name!r}: expected '[<stack>.]blocks.<layer>.<point>'")
        layer, rest = None, name
    else:
        layer, rest = int(match.group("layer")), match.group("rest")
        stack = match.group("stack")
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
        return ActivationPoint(component, slot, layer, contribution, subhook, caution, stack=stack)
    if (entry := ALIASES.get(base)) is not None:
        if strict:
            raise DeprecatedPointError(
                f"{name!r} uses the deprecated spelling {base!r} (since {entry.deprecated_since}); write "
                f"{entry.suggested!r} instead"
            )
        canonical = parse(entry.canonical)
        return ActivationPoint(
            canonical.component,
            canonical.slot,
            layer,
            canonical.contribution,
            subhook,
            entry.caution or canonical.caution,
            alias=base,
            stack=stack,
        )
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
    return ActivationPoint(component, slot, layer, None, subhook, None, stack=stack)


def register_alias(
    alias: str,
    canonical: str,
    *,
    level: AliasLevel | str = AliasLevel.COMPONENT,
    deprecated_since: str = "unversioned",
    replacement: str | None = None,
    caution: str | None = None,
    source: str = "runtime",
    replace: bool = False,
) -> Alias:
    """Register an adapter's own deprecated spelling; ``canonical`` must itself parse without the table."""
    parse(canonical, strict=True)
    entry = Alias(alias, canonical, AliasLevel(level), deprecated_since, replacement, caution, source)
    ALIASES.register(entry, replace=replace)
    return entry


def semantic_names() -> tuple[str, ...]:
    """Every semantic spelling plus every registered alias: the non-component names the parser accepts."""
    return tuple(sorted({*_SEMANTIC, *(e.alias for e in ALIASES)}))


def spellings(name: str) -> tuple[str, ...]:
    """Every spelling of ``name``'s point: the name itself, its canonical component form, the semantic names for
    the same slot, and every alias meaning any of those. Layer and sub-hook suffix are carried through.

    For a semantic contribution (``hook_attn_out``) only the spelling given is returned: the component it resolves
    to depends on the architecture, and naming the raw module would silently substitute the pre-norm output on a
    sandwich-norm model. Consumers that need the component ask the resolver with a map.
    """
    point = parse(name)
    head = f"blocks.{point.layer}." if point.layer is not None else ""
    tail = f".{point.subhook}" if point.subhook else ""
    given = name[len(head) : len(name) - len(tail)] if tail else name[len(head) :]
    bases: list[str] = [given]
    if point.contribution is None:
        bases.append(point.base)
        # A cautioned semantic name (`hook_mlp_in`) is matched only when written: it names the same tensor as
        # `hook_resid_mid`, but half its callers mean the post-norm input, so it is never offered as a variant.
        bases.extend(
            sem
            for sem, (component, slot, contribution, caution) in _SEMANTIC.items()
            if (component, slot) == (point.component, point.slot) and contribution is None and caution is None
        )
        for base in list(bases):
            bases.extend(ALIASES.aliases_for(base))
    return tuple(dict.fromkeys(f"{head}{b}{tail}" for b in bases))
