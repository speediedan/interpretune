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
"""A finetuning-scheduler ``TunableView`` over a TransformerLens ``TransformerBridge``, in TransformerLens names.

Prototype (interpretune#758): the view the protocol freeze is checked against. Logical names are the bridge's
``tl_named_parameters()`` names (``blocks.9.attn.W_Q``); physical names are the bridge's own ``named_parameters()``.
Each TransformerLens name maps to the one physical parameter its tensor is a view of, found by tracing the bridge's
components and matching by identity (not by ``data_ptr``, which a fused projection shares with its first split).

Two kinds of tensor are thawed implicitly (``coupled``):

- every norm of a block whose parameters are thawed, and the final norm with the embeddings (as the legacy
  ``TLNamesModelView`` does with ``implicit_ln_thaw``), found by component type so norms TransformerLens does not
  name are covered too;
- physical parameters that alias a thawed tensor's storage but that no logical name reaches, e.g. GPT-2's fused
  ``attn.qkv`` weight, whose ``q``/``k``/``v`` splits are strided views of it. A step through a split changes the
  fused parameter's values, so the fused parameter is thawed with it.

``depth_groups`` keeps tensors that share storage in one group: the unembedding with an embedding tied to it, and
each block whole (its query, key and value splits may share one fused tensor).

Checkpoints use the module tree's own keys, saved without state-dict hooks. Neither of the bridge's own paths
round-trips: ``TransformerBridge.state_dict()`` returns TransformerLens-format keys that its ``load_state_dict()`` does
not map back (it drops them without error when the key counts differ), and ``JointQKVAttentionBridge``'s state-dict hook
removes the fused ``qkv`` keys that a strict ``torch.nn.Module.load_state_dict()`` then requires.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from torch import nn, Tensor

from finetuning_scheduler.core.views import TunableSupport, Unresolvable

_BLOCK = re.compile(r"^blocks\.(\d+)\.")
_ATTN = {
    "W_Q": ("q", "weight"),
    "W_K": ("k", "weight"),
    "W_V": ("v", "weight"),
    "W_O": ("o", "weight"),
    "b_Q": ("q", "bias"),
    "b_K": ("k", "bias"),
    "b_V": ("v", "bias"),
    "b_O": ("o", "bias"),
}
_MLP = {
    "W_in": ("weight", ("in", "input")),
    "b_in": ("bias", ("in", "input")),
    "W_out": ("weight", ("out",)),
    "b_out": ("bias", ("out",)),
    "W_gate": ("weight", ("gate",)),
    "b_gate": ("bias", ("gate",)),
}


def _first_attr(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        found = getattr(obj, name, None)
        if found is not None:
            return found
    return None


def _norm_tensor(ln: Any, param_name: str) -> Tensor | None:
    if param_name in ("w", "weight"):
        return getattr(ln, "weight", None)
    if param_name in ("b", "bias"):
        return getattr(ln, "bias", None)
    return None


def underlying_tensor(bridge: Any, tl_name: str) -> Tensor | None:
    """The component parameter a TransformerLens name is a view of, or ``None`` when it cannot be traced."""
    parts = tl_name.split(".")
    try:
        if parts[0] == "blocks":
            block = bridge.blocks[int(parts[1])]
            component, param_name = parts[2], parts[3]
            if component == "attn" and param_name in _ATTN:
                comp_name, attr = _ATTN[param_name]
                comp = getattr(block.attn, comp_name, None)
                return getattr(comp, attr, None) if comp is not None else None
            if component == "mlp" and param_name in _MLP:
                attr, names = _MLP[param_name]
                comp = _first_attr(block.mlp, names)
                return getattr(comp, attr, None) if comp is not None else None
            if component in ("ln1", "ln2"):
                ln = getattr(block, component, None)
                return _norm_tensor(ln, param_name) if ln is not None else None
        elif parts[0] == "embed":
            return getattr(bridge.embed, "weight", None)
        elif parts[0] == "pos_embed":
            pos_embed = getattr(bridge, "pos_embed", None)
            return getattr(pos_embed, "weight", None) if pos_embed is not None else None
        elif parts[0] == "unembed":
            return getattr(bridge.unembed, {"W_U": "weight", "b_U": "bias"}.get(parts[1], ""), None)
        elif parts[0] == "ln_final":
            ln_final = getattr(bridge, "ln_final", None)
            return _norm_tensor(ln_final, parts[1]) if ln_final is not None else None
    except (AttributeError, IndexError, KeyError, ValueError):
        return None
    return None


def _storage_key(t: Tensor) -> tuple[str, int] | None:
    if hasattr(t, "to_local") or t.device.type == "meta" or t.numel() == 0:
        return None
    return str(t.device), t.untyped_storage().data_ptr()


class TransformerBridgeView:
    """A ``TunableView`` over a ``TransformerBridge`` in TransformerLens names.

    It does not inherit from a finetuning-scheduler class: finetuning-scheduler supplies the optional members it lacks
    (``roles``, the checkpoint members and ``promote``).

    Args:
        bridge: The bridge (the view's root).
        implicit_norm_thaw: Thaw a block's norms with any of its parameters, and the final norm with the embeddings.
    """

    def __init__(self, bridge: nn.Module, implicit_norm_thaw: bool = True) -> None:
        from transformer_lens.model_bridge.generalized_components import NormalizationBridge

        self.root = bridge
        self.implicit_norm_thaw = implicit_norm_thaw
        named = list(bridge.named_parameters())
        name_of = {id(p): n for n, p in named}
        self._table: dict[str, tuple[str, ...]] = {}
        self._tl_tensors: dict[str, Tensor] = {}
        self.untraced: list[str] = []
        for tl_name, tl_tensor in bridge.tl_named_parameters():  # type: ignore[attr-defined]
            under = underlying_tensor(bridge, tl_name)
            physical = name_of.get(id(under)) if under is not None else None
            if physical is None:
                self.untraced.append(tl_name)
                continue
            self._table[tl_name] = (physical,)
            self._tl_tensors[tl_name] = tl_tensor
        self._reverse: dict[str, str] = {}
        for logical, physical in self._table.items():
            for p in physical:
                self._reverse.setdefault(p, logical)
        reached = set(self._reverse)
        # physical parameters on one storage, for the aliases no logical name reaches (a fused projection)
        by_storage: dict[tuple[str, int], list[str]] = {}
        for n, p in named:
            key = _storage_key(p)
            if key is not None:
                by_storage.setdefault(key, []).append(n)
        self._aliases: dict[str, list[str]] = {}
        for group in by_storage.values():
            unreached = [n for n in group if n not in reached]
            for n in group:
                if n in reached and unreached:
                    self._aliases[n] = unreached

        def _norms(module: nn.Module) -> list[str]:
            return [
                name_of[id(p)]
                for c in module._modules.values()
                if isinstance(c, NormalizationBridge)
                for p in c.parameters()
                if id(p) in name_of
            ]

        self._block_norms = {i: _norms(block) for i, block in enumerate(bridge.blocks)}  # type: ignore[attr-defined]
        final = getattr(bridge, "ln_final", None)
        self._final_norm = (
            [name_of[id(p)] for p in final.parameters() if id(p) in name_of]
            if isinstance(final, NormalizationBridge)
            else []
        )
        self._block_of = {
            p: int(m.group(1)) for logical, ps in self._table.items() if (m := _BLOCK.match(logical)) for p in ps
        }
        self._embedding_physical = {
            p
            for logical, ps in self._table.items()
            if logical.startswith(("embed.", "pos_embed.", "unembed."))
            for p in ps
        }

    def support(self) -> TunableSupport:
        """Trainable, in TransformerLens names that are reshaped views of the physical tensors."""
        return TunableSupport(trainable=True, naming_scheme="tl", logical_names_are_views=True)

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Each TransformerLens name mapped to the one physical parameter it is a view of."""
        unresolved = [n for n in logical if n not in self._table]
        if unresolved:
            raise Unresolvable(unresolved, "They are not TransformerLens names this bridge's view could trace.")
        return {n: self._table[n] for n in logical}

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        """Physical names as TransformerLens names; a name no TransformerLens name reaches passes through."""
        return list(dict.fromkeys(self._reverse.get(p, p) for p in physical))

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """The traced TransformerLens names with their TransformerLens tensors, in the bridge's order.

        Names TransformerLens synthesizes with no parameter behind them (zero biases of a bias-free projection) are left
        out.
        """
        yield from self._tl_tensors.items()

    def coupled(self, physical: Sequence[str]) -> list[str]:
        """The unnamed storage aliases of ``physical`` and, with ``implicit_norm_thaw``, their blocks' norms (the
        final norm for the embeddings)."""
        out: list[str] = []
        for p in physical:
            out += self._aliases.get(p, [])
            if not self.implicit_norm_thaw:
                continue
            if p in self._block_of:
                out += self._block_norms.get(self._block_of[p], [])
            elif p in self._embedding_physical:
                out += self._final_norm
        return list(dict.fromkeys(out))

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Every module's own state under its module-tree key, with no state-dict hooks applied, so that a strict
        ``torch.nn.Module.load_state_dict()`` accepts it."""
        state: dict[str, Any] = {}
        for prefix, module in self.root.named_modules(remove_duplicate=False):
            module._save_to_state_dict(state, f"{prefix}." if prefix else "", False)
        return state

    def load_checkpoint_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        """Load state from :meth:`checkpoint_state_dict` through ``torch.nn.Module.load_state_dict()``."""
        nn.Module.load_state_dict(self.root, state_dict, strict=strict)

    def depth_groups(self) -> list[list[str]]:
        """The unembedding first (with any embedding tied to it), then each block whole from the last, then the
        remaining embeddings."""
        unembed = [n for n in self._table if n.startswith("unembed.")]
        unembed_physical = {p for n in unembed for p in self._table[n]}
        embeds = [n for n in self._table if n.startswith(("embed.", "pos_embed."))]
        tied = [n for n in embeds if set(self._table[n]) & unembed_physical]
        blocks: dict[int, list[str]] = {}
        for n in self._table:
            if m := _BLOCK.match(n):
                blocks.setdefault(int(m.group(1)), []).append(n)
        groups = [unembed + tied] if unembed or tied else []
        groups += [blocks[i] for i in sorted(blocks, reverse=True)]
        rest = [n for n in embeds if n not in tied]
        if rest:
            groups.append(rest)
        return [g for g in groups if g]
