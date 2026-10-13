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
Each TransformerLens name maps to the parameters its tensor is computed from, found by walking the tensor's autograd
graph, so the mapping follows whatever TransformerLens emits rather than a table kept here. A name with no parameter
behind it (a zero bias TransformerLens synthesizes for a bias-free projection) is left out. Parameters are matched by
identity, never by ``data_ptr``, which a fused projection shares with its first split.

Two kinds of tensor are thawed implicitly (``coupled``):

- every norm of a block whose parameters are thawed, and the final norm with the embeddings (as the legacy
  ``TLNamesModelView`` does with ``implicit_ln_thaw``), found by component type so norms TransformerLens does not
  name are covered too;
- physical parameters that alias a thawed tensor's storage but that no logical name reaches, e.g. GPT-2's fused
  ``attn.qkv`` weight, whose ``q``/``k``/``v`` splits are strided views of it. A step through a split changes the
  fused parameter's values, so the fused parameter is thawed with it.

``depth_groups`` keeps tensors that share storage in one group: the unembedding with an embedding tied to it, and
each block whole (its query, key and value splits may share one fused tensor).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import nn, Tensor

from finetuning_scheduler.core.views import TunableSupport, Unresolvable

_BLOCK = re.compile(r"^blocks\.(\d+)\.")


def autograd_leaves(t: Tensor) -> list[Tensor]:
    """The leaf tensors ``t`` was computed from, found by walking its autograd graph (``t`` itself when it is a
    leaf that requires grad).

    A tensor with no graph, such as a synthesized zero bias, has none.
    """
    if t.grad_fn is None:
        return [t] if t.requires_grad else []
    leaves: dict[int, Tensor] = {}
    stack: list[Any] = [t.grad_fn]
    seen: set[int] = set()
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        variable = getattr(fn, "variable", None)  # an AccumulateGrad node holds its leaf
        if variable is not None:
            leaves.setdefault(id(variable), variable)
        stack.extend(next_fn for next_fn, _ in fn.next_functions)
    return list(leaves.values())


def _traced_by_autograd(bridge: nn.Module) -> dict[str, tuple[Tensor, list[Tensor]]]:
    """Each TransformerLens name with its tensor (detached) and the parameters it is computed from.

    Every parameter is made to require grad while the TransformerLens tensors are built, so that each one records how
    it was derived; ``requires_grad`` is restored afterwards.
    """
    params = list(bridge.parameters())
    saved = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(True)
        with torch.enable_grad():
            return {n: (t.detach(), autograd_leaves(t)) for n, t in bridge.tl_named_parameters()}  # type: ignore[attr-defined]
    finally:
        for p, flag in zip(params, saved):
            p.requires_grad_(flag)


def _storage_key(t: Tensor) -> tuple[str, int] | None:
    if hasattr(t, "to_local") or t.device.type == "meta" or t.numel() == 0:
        return None
    return str(t.device), t.untyped_storage().data_ptr()


class TransformerBridgeView:
    """A ``TunableView`` over a ``TransformerBridge`` in TransformerLens names.

    It does not inherit from a finetuning-scheduler class: finetuning-scheduler supplies the optional members it lacks
    (``roles``, the checkpoint members and ``promote``). The default checkpoint members round-trip through
    ``TransformerBridge.state_dict()`` and ``load_state_dict()`` from transformer-lens 4.0.

    The root is the module that holds the bridge (the interpretune module, whose ``model`` is the bridge), so physical
    names are that module's names (``model.blocks.0._original_component.attn.q._original_component.weight``) and
    modules registered beside the bridge stay reachable. A bridge may also be its own root.

    Args:
        root: The view's root: the interpretune module, or the bridge itself.
        bridge: The bridge, when it is not ``root.model`` (or ``root``).
        implicit_norm_thaw: Thaw a block's norms with any of its parameters, and the final norm with the embeddings.
        roles: Roles for ``root``'s parameters and buffers, by physical name (anything unlisted is ``base``), e.g.
            ``latent:sae`` for SAEs spliced into the bridge.
    """

    def __init__(
        self,
        root: nn.Module,
        bridge: nn.Module | None = None,
        implicit_norm_thaw: bool = True,
        roles: Mapping[str, str] | None = None,
    ) -> None:
        from finetuning_scheduler.core.roles import validate_roles

        self._roles = validate_roles(
            roles or {}, (n for n, _ in root.named_parameters()), (n for n, _ in root.named_buffers())
        )
        if bridge is None:
            model = root._modules.get("model")
            bridge = model if model is not None and hasattr(model, "tl_named_parameters") else root
        self.root = root
        self.bridge = bridge
        self.implicit_norm_thaw = implicit_norm_thaw
        self._built = False

    @property
    def untraced(self) -> list[str]:
        """TransformerLens names with no parameter behind them, left out of the logical names."""
        self._build()
        return self._untraced

    def _build(self) -> None:
        """Build the mapping on first use, so a session that never tunes never traces the bridge's tensors."""
        if self._built:
            return
        from transformer_lens.model_bridge.generalized_components import NormalizationBridge

        root, bridge = self.root, self.bridge
        named = list(root.named_parameters())
        name_of = {id(p): n for n, p in named}
        self._table: dict[str, tuple[str, ...]] = {}
        self._tl_tensors: dict[str, Tensor] = {}
        self._untraced: list[str] = []
        traced = {
            n: (t, [name_of[id(leaf)] for leaf in leaves if id(leaf) in name_of])
            for n, (t, leaves) in _traced_by_autograd(bridge).items()
        }
        for tl_name, (tl_tensor, physical) in traced.items():
            if not physical:
                self._untraced.append(tl_name)
                continue
            self._table[tl_name] = tuple(physical)
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
        self._built = True

    def support(self) -> TunableSupport:
        """Trainable, in TransformerLens names that are reshaped views of the physical tensors."""
        from finetuning_scheduler.core.roles import declared_kinds

        return TunableSupport(
            trainable=True, naming_scheme="tl", logical_names_are_views=True, declared_roles=declared_kinds(self._roles)
        )

    def roles(self) -> Mapping[str, str]:
        """The roles given at construction (anything unlisted is ``base``)."""
        return dict(self._roles)

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Each TransformerLens name mapped to the parameters its tensor is computed from."""
        self._build()
        unresolved = [n for n in logical if n not in self._table]
        if unresolved:
            raise Unresolvable(unresolved, "They are not TransformerLens names this bridge's view could trace.")
        return {n: self._table[n] for n in logical}

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        """Physical names as TransformerLens names; a name no TransformerLens name reaches passes through."""
        self._build()
        return list(dict.fromkeys(self._reverse.get(p, p) for p in physical))

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """The traced TransformerLens names with their TransformerLens tensors, in the bridge's order.

        Names TransformerLens synthesizes with no parameter behind them (zero biases of a bias-free projection) are left
        out.
        """
        self._build()
        yield from self._tl_tensors.items()

    def coupled(self, physical: Sequence[str]) -> list[str]:
        """The unnamed storage aliases of ``physical`` and, with ``implicit_norm_thaw``, their blocks' norms (the
        final norm for the embeddings)."""
        self._build()
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

    def depth_groups(self) -> list[list[str]]:
        """The unembedding and final norm first (with any embedding tied to the unembedding), then each block whole
        from the last, then the remaining embeddings."""
        self._build()
        unembed = [n for n in self._table if n.startswith(("unembed.", "ln_final."))]
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
