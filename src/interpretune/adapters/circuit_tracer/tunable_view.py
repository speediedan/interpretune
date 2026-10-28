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
r"""A finetuning-scheduler ``TunableView`` over a circuit-tracer composition's transcoders (#787).

The view's logical names are the interpretune module's own parameter names for the transcoders (``physical``
naming). Every transcoder tensor starts as ``instrument`` (a published artifact: never thawed, optimized or
checkpointed) and may be promoted to ``latent:<name>`` through finetuning-scheduler's promotion, after which a
schedule can thaw it.

**The base model is refused by name unless asked for.** The transcoders were trained on the base model's activations,
so tuning the base model under them changes what they are asked to reconstruct. By default a schedule naming a
base-model parameter gets an ``Unresolvable`` naming that reason. With ``CircuitTracerConfig.tune_base`` the base
model's parameters join the view as ``base`` (the transcoders stay ``instrument`` unless promoted), and a generated
schedule thaws them in pairs from the last, ahead of the transcoder groups, which only a promotion makes trainable.
Attribution after base tuning stays complete, since circuit-tracer's error nodes absorb whatever the transcoders no
longer reconstruct, so the cost of the drift is interpretability rather than correctness:
:class:`~interpretune.adapters.circuit_tracer.drift.TranscoderFitMeter` measures it.

**Lazily loaded weights.** With ``lazy_encoder`` or ``lazy_decoder`` circuit-tracer re-reads $W_{enc}$ or $W_{dec}$
from safetensors on every access, so they are plain tensors, not parameters, and no optimizer can track them. That is
what makes attribution fit on smaller accelerators (only the decoder rows of active features are ever resident), so
the view does not refuse it. Instead, with ``CircuitTracerConfig.lazy_delta_rank`` set, each lazily loaded weight
$W$ gets a trainable low-rank correction $\Delta = A B$ (zero at initialization), and every read circuit-tracer makes
returns $W + \Delta$: whole-matrix reads add $A B$, and decoder-row reads for a feature set $S$ add $A_S B$, so a
sparse read never materializes the full correction. The corrections are ordinary parameters under the transcoder
(``lazy_deltas``), with the same roles as the rest of it. Without a rank the lazily loaded weights are simply absent
from the view, and the module warns once, naming them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

import torch
from torch import nn, Tensor

from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

if TYPE_CHECKING:
    from finetuning_scheduler.tunable_view import TunableSupport

#: The attribute each transcoder carries its low-rank corrections under.
DELTAS_ATTR = "lazy_deltas"

BASE_MODEL_REASON = (
    "they are base-model parameters, which this circuit-tracer composition does not tune: its transcoders were trained"
    " on the untuned base model's activations, so tuning the base moves what they reconstruct. Set"
    " `CircuitTracerConfig.tune_base=True` to tune them as `base` anyway; otherwise only transcoder tensors resolve."
)

PROMOTE_BASE_REASON = (
    "they are base-model parameters: only transcoder tensors are promoted (to a `latent:<name>` role), since the base"
    " model's own tensors are already `base` under `CircuitTracerConfig.tune_base`."
)

BASE_UNREACHABLE_REASON = (
    "`CircuitTracerConfig.tune_base` is set, but none of the base model's parameters are parameters of this module:"
    " the nnsight backend's replacement model is not a torch module, so only its transcoders are registered here,"
    " and tuning would silently leave the base model untouched. Register the base model's torch module on the"
    " interpretune module, or leave `tune_base` unset."
)

DISK_OFFLOAD_REASON = (
    "circuit-tracer's disk offload reloads every offloaded module with `load_state_dict(assign=True)`, replacing its"
    " Parameter objects, so an optimizer built over this view would keep stepping the replaced tensors after an"
    " attribution; set `CircuitTracerConfig.offload` to None or 'cpu' to tune."
)


class LowRankDelta(nn.Module):
    """A trainable rank-``rank`` correction $\\Delta = A B$ to a ``rows`` by ``cols`` weight, zero at
    initialization.

    $A$ is drawn with standard deviation $1 / \\sqrt{r}$ and $B$ starts at zero (the LoRA convention), so the corrected
    weight equals the published one until a step moves $B$. Parameters are float32 whatever the weight's dtype, and the
    correction is cast to the weight's dtype when applied.

    Args:
        rows: The weight's leading dimension (one row per transcoder feature).
        cols: The product of the weight's remaining dimensions.
        rank: The correction's rank.
        device: Where the parameters live (the transcoder's device).
    """

    def __init__(self, rows: int, cols: int, rank: int, device: torch.device | str | None = None) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"A low-rank correction needs rank >= 1, got {rank}.")
        self.A = nn.Parameter(torch.randn(rows, rank, device=device) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(rank, cols, device=device))

    def apply_to(self, weight: Tensor, rows: Any = None) -> Tensor:
        """``weight`` plus the correction for ``rows`` (all rows when ``None`` or a full slice).

        ``weight`` is either the whole weight or the rows ``rows`` selects, in the weight's own trailing shape.
        """
        A = self.A if rows is None or (isinstance(rows, slice) and rows == slice(None)) else self.A[_as_index(rows)]
        return weight + (A @ self.B).to(weight.dtype).reshape(weight.shape)


def _as_index(rows: Any) -> Any:
    if isinstance(rows, (Tensor, slice)):
        return rows
    return torch.as_tensor(rows)


def _deltas(module: nn.Module, name: str) -> Any | None:
    # read from `_modules` directly: this runs inside `__getattr__`, where attribute access would recurse
    deltas = module._modules.get(DELTAS_ATTR)
    return deltas[name] if isinstance(deltas, nn.ModuleDict) and name in deltas else None


class SingleLayerTranscoderWithDeltas(SingleLayerTranscoder):
    """A ``SingleLayerTranscoder`` whose lazily loaded weights read through their low-rank corrections.

    circuit-tracer reads a lazy $W_{enc}$ or $W_{dec}$ in ``__getattr__``, which Python looks up on the type, so the
    corrections are applied by a subclass rather than by wrapping methods on the instance.
    """

    def __getattr__(self, name: str) -> Any:
        value = super().__getattr__(name)
        if name in ("W_enc", "W_dec") and (delta := _deltas(self, name)) is not None:
            value = delta.apply_to(value)
        return value

    def _get_decoder_vectors(self, feat_ids=None):
        vectors = super()._get_decoder_vectors(feat_ids)
        # the lazy path slices the file directly rather than reading `W_dec`, so its rows are corrected here
        if self.lazy_decoder and (delta := _deltas(self, "W_dec")) is not None:
            vectors = delta.apply_to(vectors, feat_ids)
        return vectors


class CrossLayerTranscoderWithDeltas(CrossLayerTranscoder):
    """A ``CrossLayerTranscoder`` whose lazily loaded weights read through their per-layer low-rank corrections."""

    def _get_encoder_weights(self, layer_id=None):
        weights = super()._get_encoder_weights(layer_id)
        deltas = _deltas(self, "W_enc") if self.lazy_encoder else None
        if deltas is None:
            return weights
        if layer_id is None:
            return weights + torch.stack([d.A @ d.B for d in deltas]).to(weights.dtype).reshape(weights.shape)
        return deltas[layer_id].apply_to(weights)

    def _get_decoder_vectors(self, layer_id, feat_ids=None):
        vectors = super()._get_decoder_vectors(layer_id, feat_ids)
        deltas = _deltas(self, "W_dec") if self.lazy_decoder else None
        return vectors if deltas is None else deltas[layer_id].apply_to(vectors, feat_ids)


def lazy_weights(transcoders: nn.Module) -> list[str]:
    """The lazily loaded weights of ``transcoders`` (a ``TranscoderSet`` or a ``CrossLayerTranscoder``), by
    name."""
    if isinstance(transcoders, CrossLayerTranscoder):
        return [n for n, lazy in (("W_enc", transcoders.lazy_encoder), ("W_dec", transcoders.lazy_decoder)) if lazy]
    if isinstance(transcoders, TranscoderSet):
        return [
            f"transcoders.{i}.{n}"
            for i, tc in enumerate(transcoders.transcoders)
            for n, lazy in (("W_enc", tc.lazy_encoder), ("W_dec", tc.lazy_decoder))
            if lazy
        ]
    raise TypeError(_unknown_layout(transcoders))


def _unknown_layout(transcoders: Any) -> str:
    return (
        f"{type(transcoders).__name__} is not a transcoder layout this view knows (a circuit-tracer `TranscoderSet` of"
        " `SingleLayerTranscoder`s, or a `CrossLayerTranscoder`), so its tensors cannot be mapped."
    )


def _swap(module: nn.Module, base: type, with_deltas: type) -> None:
    if type(module) is base:
        module.__class__ = with_deltas
    elif not isinstance(module, with_deltas):
        raise TypeError(
            f"{type(module).__name__} subclasses {base.__name__}; lazy-weight corrections are applied only to"
            f" {base.__name__} itself, since a subclass may read its weights some other way."
        )


def attach_lazy_deltas(transcoders: nn.Module, rank: int) -> list[str]:
    """Give every lazily loaded weight in ``transcoders`` a trainable rank-``rank`` correction.

    Idempotent: a weight that already has a correction keeps it.

    Returns:
        The lazily loaded weights that received a correction, by name (see :func:`lazy_weights`).

    Raises:
        TypeError: For a transcoder layout this view does not know, or a subclass of a known transcoder class.
    """
    if isinstance(transcoders, CrossLayerTranscoder):
        _swap(transcoders, CrossLayerTranscoder, CrossLayerTranscoderWithDeltas)
        device = transcoders.b_enc.device
        n, d_tc, d_model = transcoders.n_layers, transcoders.d_transcoder, transcoders.d_model
        shapes = {
            "W_enc": [d_model] * n if transcoders.lazy_encoder else None,
            "W_dec": [(n - i) * d_model for i in range(n)] if transcoders.lazy_decoder else None,
        }
        deltas = transcoders._modules.setdefault(DELTAS_ATTR, nn.ModuleDict())
        for name, cols in shapes.items():
            if cols is not None and name not in deltas:  # type: ignore[operator]
                deltas[name] = nn.ModuleList(LowRankDelta(d_tc, c, rank, device) for c in cols)  # type: ignore[index]
        return lazy_weights(transcoders)
    if not isinstance(transcoders, TranscoderSet):
        raise TypeError(_unknown_layout(transcoders))
    for tc in transcoders.transcoders:
        if not (tc.lazy_encoder or tc.lazy_decoder):
            continue
        _swap(tc, SingleLayerTranscoder, SingleLayerTranscoderWithDeltas)
        deltas = tc._modules.setdefault(DELTAS_ATTR, nn.ModuleDict())
        for name, lazy in (("W_enc", tc.lazy_encoder), ("W_dec", tc.lazy_decoder)):
            if lazy and name not in deltas:  # type: ignore[operator]
                deltas[name] = LowRankDelta(tc.d_transcoder, tc.d_model, rank, tc.b_enc.device)  # type: ignore[index]
    return lazy_weights(transcoders)


REFIT_SUGGESTION = (
    "Attributing with base-model parameters thawed (`CircuitTracerConfig.tune_base`): the transcoders were trained on"
    " the untuned base model, so the graph stays complete but may route more influence through error nodes. Measure"
    " the transcoders' fit with `interpretune.adapters.circuit_tracer.drift.TranscoderFitMeter`; if it has degraded,"
    " consider re-fitting them by promoting them to `latent:transcoder` and tuning them against the tuned base."
)


@contextmanager
def attribution_frozen(replacement_model: Any) -> Iterator[list[nn.Parameter]]:
    """Run circuit-tracer attribution with every parameter of ``replacement_model`` frozen, restoring each
    parameter's ``requires_grad`` afterwards. Yields the parameters it froze.

    circuit-tracer freezes the replacement model at construction and attributes with ``backward()``, so a parameter
    a schedule has since thawed changes attribution: a thawed embedding fails it outright (its output is no longer a
    leaf whose ``requires_grad`` circuit-tracer can set), and other thawed base-model parameters accumulate attribution
    gradients into ``.grad``, where the next optimizer step would read them as training gradients. Gradients already
    accumulated are left untouched.
    """
    thawed = [p for p in replacement_model.parameters() if p.requires_grad]
    for p in thawed:
        p.requires_grad_(False)
    try:
        yield thawed
    finally:
        for p in thawed:
            p.requires_grad_(True)


class CircuitTracerView:
    """A ``TunableView`` over a circuit-tracer composition: its transcoders, as ``instrument`` tensors.

    It does not inherit from a finetuning-scheduler class: finetuning-scheduler supplies the members it lacks
    (``checkpoint_state_dict``, which leaves the ``instrument`` tensors out, and its loader).

    Args:
        root: The interpretune module.
        transcoders: The replacement model's transcoders, registered on ``root``.
        refusal: A reason to refuse the whole composition (disk offload), or ``None``.
        tune_base: Whether the base model's parameters (every parameter of ``root`` outside the transcoders) join
            the view as ``base``. With none outside the transcoders (the nnsight backend's replacement model is not a
            torch module), the whole composition is refused, naming why.
    """

    def __init__(
        self, root: nn.Module, transcoders: nn.Module, refusal: str | None = None, tune_base: bool = False
    ) -> None:
        from finetuning_scheduler.tunable_view.roles import validate_roles

        lazy_weights(transcoders)  # refuses an unknown layout by name
        self.root = root
        self.transcoders = transcoders
        self.tune_base = tune_base
        ids = {id(p) for p in transcoders.parameters()}
        self._names = [n for n, p in root.named_parameters() if id(p) in ids]
        self._base_names = [n for n, p in root.named_parameters() if id(p) not in ids] if tune_base else []
        # an opt-in that reaches nothing is refused by name rather than tuning the transcoders alone (#859)
        self.refusal = refusal or (BASE_UNREACHABLE_REASON if tune_base and not self._base_names else None)
        if len(self._names) != len(ids):
            raise ValueError(
                f"{len(ids) - len(self._names)} of the transcoders' {len(ids)} parameters are not parameters of"
                f" {type(root).__name__}; register the transcoders on the interpretune module before building its view."
            )
        self._roles = validate_roles(
            dict.fromkeys(self._names, "instrument"),
            (n for n, _ in root.named_parameters()),
            (n for n, _ in root.named_buffers()),
        )

    def support(self) -> TunableSupport:
        """Trainable through the transcoders (``instrument`` until promoted); refused under disk offload."""
        from finetuning_scheduler.tunable_view import TunableSupport
        from finetuning_scheduler.tunable_view.roles import declared_kinds

        if self.refusal:
            return TunableSupport(trainable=False, reason=self.refusal)
        # `latent` too: every transcoder tensor may be promoted to it, and a consumer reads this record once, before
        # any promotion
        roles = declared_kinds(self._roles) | {"latent"}  # declared_kinds always includes `base`
        return TunableSupport(trainable=True, naming_scheme="physical", declared_roles=roles)

    def roles(self) -> Mapping[str, str]:
        """``instrument`` for every transcoder tensor; base-model tensors are unlisted, which is ``base``."""
        return dict(self._roles)

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """The transcoders' parameters (and the base model's, under ``tune_base``), under the interpretune module's
        names."""
        names = set(self._names) | set(self._base_names)
        return ((n, p) for n, p in self.root.named_parameters() if n in names)

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Each name in the view maps to itself; a base-model name outside ``tune_base`` is refused, naming why."""
        from finetuning_scheduler.tunable_view import Unresolvable

        if self.refusal:
            raise Unresolvable(list(logical), self.refusal)
        names = set(self._names) | set(self._base_names)
        params = {n for n, _ in self.root.named_parameters()}
        if base := [n for n in logical if n in params and n not in names]:
            raise Unresolvable(base, BASE_MODEL_REASON)
        if unknown := [n for n in logical if n not in names]:
            raise Unresolvable(unknown, "They are not parameters of this circuit-tracer composition's transcoders.")
        return {n: (n,) for n in logical}

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        """Names pass through unchanged."""
        return list(dict.fromkeys(physical))

    def depth_groups(self) -> list[list[str]]:
        """Under ``tune_base``, the base model's parameters in pairs from the last; then one group per transcoder
        layer, from the last (a cross-layer transcoder is one group)."""
        from finetuning_scheduler.tunable_view.views import pair_groups

        groups = pair_groups(self._base_names) if self._base_names else []
        if isinstance(self.transcoders, CrossLayerTranscoder):
            return groups + ([list(self._names)] if self._names else [])
        for tc in reversed(list(self.transcoders.transcoders)):  # type: ignore[union-attr]
            ids = {id(p) for p in tc.parameters()}
            group = [n for n, p in self.root.named_parameters() if id(p) in ids]
            if group:
                groups.append(group)
        return groups

    def promote(self, physical: Sequence[str], role: str) -> Mapping[str, str]:
        """Promote transcoder tensors (only; a base-model name is refused) to ``role``, keeping their names."""
        from finetuning_scheduler.tunable_view import CanonicalView, Unresolvable

        self.to_physical(physical)  # refuses unknown names, and base-model names outside `tune_base`
        if base := [n for n in physical if n in set(self._base_names)]:
            raise Unresolvable(base, PROMOTE_BASE_REASON)
        return CanonicalView(self.root, roles=self._roles).promote(physical, role)
