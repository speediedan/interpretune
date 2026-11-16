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
r"""Readouts: the scalar a backward pass differentiates.

A gradient intervention edits the gradient of one scalar of the forward pass, and which scalar decides what an edit
means: adding a vector to the gradient of a logit difference and to the gradient of a probe's projection are different
operations. A :class:`Readout` names that scalar. It is a function of the logits and of the activations at the sites it
names, which a model backend keeps from the same forward pass the backward runs from, so a readout of an intermediate
activation, or of a latent model's feature, is differentiated through the graph a logit readout is.

This is the callable form, for use in process. The declared form, a weighted sum of terms each applying a registered
reduction to an addressed input, is venngram's to declare, and interpretune's to execute.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

#: A readout's computation: ``(logits, activations) -> scalar``, where ``activations`` maps each site the readout names
#: to its activation from the same forward pass.
ReadoutFn = Callable[[Any, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Readout:
    """A scalar of a forward pass: ``fn(logits, activations)``, with ``activations`` holding the activation at each of
    ``sites`` from that pass.

    ``sites`` are hook names, or patterns with ``*``, resolved as a gradient capture's are: ``activations`` is keyed by
    each concrete hook name a pattern matches, and by the name as written for an exact one. A site under an attached
    latent model (``<hook>.hook_sae_acts_post``) reads that model's stage value. ``fn`` must use standard PyTorch ops
    only, so a traced backend can apply it to its proxies, and it must return a scalar, the value a backward pass starts
    from; anything else is refused by name rather than reduced.
    """

    fn: ReadoutFn
    sites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(f"a readout's `fn` must be callable as fn(logits, activations), got {type(self.fn)!r}")
        sites = (self.sites,) if isinstance(self.sites, str) else tuple(self.sites)
        if bad := [site for site in sites if not isinstance(site, str) or not site]:
            raise TypeError(f"a readout's sites are hook names or patterns, got {bad!r}")
        object.__setattr__(self, "sites", sites)

    def __call__(self, logits: Any, activations: Mapping[str, Any] | None = None) -> Any:
        """The readout's value for one forward pass's logits and activations, refused unless it is a scalar."""
        value = self.fn(logits, dict(activations or {}))
        if getattr(value, "ndim", None) != 0:
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            raise ValueError(
                f"a readout must return a scalar tensor, the value a backward pass starts from; it returned {shape}. "
                "Reduce it in the readout (a sum over rows, say), so the reduction is declared rather than guessed"
            )
        return value


def as_readout(readout: Readout | Callable[[Any], Any]) -> Readout:
    """``readout`` as a :class:`Readout`: a callable of the logits alone is a readout naming no sites."""
    if isinstance(readout, Readout):
        return readout
    if callable(readout):
        of_logits = readout

        def fn(logits: Any, activations: Mapping[str, Any]) -> Any:
            return of_logits(logits)

        return Readout(fn)
    raise TypeError(f"a readout is a Readout or a callable of the logits, got {type(readout)!r}")


def resolve_readout_sites(readout: Readout, available: Mapping[str, str]) -> dict[str, str]:
    """Each hook name ``readout`` reads, mapped to the key its activation is handed under: the site as written for
    an exact name (in any spelling the point vocabulary knows), and the concrete name for each match of a pattern.

    ``available`` is a backend's map of the hook names it exposes, as for intervention patterns; a site matching none
    of them is refused by name.
    """
    from interpretune.analysis.backends.interventions import expand_intervention_patterns

    keys: dict[str, str] = {}
    for site, matches in expand_intervention_patterns(list(readout.sites), available).items():
        for name in matches:
            keys[name] = site if "*" not in site and len(matches) == 1 else name
    return keys


def last_token_logit_readout(target_ids: Sequence[int] | torch.Tensor | None = None) -> Readout:
    """The last-token logits of ``target_ids`` summed over them and over the batch, or, with none, each row's top
    last-token logit summed over the batch: ``model_bwd_intervention``'s readout when it is given none."""
    ids = None if target_ids is None else torch.as_tensor(target_ids, dtype=torch.long).reshape(-1)

    def fn(logits: Any, activations: Mapping[str, Any]) -> Any:
        last = logits[:, -1, :]
        if ids is None:
            return last.max(dim=-1).values.sum()
        return last[:, ids.to(last.device)].sum()

    return Readout(fn)


__all__ = ["Readout", "ReadoutFn", "as_readout", "last_token_logit_readout", "resolve_readout_sites"]
