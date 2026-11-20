"""The latent family's tensor ops: one executor for every frame and chart edit of a feature block.

A latent edit op's spec carries its compiled edit in ``params["latent_edit"]``
(:func:`interpretune.analysis.venngram.latent_edit_spec` builds it). The executor reads the edit through three
names only, so this module depends on no venngram internals: ``decoder_rows()`` (the block's decoder rows),
``edit_values(values)`` (the catalog's edit of the block's decoded values) and ``rank_rtol`` (the relative
singular-value threshold the edit's frame counts rank with).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from interpretune.analysis.backends import InterventionSpec

_EDIT_NAMES = ("decoder_rows", "edit_values", "rank_rtol")


def latent_edit_executor(region: torch.Tensor, operand: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
    """A block's activations ``(rows, block_size)`` moved so their decoded values take the compiled edit, by the
    minimum-norm change, scaled by the dose ``operand``.

    Raises:
        ValueError: For a spec carrying no compiled edit, a region that is not the block's columns, or an operand that
            is not a scalar dose.
    """
    edit = (spec.params or {}).get("latent_edit")
    if edit is None or any(not hasattr(edit, name) for name in _EDIT_NAMES):
        raise ValueError(
            f"op {spec.op!r} runs a compiled latent edit, and this spec carries none in params['latent_edit'] (an"
            f" object with {list(_EDIT_NAMES)}); build it with interpretune.analysis.venngram.latent_edit_spec"
        )
    rows = edit.decoder_rows().detach()
    if region.shape[-1] != rows.shape[0]:
        raise ValueError(
            f"latent edit {spec.op!r} needs the block's {rows.shape[0]} columns, got a region of width"
            f" {region.shape[-1]}; latent_edit_spec selects them"
        )
    if operand.numel() != 1:
        raise ValueError(
            f"latent edit {spec.op!r} takes its dose, a scalar, as its operand; got {tuple(operand.shape)}"
        )
    inverse = torch.linalg.pinv(rows.to(torch.float64), rtol=edit.rank_rtol)
    rows, inverse = rows.to(region), inverse.to(region)
    values = region @ rows
    return region + operand.reshape(()) * ((edit.edit_values(values) - values) @ inverse)


class _ProbeEdit:
    """A frame coordinate replacement on a random three-row block in a five-dimensional space: enough to hold the
    shared executor to the tensor-op contract at load, without a catalog."""

    rank_rtol = 1e-6

    def __init__(self, generator: torch.Generator) -> None:
        self._rows = torch.randn(3, 5, generator=generator, dtype=torch.float64)
        self._frame = torch.linalg.qr(self._rows.T).Q.T
        self._coordinates = torch.randn(3, generator=generator, dtype=torch.float64)

    def decoder_rows(self) -> torch.Tensor:
        return self._rows

    def edit_values(self, values: torch.Tensor) -> torch.Tensor:
        frame, coordinates = self._frame.to(values), self._coordinates.to(values)
        return values + (coordinates - values @ frame.T) @ frame


def probe_latent_edit(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """The contract probe every latent edit op names: a block region, a dose and a stand-in edit."""
    edit = _ProbeEdit(generator)
    return torch.randn(4, 3, generator=generator), torch.tensor(0.75), {"latent_edit": edit}
