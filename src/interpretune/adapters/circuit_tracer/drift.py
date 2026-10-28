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
r"""How well a set of transcoders still fits a model's MLPs, per layer: the drift measures of interpretune#709.

Fixed transcoders reconstruct a fine-tuned model's MLPs less well than the model they were trained on, error nodes
grow, and attribution graphs change for reasons unrelated to how the model computes. :class:`TranscoderFitMeter`
measures, for each layer $\ell$ over a sample of positions:

- the fraction of variance unexplained of the MLP output $y$ by the transcoder's reconstruction $\hat y$, with
  $\bar y$ the mean output:
  $\mathrm{FVU}_\ell = \sum \lVert y - \hat y \rVert^2 / \sum \lVert y - \bar y \rVert^2$;
- $L_0$, the mean number of active features per position;
- dead features (active at no sampled position) and dense ones (active at more than ``dense_threshold`` of them).

Activations are accumulated chunk by chunk, so a sample larger than memory is measured by calling
:meth:`TranscoderFitMeter.update` once per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder


@dataclass(frozen=True)
class LayerFit:
    """One layer's fit.

    Attributes:
        layer: The layer index.
        fvu: Fraction of variance unexplained of the MLP output.
        l0: Mean number of active features per position.
        dead: Features active at no sampled position.
        dense: Features active at more than the meter's ``dense_threshold`` of sampled positions.
        positions: Positions sampled.
    """

    layer: int
    fvu: float
    l0: float
    dead: int
    dense: int
    positions: int


class TranscoderFitMeter:
    """Accumulates the per-layer fit of ``transcoders`` (a circuit-tracer ``TranscoderSet`` or
    ``CrossLayerTranscoder``) over chunks of MLP inputs and outputs.

    Args:
        transcoders: The transcoders, whose ``encode`` and ``decode`` take and return one tensor per layer.
        dense_threshold: The fraction of positions above which an active feature counts as dense.
    """

    def __init__(self, transcoders: TranscoderSet | CrossLayerTranscoder, dense_threshold: float = 0.1) -> None:
        if not 0 < dense_threshold < 1:
            raise ValueError(f"`dense_threshold` is a fraction of positions in (0, 1), got {dense_threshold}.")
        self.transcoders = transcoders
        self.dense_threshold = dense_threshold
        self._positions = 0
        self._sq_err: Tensor | None = None
        self._sum: Tensor | None = None
        self._sum_sq: Tensor | None = None
        self._active: Tensor | None = None
        self._fires: Tensor | None = None

    @torch.no_grad()
    def update(self, mlp_inputs: Tensor, mlp_outputs: Tensor) -> None:
        """Add one chunk of positions.

        Args:
            mlp_inputs: Each layer's transcoder input, shaped ``(n_layers, n_positions, d_model)``.
            mlp_outputs: Each layer's MLP output at the same positions, in the same shape.
        """
        if mlp_inputs.shape != mlp_outputs.shape or mlp_inputs.dim() != 3:
            raise ValueError(
                "MLP inputs and outputs must share one (n_layers, n_positions, d_model) shape, got"
                f" {tuple(mlp_inputs.shape)} and {tuple(mlp_outputs.shape)}."
            )
        acts = self.transcoders.encode(mlp_inputs)
        recon = self.transcoders.decode(acts, mlp_inputs).float()
        if recon.shape != mlp_outputs.shape:
            # circuit-tracer's cross-layer decode sizes its output by the last position with an active feature
            raise ValueError(
                f"The transcoders' reconstruction has shape {tuple(recon.shape)} for MLP outputs of shape"
                f" {tuple(mlp_outputs.shape)}; a cross-layer transcoder drops trailing positions with no active"
                " feature, so measure a chunk whose last position has one."
            )
        y = mlp_outputs.float()
        active = acts != 0
        chunk = (
            ((y - recon) ** 2).sum(dim=(1, 2)),
            y.sum(dim=1),
            (y**2).sum(dim=(1, 2)),
            active.sum(dim=(1, 2)).double(),
            active.sum(dim=1).double(),
        )
        if self._sq_err is None:
            self._sq_err, self._sum, self._sum_sq, self._active, self._fires = (c.double() for c in chunk)
        else:
            for name, c in zip(("_sq_err", "_sum", "_sum_sq", "_active", "_fires"), chunk):
                setattr(self, name, getattr(self, name) + c.double())
        self._positions += mlp_inputs.shape[1]

    def compute(self) -> list[LayerFit]:
        """Each layer's fit over every position added so far.

        Raises:
            ValueError: If no position was added.
        """
        if self._sq_err is None or self._positions == 0:
            raise ValueError("No positions were added; call `update` with at least one chunk first.")
        assert (
            self._sum is not None and self._sum_sq is not None and self._active is not None and self._fires is not None
        )
        n = self._positions
        # total variance about the per-dimension mean: sum ||y||^2 - n ||mean y||^2
        variance = self._sum_sq - (self._sum**2).sum(dim=-1) / n
        frequency = self._fires / n
        return [
            LayerFit(
                layer=layer,
                fvu=float(self._sq_err[layer] / variance[layer]),
                l0=float(self._active[layer] / n),
                dead=int((frequency[layer] == 0).sum()),
                dense=int((frequency[layer] > self.dense_threshold).sum()),
                positions=n,
            )
            for layer in range(self._sq_err.shape[0])
        ]
