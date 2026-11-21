r"""Helpers for the block-sparse featurizer example (#866): #708's featurizers for gemma-3-1b-it, loaded from the
Hub.

A block-sparse featurizer (Goodfire's ``block-sparse-featurizer``, its vanilla variant) encodes a normalized activation
:math:`\tilde x = (x - \mu) / s` into per-block codes :math:`a = \tilde x W_{enc} + b_{enc}`, keeps the :math:`L_0`
blocks of largest norm and zeroes the rest, and decodes :math:`\hat x = s (a W_{dec}) + \mu`. Each block is a few
latent coordinates (three here), so a block names a small subspace rather than one direction.

#708 trained one on gemma-3-1b-it's layer-17 residual stream and one on the same activations mapped into J-space (the
``neuronpedia/jacobian-lens`` lens). They are published, with their measures, in the private model repository
:data:`REPO` at :data:`REVISION`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

#: The Hub repository of #708's layer-17 featurizers, and the revision this example pins.
REPO = "speediedan/gemma-3-1b-it-bsf-l17"
REVISION = "fdf5e15326dbc4db3bb35e3fab59cf4164bd3a77"
#: The spaces the featurizers were trained in: the residual stream, and J-space through the Jacobian lens.
SPACES = ("resid", "jspace")


def block_topk(norms: torch.Tensor, l0: int) -> torch.Tensor:
    """A ``{0, 1}`` mask keeping each row's ``l0`` largest block norms, ``(items, blocks)``."""
    keep = norms.topk(min(l0, norms.shape[-1]), dim=-1).indices
    return torch.zeros_like(norms).scatter_(-1, keep, 1.0)


class BlockSparseFeaturizer(nn.Module):
    """A vanilla block-sparse featurizer with its input normalization, as #708 saved it.

    Args:
        state: The checkpoint dict: ``state_dict`` (``W_enc``, ``b_enc``, ``W_dec``), ``group_size``, ``n_groups``,
            ``l0``, ``mean``, ``scale``, ``layer`` and ``space``.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        weights = state["state_dict"]
        self.W_enc = nn.Parameter(weights["W_enc"].float())
        self.b_enc = nn.Parameter(weights["b_enc"].float())
        self.W_dec = nn.Parameter(weights["W_dec"].float())
        self.register_buffer("mean", torch.as_tensor(state["mean"]).float())
        self.register_buffer("scale", torch.as_tensor(float(state["scale"])))
        self.group_size, self.n_groups, self.l0 = int(state["group_size"]), int(state["n_groups"]), int(state["l0"])
        self.layer, self.space = int(state["layer"]), str(state["space"])
        if self.W_dec.shape[0] != self.group_size * self.n_groups or self.W_enc.shape[1] != self.W_dec.shape[0]:
            raise ValueError(
                f"A featurizer of {self.n_groups} blocks of {self.group_size} needs {self.n_groups * self.group_size}"
                f" latent coordinates; its encoder has {self.W_enc.shape[1]} and its decoder {self.W_dec.shape[0]}"
            )

    def codes(self, x: torch.Tensor) -> torch.Tensor:
        """Every block's code before the top-:math:`L_0` gate, ``(items, blocks, block size)``."""
        a = ((x.float() - self.mean) / self.scale) @ self.W_enc + self.b_enc
        return a.view(*a.shape[:-1], self.n_groups, self.group_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The gated codes: the :math:`L_0` blocks of largest norm kept, the rest zero."""
        a = self.codes(x)
        return a * block_topk(a.norm(dim=-1), self.l0).unsqueeze(-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """The reconstruction at the activation's own scale, from ``(..., blocks, block size)`` codes."""
        return (codes.flatten(-2) @ self.W_dec) * self.scale + self.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def block(self, b: int) -> torch.Tensor:
        """Block ``b``'s decoder rows, ``(block size, width)``."""
        return self.W_dec[b * self.group_size : (b + 1) * self.group_size]

    def binding(self, name: str, *, point: str = "hook_resid_post", space: str | None = None) -> Any:
        """The featurizer as a venngram latent binding at ``blocks.{layer}.{point}``, one addressable block per
        group, so ``features:<name> & layers:<layer> & units:[b]`` selects block ``b``.

        ``space`` names the space its decoder rows are in (``None`` for the site's own activation space): pass the
        linear space registered for the Jacobian lens for the J-space featurizer.
        """
        from interpretune.analysis.venngram import LatentBinding

        return LatentBinding(name, self.layer, point, self, self.W_dec.detach(), self.group_size, space)


def load_featurizer(
    space: str, *, repo_id: str = REPO, revision: str = REVISION, device: str | torch.device = "cpu"
) -> BlockSparseFeaturizer:
    """Download one of #708's featurizers (``space`` is ``resid`` or ``jspace``) at a pinned revision."""
    from huggingface_hub import hf_hub_download

    if space not in SPACES:
        raise ValueError(f"#708 trained featurizers in {SPACES}, not {space!r}")
    path = hf_hub_download(repo_id, f"bsf_{space}_L17_block3.pt", revision=revision)
    state = torch.load(path, map_location="cpu", weights_only=False)
    return BlockSparseFeaturizer(state).to(device).eval()


@dataclass
class _Tiny:
    """A small random checkpoint in #708's layout, for tests."""

    d: int = 8
    n_groups: int = 6
    group_size: int = 3
    l0: int = 2

    def state(self, seed: int = 0) -> dict[str, Any]:
        g = torch.Generator().manual_seed(seed)
        width = self.n_groups * self.group_size
        return {
            "state_dict": {
                "W_enc": torch.randn(self.d, width, generator=g),
                "b_enc": torch.randn(width, generator=g),
                "W_dec": torch.randn(width, self.d, generator=g),
            },
            "group_size": self.group_size,
            "n_groups": self.n_groups,
            "l0": self.l0,
            "mean": torch.randn(self.d, generator=g),
            "scale": 2.0,
            "layer": 3,
            "space": "resid",
        }
