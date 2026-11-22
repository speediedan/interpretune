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

import types
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

    It is also a latent model a model backend can attach (``latent_model_handles=[featurizer]``) at
    ``blocks.{layer}.hook_resid_post``, so an edit of one block (``block_edit_spec``) runs natively at that hook's
    latent sub-hook. Its latents are the gated codes, flat (``(..., blocks * block size)``), and with the error term an
    attached pass with no edit is the clean pass. The decoder is affine, so an edit of a block's codes moves the
    activation by exactly that block's decoded change, at the activation's own scale.

    Args:
        state: The checkpoint dict: ``state_dict`` (``W_enc``, ``b_enc``, ``W_dec``), ``group_size``, ``n_groups``,
            ``l0``, ``mean``, ``scale``, ``layer`` and ``space``.
        io_dtype: The dtype of the model's activations, which :meth:`decode` writes back.
    """

    use_error_term = True
    # registered buffers, declared so a type checker reads them as tensors rather than `Tensor | Module`
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, state: dict[str, Any], io_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.io_dtype = io_dtype
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

    @property
    def cfg(self) -> Any:
        """The hook the featurizer reads and writes, where a model backend splices it."""
        return types.SimpleNamespace(metadata=types.SimpleNamespace(hook_name=f"blocks.{self.layer}.hook_resid_post"))

    def codes(self, x: torch.Tensor) -> torch.Tensor:
        """Every block's code before the top-:math:`L_0` gate, ``(..., blocks, block size)``."""
        a = ((x.float() - self.mean) / self.scale) @ self.W_enc + self.b_enc
        return a.view(*a.shape[:-1], self.n_groups, self.group_size)

    def block_codes(self, x: torch.Tensor) -> torch.Tensor:
        """The gated codes by block, ``(..., blocks, block size)``: the :math:`L_0` blocks of largest norm kept,
        the rest zero."""
        a = self.codes(x)
        return a * block_topk(a.norm(dim=-1), self.l0).unsqueeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The gated codes, flat: ``(..., blocks * block size)``, block :math:`b` at coordinates :math:`bs` to
        :math:`bs + s - 1`."""
        return self.block_codes(x).flatten(-2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """The reconstruction at the activation's own scale and dtype, from flat codes."""
        return ((codes.float() @ self.W_dec) * self.scale + self.mean).to(self.io_dtype)

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
    space: str,
    *,
    repo_id: str = REPO,
    revision: str = REVISION,
    device: str | torch.device = "cpu",
    io_dtype: torch.dtype = torch.float32,
) -> BlockSparseFeaturizer:
    """Download one of #708's featurizers (``space`` is ``resid`` or ``jspace``) at a pinned revision; ``io_dtype``
    is the dtype of the model's activations, which an attached featurizer writes back."""
    from huggingface_hub import hf_hub_download

    if space not in SPACES:
        raise ValueError(f"#708 trained featurizers in {SPACES}, not {space!r}")
    path = hf_hub_download(repo_id, f"bsf_{space}_L17_block3.pt", revision=revision)
    state = torch.load(path, map_location="cpu", weights_only=False)
    return BlockSparseFeaturizer(state, io_dtype=io_dtype).to(device).eval()


class KeptError(nn.Module):
    r"""A featurizer spliced in with its own reconstruction error kept, to tune a few of its blocks.

    Attached with ``replace`` (:func:`~interpretune.tuning.latents.attach_latent_model`), it writes
    :math:`x + f_\theta(x) - f_0(x)`, where :math:`f_\theta` is the featurizer with its adapters and :math:`f_0` the
    featurizer as loaded. While the adapters are zero the activation passes through unchanged, and a trained adapter
    moves it by exactly the change it makes to the reconstruction. ``replace`` alone would run the model on the
    reconstruction, and ``replace_with_error`` keeps the activation's value whatever the adapters learn (its error term
    is detached from the same reconstruction), so neither tunes a few blocks of a featurizer that stays as it is.
    """

    # the loaded weights, registered as buffers below
    W_enc_0: torch.Tensor
    b_enc_0: torch.Tensor
    W_dec_0: torch.Tensor

    def __init__(self, featurizer: BlockSparseFeaturizer) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.layer, self.group_size = featurizer.layer, featurizer.group_size
        # the loaded weights, as buffers: not parameters, so an adapter neither resolves nor trains them
        for leaf in ("W_enc", "b_enc", "W_dec"):
            self.register_buffer(f"{leaf}_0", getattr(featurizer, leaf).detach().clone(), persistent=False)
        self._x: torch.Tensor | None = None

    @property
    def W_dec(self) -> torch.Tensor:
        """The featurizer's decoder rows, adapters included."""
        return self.featurizer.W_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._x = x
        return self.featurizer.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        x, self._x = self._x, None
        if x is None:
            raise RuntimeError("KeptError decodes the codes of the activation it last encoded; encode first")
        f = self.featurizer
        with torch.no_grad():
            a = ((x.float() - f.mean) / f.scale) @ self.W_enc_0 + self.b_enc_0
            a = a.view(*a.shape[:-1], f.n_groups, f.group_size)
            a = a * block_topk(a.norm(dim=-1), f.l0).unsqueeze(-1)
            loaded = (a.flatten(-2) @ self.W_dec_0) * f.scale
        tuned = (codes.float() @ f.W_dec) * f.scale
        return (x.float() + tuned - loaded).to(x.dtype)

    def binding(self, name: str) -> Any:
        """The spliced featurizer as a latent binding, for
        :func:`~interpretune.tuning.weights.attach_feature_adapter`:

        ``features:<name> & layers:<layer> & units:[b]`` selects block ``b``.
        """
        from interpretune.analysis.venngram import LatentBinding

        return LatentBinding(name, self.layer, "hook_resid_post", self, self.featurizer.W_dec.detach(), self.group_size)


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


#: The blocks that carry the answer in #708's evaluation set (377 RTE answer positions), ranked by the AUROC of the
#: block's code norm against the model answering Yes: one toward each answer in each space.
DECISION_BLOCKS = {"resid": {"yes": 327, "no": 3504}, "jspace": {"yes": 1048, "no": 415}}


def decision_book(
    *, model_revision: str, blocks: dict[str, dict[str, int]] | None = None, jlens_space: str = "jlens"
) -> dict[str, Any]:
    """A checkpoint address book naming the decision blocks of both featurizers.

    A block index means something for one featurizer trained on one checkpoint, so the book pins
    ``google/gemma-3-1b-it`` at ``model_revision`` (unprocessed basis). Each set selects one block,
    ``features:bsf_<space> & layers:17 & units:[b]``, and the J-space featurizer's sets carry ``space`` so a consumer
    knows their rows are in the lens's coordinates.
    """
    blocks = DECISION_BLOCKS if blocks is None else blocks
    sets: dict[str, Any] = {}
    for space, by_answer in blocks.items():
        for answer, block in by_answer.items():
            entry: dict[str, Any] = {"features": f"bsf_{space}", "layers": [17], "units": [int(block)]}
            if space == "jspace":
                entry["space"] = jlens_space
            sets[f"{space}_{answer}"] = entry
    return {
        "schema_version": 1,
        "name": "gemma-3-1b-it-bsf-l17-decision",
        "description": "The blocks of #708's layer-17 featurizers that carry gemma-3-1b-it's Yes or No on RTE",
        "portability": "checkpoint",
        "applies_to": {
            "architectures": ["Gemma3ForCausalLM"],
            "models": [{"id": "google/gemma-3-1b-it", "revision": model_revision, "basis": "unprocessed"}],
        },
        "sets": sets,
    }
