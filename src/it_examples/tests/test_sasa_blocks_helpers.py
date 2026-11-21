"""CPU tests for the SASA blocks notebook's helpers: the release as an attached latent model, and native edits."""

from __future__ import annotations

import math

import pytest
import torch

from it_examples.utils.sasa_blocks_helpers import SASARelease, native_matches_functional

LAYER, GROUPS, RANK = 3, 16, 4


@pytest.fixture(scope="module")
def gpt2():
    from nnsight import TransformersModel

    from interpretune.adapters.nnsight.backends import NNsightModelBackend
    from interpretune.analysis.backends.hook_mapping import HookNameResolver

    lm = TransformersModel("gpt2", task="text-generation", device_map="cpu", dispatch=True)
    lm._module.requires_grad_(False)
    with torch.no_grad():
        yield lm, NNsightModelBackend(HookNameResolver(type(lm._module).__name__))


def _release(d_model: int) -> SASARelease:
    """A random release with an open gate, so every group carries codes."""
    gen = torch.Generator().manual_seed(0)
    w_enc = torch.randn(d_model, GROUPS * RANK, generator=gen) / math.sqrt(d_model)
    w_dec = 0.3 * torch.randn(GROUPS * RANK, d_model, generator=gen)
    return SASARelease("sasa", LAYER, w_enc, torch.zeros(GROUPS * RANK), w_dec, RANK, 2, threshold=0.0)


def _catalog(release: SASARelease, group: int):
    from interpretune.analysis.venngram import LatentCatalog, LatentChart, LatentFrame

    catalog = LatentCatalog([release.binding()])
    target = catalog.resolve(f"features:sasa & units:[{group}]", n_layers=12).targets[0]
    rows = torch.linalg.qr(catalog.basis(target).double().T)[0].T.float()
    frame = LatentFrame("f", "v1", target, rows)
    ring = LatentChart(
        "ring", "v1", "pca_plane", target, "f", "v1", 2, {1: 2 * math.pi},
        {"origin": torch.zeros(RANK), "axes": torch.eye(RANK)[:2]},
    )  # fmt: skip
    return LatentCatalog([release.binding()], frames=[frame], charts=[ring]), target


def test_the_release_reconstructs_its_block_contribution():
    """A block's decoded codes are the block's contribution at the activation scale, zero where its gate is
    shut."""
    release = _release(32)
    h = torch.randn(5, 32)
    codes = release.encode(h)
    for group in (0, 7):
        block = codes[:, group * RANK : (group + 1) * RANK] @ release.block(group)
        torch.testing.assert_close(block, release.contribution(h, group))
    release.threshold = float("inf")
    assert not release.encode(h).any()


@pytest.mark.parametrize(
    "op, args",
    [
        ("chart_rotate", {"name": "ring", "by": {1: 1.0}}),
        ("frame_replace", {"name": "f", "coordinates": [1, -2, 0, 3]}),
    ],
)
def test_a_native_block_edit_matches_the_same_edit_applied_functionally(gpt2, op, args):
    """The edit runs at the attached release's latent sub-hook with its error term, and gives the logits of writing
    the state plus the catalog's edit of the block's contribution back with a per-row ``replace``."""
    from interpretune.analysis.venngram import latent_edit_spec

    lm, backend = gpt2
    release = _release(lm._module.config.n_embd)
    catalog, target = _catalog(release, group=5)
    hook, spec = latent_edit_spec(catalog, target, op, version="v1", **args)
    assert hook == f"blocks.{LAYER}.hook_resid_pre.hook_sae_acts_post"
    batch = {"input_ids": lm.tokenizer(["The capital of France is", "The capital of Spain is"])["input_ids"]}
    batch = {"input_ids": torch.tensor(batch["input_ids"])}
    result = native_matches_functional(
        lm, backend, release, catalog, target, batch, 2, f"blocks.{LAYER}.hook_in", hook, spec
    )
    assert result["edit moved"] > 0.1
    assert result["native vs functional"] < 1e-3


def test_attached_with_a_zero_dose_the_release_leaves_the_forward_pass_unchanged(gpt2):
    from interpretune.analysis.venngram import latent_edit_spec
    from it_examples.utils.sasa_blocks_helpers import at_column

    lm, backend = gpt2
    release = _release(lm._module.config.n_embd)
    catalog, target = _catalog(release, group=5)
    hook, spec = latent_edit_spec(catalog, target, "chart_rotate", version="v1", name="ring", by={1: 1.0}, dose=0.0)
    batch = {"input_ids": torch.tensor(lm.tokenizer(["The capital of France is"])["input_ids"])}
    clean, attached = backend.fwd_w_intervention(lm, batch, {hook: at_column(spec, 2)}, latent_model_handles=[release])
    torch.testing.assert_close(attached, clean, atol=1e-4, rtol=0)
