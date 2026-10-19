"""Latent projection and ablation across device placement (#795).

The G-6 contract: a latent projection and an ablation run with the MODEL SPLIT across two devices must reproduce
the same run on one GPU of the same machine, with placement asserted on the outputs of operations that cross devices,
and with parameter and optimizer movement asserted where the exercised path owns parameters.

Three layers of evidence, each able to fail on the defect it names:

- A single-device CPU reference for the two in-tree latent ops (``ablate_sae_latent``, ``project_embeddings``),
  seeded so placement is the only variable. These run everywhere.
- Single-device CUDA inputs to those ops, compared EXACTLY with the CPU reference: both ops compute the same values
  from bit-identical float32 input (the projection on CPU by contract, the ablation as an index write), so exact
  equality is the true statement and the one that can fail.
- A real model split across two devices: gpt2 loaded with an explicit dict ``device_map`` (blocks 0 to 5 and the
  tied embedding/unembedding on ``cuda:0``, blocks 6 to 11 and the final norm on ``cuda:1``, so a forward crosses
  devices twice and a backward crosses back), with ``gpt2-small-res-jb`` SAEs spliced in at a block on each device.
  A dict rather than ``max_memory`` keeps the split identical on heterogeneous and identical cards. The reference is
  the same model and SAEs on ``cuda:0`` alone.

Nothing here moves a tensor to make a placement work: the splice hooks consume activations where the block produced
them and the SAE weights where they were declared, so a placement defect raises or fails an assertion rather than
being repaired silently.

Ruling on "parameter or optimizer movement" (this issue's checklist): both. The two latent ops own no parameters;
the parameters on the latent path are the SAE's. Under the split, a stateful optimizer (Adam) stepping the SAE
spliced on ``cuda:0`` must (a) change the parameters, (b) keep every parameter and every per-parameter state tensor
on the parameter's device, while the gradient reaching them has crossed from ``cuda:1``, and (c) see the same
gradient as the single-GPU reference.

This is latent (SAE) projection/ablation acceptance, distinct from J-space fine-tuning. The CUDA tests are selected
by the cuda-marked phase (``IT_RUN_CUDA_TESTS=1``, see ``tests/conftest.py``); collecting them is not a hardware
pass. Nothing here names an NVIDIA-only surface, so this file is the ROCm build-day target unchanged; ROCm execution
stays pending until AMD hardware is available.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial

import pytest
import torch

from interpretune.analysis.ops.bundled.sae.sae_ops import ablate_sae_latent
from interpretune.utils.latent_state_projection import project_embeddings
from tests.runif import RunIf

MODEL = "gpt2"
SAE_RELEASE = "gpt2-small-res-jb"
EARLY_LAYER, LATE_LAYER = 1, 9  # blocks on cuda:0 and cuda:1 under SPLIT_MAP; both SAEs are in the Hub warm manifest
SPLIT_AT = 6
PROMPTS = (
    "The quick brown fox jumps over the lazy dog because it",
    "Paris is the capital of France and its river is the",
)
# float32 throughout, with TF32 disabled: the split runs its second half on a different card (and architecture, on
# a heterogeneous host) than the reference, so GEMMs dispatch to different kernels and accumulate in a different
# order. That reorders float32 rounding (relative 1e-7 per op) through up to twelve blocks, never more: a placement
# defect (a wrong device, a dropped or duplicated block, a lost hook) moves these values by orders of magnitude more.
ATOL = RTOL = 1e-3


def _split_map(first: str, second: str) -> dict[str, str]:
    return {
        "transformer.wte": first,
        "transformer.wpe": first,
        "transformer.drop": first,
        **{f"transformer.h.{i}": first if i < SPLIT_AT else second for i in range(12)},
        "transformer.ln_f": second,
        "lm_head": first,  # tied to wte, so it stays with it: the logits cross back to the first device
    }


def _embeddings(device: str, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(16, 8, generator=gen).to(device)


def _acts(device: str, seed: int = 11) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(3, 5, 6, generator=gen).to(device)


def _project_seeded(values: torch.Tensor) -> torch.Tensor:
    # ``project_embeddings``' PCA path draws on ``torch.pca_lowrank``'s randomized sampling
    # without a seed (measured 2026-10-19: two unseeded calls on one input differ in every
    # coordinate). A placement comparison therefore seeds immediately before each call, so
    # the only variable left is placement. The unseeded nondeterminism itself is recorded
    # in ``tests/gpu_ci/ROCM_TRIAGE.md`` as a finding, not silently relied on either way.
    torch.manual_seed(1234)
    result = project_embeddings(values, method="pca", n_components=3)
    return torch.as_tensor(result.coordinates)


def _reference_projection() -> torch.Tensor:
    return _project_seeded(_embeddings("cpu"))


SEQ_POS = (0, 2, 4)


def _reference_ablation() -> torch.Tensor:
    return ablate_sae_latent(_acts("cpu"), hook=None, latent_idx=2, seq_pos=torch.tensor(SEQ_POS))


# --------------------------------------------------------------------------------------
# Single-device reference (CPU): the baseline the CUDA inputs are compared against.
# --------------------------------------------------------------------------------------


def test_latent_projection_cpu_reference_is_deterministic():
    first, second = _reference_projection(), _reference_projection()
    assert torch.equal(first, second)
    assert first.shape == (16, 3)


def test_latent_ablation_cpu_reference_zeroes_only_the_declared_latent():
    before = _acts("cpu")
    acts = before.clone()
    after = ablate_sae_latent(acts, hook=None, latent_idx=2, seq_pos=torch.tensor(SEQ_POS))
    # The op writes in place and returns its input: callers that keep the pre-ablation value must clone first.
    assert after is acts
    assert torch.all(after[torch.arange(3), torch.tensor(SEQ_POS), 2] == 0)
    keep = torch.ones_like(before, dtype=torch.bool)
    keep[torch.arange(3), torch.tensor(SEQ_POS), 2] = False
    assert torch.equal(after[keep], before[keep])
    assert not torch.equal(after, before)


# --------------------------------------------------------------------------------------
# Single CUDA device: the ops reproduce the CPU reference exactly.
# --------------------------------------------------------------------------------------


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=0.5)
def test_latent_projection_cuda_input_matches_cpu_reference_exactly():
    coords = _project_seeded(_embeddings("cuda:0"))
    assert coords.device.type == "cpu"  # the projection's compute is CPU-pinned by contract
    assert torch.equal(coords, _reference_projection())


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=0.5)
def test_latent_ablation_single_cuda_device_matches_cpu_reference_exactly():
    # The op indexes a GPU tensor with a CPU ``torch.arange`` beside a GPU ``seq_pos``: a mixed-device index that is
    # also a ROCm probe. Exact: the result is an index write of 0.0 into identical values.
    out = ablate_sae_latent(_acts("cuda:0"), hook=None, latent_idx=2, seq_pos=torch.tensor(SEQ_POS, device="cuda:0"))
    assert torch.equal(out.cpu(), _reference_ablation())


# --------------------------------------------------------------------------------------
# A real model split across two devices, against the same run on one GPU.
# --------------------------------------------------------------------------------------


@contextmanager
def _highest_float32_precision() -> Iterator[None]:
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.fixture(scope="module")
def tokens() -> dict[str, torch.Tensor]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    # A leading BOS, as the SAEs were trained with, and prompts of equal token length so no padding enters.
    ids = [[tok.bos_token_id, *tok(p).input_ids] for p in PROMPTS]
    assert len({len(i) for i in ids}) == 1, "prompts must tokenize to the same length"
    return {"input_ids": torch.tensor(ids)}


def _load_model(device_map: dict[str, str]) -> torch.nn.Module:
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(MODEL, device_map=device_map, dtype=torch.float32).eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def models() -> Iterator[dict[str, torch.nn.Module]]:
    with _highest_float32_precision():
        loaded = {
            "single": _load_model(_split_map("cuda:0", "cuda:0")),
            "split": _load_model(_split_map("cuda:0", "cuda:1")),
        }
        yield loaded
    del loaded
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def cpu_saes() -> dict[int, torch.nn.Module]:
    from sae_lens import SAE

    def load(layer: int) -> torch.nn.Module:
        sae = SAE.from_pretrained(SAE_RELEASE, f"blocks.{layer}.hook_resid_pre", device="cpu")
        return sae[0] if isinstance(sae, tuple) else sae

    return {layer: load(layer) for layer in (EARLY_LAYER, LATE_LAYER)}


def _sae_on(cpu_saes: dict[int, torch.nn.Module], layer: int, device: str) -> torch.nn.Module:
    sae = copy.deepcopy(cpu_saes[layer]).to(device)
    sae.cfg.device = device
    return sae


def _devices(module: torch.nn.Module) -> set[torch.device]:
    return {p.device for p in module.parameters()}


def _splice(model: torch.nn.Module, layer: int, sae: torch.nn.Module, record: dict) -> Callable[[], None]:
    """Replace block ``layer``'s input with the SAE reconstruction, on the devices things already live on.

    gpt2-small-res-jb was trained on TransformerLens activations, whose ``center_writing_weights`` leaves the
    residual stream mean-zero across ``d_model``; HF gpt2 does not, so the splice centres, reconstructs, and adds the
    mean back. The SAE's own metadata records ``center_writing_weights: True``; measured on CPU, L0 is about 70
    centred against about 680 uncentred.
    """

    def hook(module, args):
        x = args[0]
        record["block_input_device"] = x.device
        mean = x.mean(-1, keepdim=True)
        out = sae(x - mean) + mean
        record["splice_output_device"] = out.device
        return (out, *args[1:])

    handle = model.transformer.h[layer].register_forward_pre_hook(hook)
    return handle.remove


def _latents(model, sae, layer, tokens, record) -> torch.Tensor:
    def keep(acts, hook):
        record["latents"] = acts.detach()
        return acts

    sae.hook_sae_acts_post.add_hook(keep)
    remove = _splice(model, layer, sae, record)
    try:
        with torch.no_grad():
            model(**tokens)
    finally:
        remove()
        sae.reset_hooks()
    return record["latents"]


def _logits(model, sae, layer, tokens, record, ablate: Callable | None = None) -> torch.Tensor:
    if ablate is not None:
        sae.hook_sae_acts_post.add_hook(ablate)
    remove = _splice(model, layer, sae, record)
    try:
        with torch.no_grad():
            return model(**tokens).logits
    finally:
        remove()
        sae.reset_hooks()


SPLIT = dict(min_cuda_gpus=2, min_gpu_mem_gb=2)


@RunIf(**SPLIT)
def test_split_model_spans_exactly_both_devices(models):
    # Positive control: without it a split that silently landed on one card would read as a pass below.
    split, single = models["split"], models["single"]
    assert _devices(split) == {torch.device("cuda:0"), torch.device("cuda:1")}
    assert _devices(split.transformer.h[SPLIT_AT - 1]) == {torch.device("cuda:0")}
    assert _devices(split.transformer.h[SPLIT_AT]) == {torch.device("cuda:1")}
    assert _devices(split.transformer.h[LATE_LAYER]) == {torch.device("cuda:1")}
    assert _devices(split.transformer.h[EARLY_LAYER]) == {torch.device("cuda:0")}
    assert _devices(single) == {torch.device("cuda:0")}


@RunIf(**SPLIT)
def test_split_sae_latents_and_projection_match_single_gpu(models, cpu_saes, tokens, record_property):
    with _highest_float32_precision():
        sae = _sae_on(cpu_saes, LATE_LAYER, "cuda:1")
        split_record: dict = {}
        latents = _latents(
            models["split"], sae, LATE_LAYER, {k: v.to("cuda:0") for k, v in tokens.items()}, split_record
        )
        reference = _latents(
            models["single"],
            _sae_on(cpu_saes, LATE_LAYER, "cuda:0"),
            LATE_LAYER,
            {k: v.to("cuda:0") for k, v in tokens.items()},
            {},
        )
    # The activation reached the SAE where block 8 (on cuda:1) produced it, and the encode ran there.
    assert split_record["block_input_device"] == torch.device("cuda:1")
    assert latents.device == torch.device("cuda:1")
    assert split_record["splice_output_device"] == torch.device("cuda:1")
    assert _devices(sae) == {torch.device("cuda:1")}, "the forward moved the SAE's weights"
    assert (latents > 0).any(), "no SAE latent fired: the comparison below would be vacuous"
    # Measured differences go to the junit report, the evidence the tolerance is set against.
    record_property("latents_max_abs_diff", float((latents.cpu() - reference.cpu()).abs().max()))
    torch.testing.assert_close(latents.cpu(), reference.cpu(), atol=ATOL, rtol=RTOL)

    flat = latents.flatten(0, 1)
    coords = _project_seeded(flat)
    # The projection is placement-independent: the cuda:1 latents and the same values on CPU project identically.
    assert torch.equal(coords, _project_seeded(flat.cpu()))
    torch.testing.assert_close(coords, _project_seeded(reference.flatten(0, 1).cpu()), atol=ATOL, rtol=RTOL)


@RunIf(**SPLIT)
def test_split_ablation_hook_matches_single_gpu(models, cpu_saes, tokens, record_property):
    ids = {k: v.to("cuda:0") for k, v in tokens.items()}
    batch, last = ids["input_ids"].shape[0], ids["input_ids"].shape[1] - 1
    with _highest_float32_precision():
        reference_sae = _sae_on(cpu_saes, LATE_LAYER, "cuda:0")
        reference_latents = _latents(models["single"], reference_sae, LATE_LAYER, ids, {})
        # Ablate the latent with the largest activation that every prompt shares at the last position, so the
        # ablation changes something in each batch row (measured on CPU: latent 21794, logits move by about 1.1).
        latent_idx = int(reference_latents[:, last].min(0).values.argmax())
        assert torch.all(reference_latents[:, last, latent_idx] > 0)

        def ablation(device: str) -> Callable:
            return partial(ablate_sae_latent, latent_idx=latent_idx, seq_pos=torch.full((batch,), last, device=device))

        reference_plain = _logits(models["single"], reference_sae, LATE_LAYER, ids, {})
        reference_ablated = _logits(models["single"], reference_sae, LATE_LAYER, ids, {}, ablation("cuda:0"))
        sae, record = _sae_on(cpu_saes, LATE_LAYER, "cuda:1"), {}
        split_ablated = _logits(models["split"], sae, LATE_LAYER, ids, record, ablation("cuda:1"))

    # Positive control on the ablation itself: it moves the logits by far more than the comparison tolerance.
    record_property("ablation_effect_max_abs", float((reference_ablated - reference_plain).abs().max()))
    record_property("logits_max_abs_diff", float((split_ablated.cpu() - reference_ablated.cpu()).abs().max()))
    assert (reference_ablated - reference_plain).abs().max() > 100 * ATOL
    assert record["splice_output_device"] == torch.device("cuda:1"), "the splice left the hooked block's device"
    assert split_ablated.device == torch.device("cuda:0"), "the logits did not cross back to the unembedding's device"
    assert _devices(sae) == {torch.device("cuda:1")}
    torch.testing.assert_close(split_ablated.cpu(), reference_ablated.cpu(), atol=ATOL, rtol=RTOL)


def _sae_step(model, sae, tokens) -> tuple[torch.optim.Optimizer, list[torch.Tensor], list[torch.Tensor]]:
    before = [p.detach().clone() for p in sae.parameters()]
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    remove = _splice(model, EARLY_LAYER, sae, {})
    try:
        ids = tokens["input_ids"]
        logits = model(**tokens).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten().to(logits.device))
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad.detach().clone() for p in sae.parameters()]
        optimizer.step()
    finally:
        remove()
    return optimizer, before, grads


@RunIf(**SPLIT)
def test_split_backward_steps_sae_and_keeps_adam_state_with_its_parameters(models, cpu_saes, tokens, record_property):
    ids = {k: v.to("cuda:0") for k, v in tokens.items()}
    with _highest_float32_precision():
        sae = _sae_on(cpu_saes, EARLY_LAYER, "cuda:0")
        # The SAE sits on cuda:0 at block 1; the loss is computed from logits that passed through blocks 6 to 11 on
        # cuda:1, so every gradient reaching the SAE crossed back from cuda:1.
        optimizer, before, grads = _sae_step(models["split"], sae, ids)
        _, _, reference_grads = _sae_step(models["single"], _sae_on(cpu_saes, EARLY_LAYER, "cuda:0"), ids)

    params = list(sae.parameters())
    record_property(
        "grad_max_abs_diff", max(float((g.cpu() - r.cpu()).abs().max()) for g, r in zip(grads, reference_grads))
    )
    assert len(optimizer.state) == len(params) > 0, "Adam kept no per-parameter state: the checks below would be empty"
    for param, old, grad, reference in zip(params, before, grads, reference_grads, strict=True):
        assert param.device == torch.device("cuda:0"), "the step moved a parameter off its device"
        assert grad.abs().sum() > 0, "no gradient reached an SAE parameter across the split"
        assert not torch.equal(param.detach(), old), "the optimizer step did not change a parameter"
        torch.testing.assert_close(grad.cpu(), reference.cpu(), atol=ATOL, rtol=RTOL)
        state = optimizer.state[param]
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].device == param.device, f"Adam {key} is not on its parameter's device"
        # ``step`` is the one state entry that is not per-element: with the default ``capturable=False`` torch keeps
        # it as a CPU scalar on purpose, so it is checked as that rather than skipped.
        assert state["step"].device.type == "cpu" and state["step"].numel() == 1
