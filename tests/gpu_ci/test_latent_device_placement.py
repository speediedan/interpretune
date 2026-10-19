"""Latent projection and ablation across device placement (#795).

A single-device CPU run is the reference. Every other placement (one CUDA device, a split
across two CUDA devices, a CUDA input consumed by a CPU-pinned routine) must reproduce the
reference within tolerance, keep each tensor and parameter on the device it was declared on,
and, where an optimizer steps, actually move the parameters it owns.

This is latent (SAE) projection/ablation acceptance, distinct from J-space fine-tuning. The
CPU tests run everywhere. The CUDA tests are selected by the cuda-marked phase
(``IT_RUN_CUDA_TESTS=1``, see ``tests/conftest.py``) and, on the shared host, run only under
the GPU lease with the maintainer's authorization for this validation; collecting them is not
a hardware pass. ROCm execution of the same file stays pending until AMD hardware is
available: nothing here names an NVIDIA-only surface, so the file is the ROCm build-day
target unchanged.
"""

from __future__ import annotations

import torch

from interpretune.analysis.ops.bundled.sae.sae_ops import ablate_sae_latent
from interpretune.utils.latent_state_projection import project_embeddings
from tests.runif import RunIf


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


def _reference_ablation() -> torch.Tensor:
    acts = _acts("cpu")
    return ablate_sae_latent(acts.clone(), hook=None, latent_idx=2, seq_pos=torch.tensor([0, 2, 4]))


# --------------------------------------------------------------------------------------
# Single-device reference (CPU): the baseline every placement below is compared against.
# --------------------------------------------------------------------------------------


def test_latent_projection_cpu_reference_is_deterministic():
    first, second = _reference_projection(), _reference_projection()
    torch.testing.assert_close(first, second)
    assert first.shape == (16, 3)


def test_latent_ablation_cpu_reference_zeroes_only_the_declared_latent():
    before, after = _acts("cpu"), _reference_ablation()
    assert torch.all(after[torch.arange(3), torch.tensor([0, 2, 4]), 2] == 0)
    keep = torch.ones_like(before, dtype=torch.bool)
    keep[torch.arange(3), torch.tensor([0, 2, 4]), 2] = False
    torch.testing.assert_close(after[keep], before[keep])


def test_latent_projection_accepts_a_non_cpu_input_without_moving_it():
    # ``project_embeddings`` pins its compute to CPU by contract; a split placement (caller on
    # one device, projection on CPU) must still reproduce the reference and leave the caller's
    # tensor where it was. Exercised here with a meta-free CPU tensor; the CUDA twin below
    # repeats it with a real second device.
    values = _embeddings("cpu")
    coords = _project_seeded(values)
    assert values.device.type == "cpu"
    torch.testing.assert_close(coords, _reference_projection())


def _step_a_decoder(device: str) -> tuple[torch.nn.Linear, torch.optim.Optimizer, list[torch.Tensor]]:
    torch.manual_seed(3)
    gen = torch.Generator(device="cpu").manual_seed(3)
    decoder = torch.nn.Linear(6, 8).to(device)
    before = [p.detach().clone() for p in decoder.parameters()]
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)
    acts = torch.randn(4, 6, generator=gen).to(device)
    target = torch.randn(4, 8, generator=gen).to(device)
    loss = torch.nn.functional.mse_loss(decoder(acts), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return decoder, optimizer, before


def test_decoder_step_moves_parameters_and_keeps_them_on_their_device_cpu():
    decoder, optimizer, before = _step_a_decoder("cpu")
    for param, old in zip(decoder.parameters(), before, strict=True):
        assert param.device.type == "cpu"
        assert not torch.equal(param.detach(), old), "the optimizer step did not move a parameter"
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"


# --------------------------------------------------------------------------------------
# CUDA placements against the CPU reference. Run only in the cuda-marked phase.
# --------------------------------------------------------------------------------------


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=0.5)
def test_latent_projection_cuda_input_matches_cpu_reference():
    values = _embeddings("cuda:0")
    coords = _project_seeded(values)
    assert values.device == torch.device("cuda:0"), "projection moved the caller's tensor"
    torch.testing.assert_close(coords, _reference_projection(), atol=1e-4, rtol=1e-4)


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=0.5)
def test_latent_ablation_single_cuda_device_matches_cpu_reference():
    acts = _acts("cuda:0")
    out = ablate_sae_latent(acts.clone(), hook=None, latent_idx=2, seq_pos=torch.tensor([0, 2, 4], device="cuda:0"))
    assert out.device == torch.device("cuda:0")
    torch.testing.assert_close(out.cpu(), _reference_ablation())


@RunIf(min_cuda_gpus=2, min_gpu_mem_gb=0.5)
def test_latent_ablation_split_devices_matches_cpu_reference():
    # Split placement: activations live on cuda:1 while the reference was computed on CPU.
    # The ablated result must travel back equal, and stay on cuda:1 until the caller moves it.
    acts = _acts("cuda:1")
    out = ablate_sae_latent(acts.clone(), hook=None, latent_idx=2, seq_pos=torch.tensor([0, 2, 4], device="cuda:1"))
    assert out.device == torch.device("cuda:1")
    torch.testing.assert_close(out.cpu(), _reference_ablation())


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=0.5)
def test_decoder_step_moves_parameters_on_cuda_and_keeps_placement():
    decoder, optimizer, before = _step_a_decoder("cuda:0")
    for param, old in zip(decoder.parameters(), before, strict=True):
        assert param.device == torch.device("cuda:0")
        assert not torch.equal(param.detach(), old.to("cuda:0")), "the optimizer step did not move a parameter"
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device == torch.device("cuda:0")
