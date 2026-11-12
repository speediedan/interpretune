"""Schedules over quantized or packed storage (#758): a phase that would thaw a tensor torch cannot make require
gradients is refused when the schedule compiles, naming it, while a frozen packed base under trainable adapters
(QLoRA's layout) still trains."""

from __future__ import annotations

import re
from functools import partial

import pytest
import torch
from torch import nn

from tests.runif import RunIf


def _packed(module: nn.Module) -> list[str]:
    return [n for n, p in module.named_parameters() if not (p.is_floating_point() or p.is_complex())]


class TestPackedStorageControl:
    """The positive control, on CPU: a packed tensor planted beside trainable ones, through Interpretune's view."""

    @staticmethod
    def _module() -> nn.Module:
        module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        module.register_parameter("packed", nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False))
        return module

    def test_a_thawed_packed_tensor_is_refused(self):
        from finetuning_scheduler.core.compile import compile_schedule
        from finetuning_scheduler.core.exceptions import FTSConfigurationError
        from interpretune.tuning.views import identity_view

        view = identity_view(self._module())
        with pytest.raises(FTSConfigurationError, match=r"cannot require gradients.*packed \(uint8\)"):
            compile_schedule({0: {"params": ["1.weight"]}, 1: {"params": ["packed"]}}, view)

    def test_the_packed_tensor_left_frozen_compiles(self):
        from finetuning_scheduler.core.compile import compile_schedule, view_default_schedule
        from interpretune.tuning.views import identity_view

        view = identity_view(self._module())
        assert compile_schedule({0: {"params": ["1.weight", "1.bias"]}}, view).physical_for(0) == ("1.weight", "1.bias")
        generated = view_default_schedule(view)
        assert "packed" not in {n for phase in generated.values() for n in phase["params"]}
        compile_schedule(generated, view)


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=1.5)
class TestQuantizedBaseRefusal:
    """GPT-2 with a bitsandbytes 4-bit base and LoRA adapters: thawing a packed base tensor is refused at setup."""

    def test_thawing_a_packed_base_tensor_is_refused(self, get_it_session__core_gpt2_peft__initonly):
        from finetuning_scheduler import TuningScheduler
        from finetuning_scheduler.core.compile import compile_schedule
        from finetuning_scheduler.core.exceptions import FTSConfigurationError
        from interpretune.tuning.views import VIEW_ATTR

        module = get_it_session__core_gpt2_peft__initonly.it_session.module
        packed = _packed(module)
        assert packed, "the 4-bit base should hold packed (uint8) storage"
        schedule = {0: {"params": [packed[0]]}}
        # the compile step the Lightning callback and the core scheduler both run on the attached view
        with pytest.raises(FTSConfigurationError, match=r"cannot require gradients.*" + re.escape(packed[0])):
            compile_schedule(schedule, getattr(module, VIEW_ATTR))
        with pytest.raises(FTSConfigurationError, match=re.escape(packed[0])):
            TuningScheduler(module, schedule, optimizer=partial(torch.optim.SGD, lr=1e-3)).setup()


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=1.5)
class TestQuantizedBaseAdapters:
    """The same composition with only its LoRA adapters scheduled: it sets up, and a step moves only the adapters."""

    def test_lora_adapters_over_the_packed_base_train(self, get_it_session__core_gpt2_peft__initonly):
        from finetuning_scheduler import TuningScheduler

        module = get_it_session__core_gpt2_peft__initonly.it_session.module
        lora = {n for n, _ in module.named_parameters() if "lora_" in n}
        assert lora and _packed(module)
        scheduler = TuningScheduler(
            module, {0: {"params": [r".*lora_.*"]}}, optimizer=partial(torch.optim.SGD, lr=1e-2)
        )
        scheduler.setup()
        assert {n for n, p in module.named_parameters() if p.requires_grad} == lora
        device = next(p.device for n, p in module.named_parameters() if n in lora)
        ids = torch.randint(0, module.model.config.vocab_size, (1, 8), device=device)
        # bitsandbytes' 4-bit linear casts its frozen bias to the input's dtype in place on its first forward, so the
        # snapshot is taken after one: only an optimizer step should move anything after that
        with torch.no_grad():
            module.model(input_ids=ids)
        before = {n: p.detach().clone() for n, p in module.named_parameters()}
        module.model(input_ids=ids, labels=ids).loss.backward()
        scheduler.optimizer.step()
        changed = {n for n, p in module.named_parameters() if not torch.equal(p.detach(), before[n])}
        assert changed and changed <= lora
