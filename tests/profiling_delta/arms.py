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
"""The sampled compositions, each an interpretune arm and a matched upstream arm.

An arm runs one repeat: build, run ``steps`` steps reporting each step's end to the recorder, release everything. The
upstream arm reproduces the interpretune arm's model, data, optimizer, precision and step order with plain PyTorch and
Hugging Face calls, so the difference between the two is the memory interpretune's orchestration adds. Configuration
is read from interpretune's realized session config (which model class, which batches, which optimizer), never its
behavior: the upstream arm's device memory comes only from upstream code.
"""

import itertools
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.optimizer import register_optimizer_step_post_hook

from tests.profiling_delta.stats import StepRecorder, accelerator_type

ARMS = ("it", "upstream")


@dataclass(frozen=True)
class Composition:
    """A sampled composition: what it measures, and the callable that runs one repeat of each arm."""

    name: str
    description: str
    it_arm: Callable[..., None]
    upstream_arm: Callable[..., None]
    needs_accelerator: bool = True

    def arm(self, which: str) -> Callable[..., None]:
        if which not in ARMS:
            raise ValueError(f"unknown arm {which!r}, expected one of {ARMS}")
        return self.it_arm if which == "it" else self.upstream_arm


def _device() -> torch.device:
    dev_type = accelerator_type()
    return torch.device(dev_type if dev_type is not None else "cpu")


########################################################################################################################
# Synthetic: the harness's own positive controls
########################################################################################################################


def _synthetic_loop(recorder: StepRecorder, steps: int, seed: int, leak_bytes: int = 0, overhead_bytes: int = 0):
    torch.manual_seed(seed)
    dev = _device()
    model = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 512, device=dev)
    retained = [torch.empty(overhead_bytes, dtype=torch.uint8, device=dev)] if overhead_bytes else []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        opt.step()
        if leak_bytes:
            retained.append(torch.empty(leak_bytes, dtype=torch.uint8, device=dev))
        recorder.step_end()


def synthetic_arm(recorder: StepRecorder, steps: int, seed: int, plant: dict[str, int] | None = None) -> None:
    """A small MLP in a plain loop.

    ``plant`` adds a known leak or a fixed overhead, which is how the harness proves
    it can fail: a check that a planted defect cannot trip is not evidence of anything.
    """
    _synthetic_loop(recorder, steps, seed, **(plant or {}))


########################################################################################################################
# Core loop, Hugging Face model (gpt2 on the RTE task, as the absolute profiling cases use)
########################################################################################################################


def _core_cfg(phase: str, steps: int, device_type: str, precision: str | None):
    from tests.parity_acceptance.cfg_aliases import bs_override
    from tests.base_defaults import BaseCfg

    @dataclass(kw_only=True)
    class _DeltaCfg(BaseCfg):
        model_src_key: str | None = "gpt2"
        force_prepare_data: bool = True

    kwargs: dict[str, Any] = dict(
        phase=phase,
        device_type=device_type,
        dm_override_cfg=bs_override,
        limit_train_batches=steps,
        limit_val_batches=0,
        limit_test_batches=steps,
        max_steps=steps,
        max_epochs=1,
    )
    if precision is not None:
        kwargs["precision"] = precision
    return _DeltaCfg(**kwargs)


def _session(cfg, tmp_path: Path):
    from tests.configuration import config_modules

    return config_modules(cfg, "profiling_delta", {}, tmp_path)


@contextmanager
def _tmpdir():
    with tempfile.TemporaryDirectory(prefix="it_profdelta_") as tmp:
        yield Path(tmp)


def core_hf_it_arm(recorder: StepRecorder, steps: int, seed: int, phase: str, precision: str | None = None) -> None:
    from lightning.fabric.utilities.seed import seed_everything

    from tests.orchestration import init_it_runner

    seed_everything(seed)
    with _tmpdir() as tmp:
        cfg = _core_cfg(phase, steps, _device().type, precision)
        session = _session(cfg, tmp)
        runner = init_it_runner(session, cfg)
        module = session.module
        step_fn_name = "training_step" if phase == "train" else "test_step"
        step_fn = getattr(module, step_fn_name)

        def recorded_step(batch, *args, **kwargs):
            recorder.note_input(batch)
            out = step_fn(batch, *args, **kwargs)
            if phase == "test":
                recorder.step_end()
            return out

        setattr(module, step_fn_name, recorded_step)
        if phase == "train":
            # Every optimizer step is a step end, whoever drives the optimizer, so the IT arm needs no knowledge of
            # the loop's internals to be measured at the same points as the upstream arm.
            handle = register_optimizer_step_post_hook(lambda *_: recorder.step_end())
            try:
                runner.train()
            finally:
                handle.remove()
        else:
            runner.test()
        delattr(module, step_fn_name)
        del runner, session, module


@dataclass
class _UpstreamSpec:
    """What the upstream arm reproduces, read from interpretune's realized (CPU) session."""

    model_class: type
    model_name: str
    config: Any
    generation_config: Any
    dtype: torch.dtype
    enable_input_require_grads: bool
    activation_checkpointing: bool
    optimizer_init: dict[str, Any]
    lr_scheduler_init: dict[str, Any] | None
    label_token_ids: torch.Tensor | None
    generate_kwargs: dict[str, Any]
    generative: bool
    batches: list[dict[str, Any]]


def _import(path: str):
    mod, _, name = path.rpartition(".")
    return getattr(__import__(mod, fromlist=[name]), name)


def _label_token_ids(it_cfg) -> torch.Tensor | None:
    # the RTE task maps its labels through an entailment mapping; other tasks use the generic classification mapping
    for attr in ("entailment_mapping_indices", "classification_mapping_indices"):
        if (ids := getattr(it_cfg, attr, None)) is not None:
            return ids
    return None


def _upstream_spec(phase: str, steps: int, precision: str | None) -> _UpstreamSpec:
    """Realize interpretune's session for this composition on the CPU and read off the configuration to reproduce.

    The session is set up on the CPU and discarded before the arm's measured run starts, so none of its state reaches
    the accelerator. Batches are materialized here (host memory) so both arms consume identical inputs.
    """
    from interpretune.base.call import it_init

    with _tmpdir() as tmp:
        cfg = _core_cfg(phase, steps, "cpu", precision)
        session = _session(cfg, tmp)
        it_init(**session)
        module, dm = session.module, session.datamodule
        loader = dm.train_dataloader() if phase == "train" else dm.test_dataloader()
        batches = [dict(b) for b in itertools.islice(loader, steps)]
        hf_cfg = module.it_cfg.hf_from_pretrained_cfg
        gen_cfg = module.it_cfg.generative_step_cfg
        spec = _UpstreamSpec(
            model_class=type(module.model),
            model_name=module.it_cfg.model_name_or_path,
            config=module.model.config,
            generation_config=getattr(module.model, "generation_config", None),
            dtype=module.model.dtype,
            enable_input_require_grads=bool(hf_cfg.enable_input_require_grads),
            activation_checkpointing=bool(hf_cfg.activation_checkpointing),
            optimizer_init=dict(module.it_cfg.optimizer_init),
            lr_scheduler_init=dict(module.it_cfg.lr_scheduler_init) if module.it_cfg.lr_scheduler_init else None,
            label_token_ids=_label_token_ids(module.it_cfg),
            generate_kwargs=dict(gen_cfg.lm_generation_cfg.generate_kwargs) if gen_cfg.enabled else {},
            generative=bool(gen_cfg.enabled),
            batches=batches,
        )
        del session, module, dm, loader
    return spec


def _upstream_model(spec: _UpstreamSpec, dev: torch.device) -> nn.Module:
    model = spec.model_class.from_pretrained(spec.model_name, config=spec.config, dtype=spec.dtype)
    if spec.generation_config is not None:
        model.generation_config = spec.generation_config
    if spec.activation_checkpointing:
        model.gradient_checkpointing_enable()
    if spec.enable_input_require_grads:
        model.enable_input_require_grads()
    return model.to(dev)


def _to(batch: dict[str, Any], dev: torch.device) -> dict[str, Any]:
    return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _autocast(dev: torch.device, dtype: torch.dtype):
    # interpretune's core loop autocasts bf16 steps; the upstream arm has to as well or the delta is the autocast
    if dtype == torch.bfloat16:
        return torch.autocast(device_type=dev.type, dtype=dtype)
    return torch.autocast(device_type=dev.type, enabled=False)


def _answer_loss(model: nn.Module, batch: dict[str, Any], label_ids: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    """The interpretune arm's training loss: cross entropy over the last position's answer-token logits.

    A function of its own so the full-vocabulary logits die on return, as they do when interpretune's
    ``training_step`` returns: held as a loop local they stay alive through backward and the optimizer step, and
    the upstream arm then reads one logits tensor heavier than the arm it is matched to.
    """
    labels = torch.take(label_ids, torch.as_tensor(batch.pop("labels"), device=label_ids.device))
    logits = model(**batch).logits
    return loss_fn(torch.squeeze(logits[:, -1, :], dim=1), labels)


def core_hf_upstream_arm(
    recorder: StepRecorder, steps: int, seed: int, phase: str, precision: str | None = None
) -> None:
    spec = _upstream_spec(phase, steps, precision)
    torch.manual_seed(seed)
    dev = _device()
    model = _upstream_model(spec, dev)
    label_ids = spec.label_token_ids.to(dev) if spec.label_token_ids is not None else None
    if phase == "train":
        model.train()
        opt = _import(spec.optimizer_init["class_path"])(model.parameters(), **spec.optimizer_init.get("init_args", {}))
        sched = None
        if spec.lr_scheduler_init:
            sched_cls = _import(spec.lr_scheduler_init["class_path"])
            sched = sched_cls(opt, **spec.lr_scheduler_init.get("init_args", {}))
        loss_fn = nn.CrossEntropyLoss()
        for batch in spec.batches:
            recorder.note_input(batch)
            batch = _to(batch, dev)
            opt.zero_grad()
            with _autocast(dev, spec.dtype):
                loss = _answer_loss(model, batch, label_ids, loss_fn)
            loss.backward()
            opt.step()
            recorder.step_end()
        del opt, sched
    else:
        model.eval()
        for batch in spec.batches:
            recorder.note_input(batch)
            with torch.inference_mode():
                batch = _to(batch, dev)
                batch.pop("labels")
                with _autocast(dev, spec.dtype):
                    if spec.generative:
                        model.generate(**batch, **spec.generate_kwargs)
                    else:
                        model(**batch)
            recorder.step_end()
    del model, spec


########################################################################################################################
# Registry
########################################################################################################################


COMPOSITIONS: dict[str, Composition] = {
    c.name: c
    for c in (
        Composition(
            name="synthetic",
            description="a small MLP in a plain loop for both arms: the harness's own control",
            it_arm=synthetic_arm,
            upstream_arm=synthetic_arm,
            needs_accelerator=False,
        ),
        Composition(
            name="core_hf_train",
            description="core session training_step on gpt2 versus a plain torch loop on the same HF model",
            it_arm=partial(core_hf_it_arm, phase="train"),
            upstream_arm=partial(core_hf_upstream_arm, phase="train"),
        ),
        Composition(
            name="core_hf_train_bf16",
            description="core/HF train at bf16 (autocast in both arms): precision is a real memory axis",
            it_arm=partial(core_hf_it_arm, phase="train", precision="bf16"),
            upstream_arm=partial(core_hf_upstream_arm, phase="train", precision="bf16"),
        ),
        Composition(
            name="core_hf_test",
            description="core session test_step (generative classification) versus HF generate on the same model",
            it_arm=partial(core_hf_it_arm, phase="test"),
            upstream_arm=partial(core_hf_upstream_arm, phase="test"),
        ),
    )
}


def with_plant(composition: Composition, plant: dict[str, int]) -> Composition:
    """The synthetic composition with a defect planted in its interpretune arm only."""
    if composition.name != "synthetic":
        raise ValueError("defects are planted only in the synthetic composition")
    return replace(composition, it_arm=partial(synthetic_arm, plant=plant))
