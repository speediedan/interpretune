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
upstream arm is the wrapped package used natively, as someone without interpretune would use it (Hugging Face models in
a torch loop, a vanilla ``LightningModule``, ``TransformerBridge.boot_transformers``, an nnsight ``trace``, the
``TuningScheduler`` in a plain loop), on the same model, batches, optimizer, precision and step order. The difference
between the two is what interpretune's composition adds, its own construction and call paths included. Configuration is
read from interpretune's realized session config (which model, which batches, which optimizer), never its behavior: the
upstream arm's device memory comes only from upstream code.
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
    window: int | None = None  # measured steps, when the composition needs more than the default (phases)
    warmup: int | None = None  # unmeasured steps, when allocator settling takes longer than the default

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


def _core_cfg(
    phase: str,
    steps: int,
    device_type: str,
    precision: str | None,
    adapter_ctx: tuple | None = None,
    cfg_extras: dict[str, Any] | None = None,
):
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
    if adapter_ctx is not None:
        kwargs["adapter_ctx"] = adapter_ctx
        if any(getattr(a, "name", a) == "nnsight" for a in adapter_ctx):
            from interpretune.adapters.nnsight.config import NNsightConfig

            kwargs["nnsight_cfg"] = NNsightConfig(model_name="openai-community/gpt2")  # as the absolute cases use
    if cfg_extras:
        # a callable builds the extras in the arm's subprocess, so importing this module never imports an adapter
        kwargs.update(cfg_extras() if callable(cfg_extras) else cfg_extras)
    return _DeltaCfg(**kwargs)


def _ctx(*names: str) -> tuple:
    from interpretune.protocol import Adapter

    return tuple(Adapter[name] for name in names)


def _session(cfg, tmp_path: Path):
    from tests.configuration import config_modules

    return config_modules(cfg, "profiling_delta", {}, tmp_path)


@contextmanager
def _tmpdir():
    with tempfile.TemporaryDirectory(prefix="it_profdelta_") as tmp:
        yield Path(tmp)


def _recording_hook(recorder: StepRecorder, root: nn.Module):
    def on_step(optimizer, *_):
        recorder.step_end()
        recorder.note_structure(structure(root, optimizer))

    return on_step


def core_hf_it_arm(
    recorder: StepRecorder,
    steps: int,
    seed: int,
    phase: str,
    precision: str | None = None,
    adapter_ctx: tuple[str, ...] | None = None,
    cfg_extras: dict[str, Any] | None = None,
) -> None:
    from lightning.fabric.utilities.seed import seed_everything

    from tests.orchestration import init_it_runner

    seed_everything(seed)
    with _tmpdir() as tmp:
        ctx = _ctx(*adapter_ctx) if adapter_ctx else None
        cfg = _core_cfg(phase, steps, _device().type, precision, ctx, cfg_extras)
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
            handle = register_optimizer_step_post_hook(_recording_hook(recorder, module))
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
    pl_lrs_cfg: dict[str, Any]
    label_token_ids: torch.Tensor | None
    generate_kwargs: dict[str, Any]
    generative: bool
    batches: list[dict[str, Any]]
    tokenizer: Any = None
    tl_bridge: bool = False
    nnsight_kwargs: dict[str, Any] | None = None
    sae_refs: tuple[tuple[str, str], ...] = ()  # (release, sae_id) per SAE spliced in at init


def _import(path: str):
    mod, _, name = path.rpartition(".")
    return getattr(__import__(mod, fromlist=[name]), name)


def _label_token_ids(it_cfg) -> torch.Tensor | None:
    # the RTE task maps its labels through an entailment mapping; other tasks use the generic classification mapping
    for attr in ("entailment_mapping_indices", "classification_mapping_indices"):
        if (ids := getattr(it_cfg, attr, None)) is not None:
            return ids
    return None


def _sae_refs(it_cfg) -> tuple[tuple[str, str], ...]:
    """The pretrained SAE references a sae_lens session splices in, so the upstream arm loads the same ones
    natively."""
    from interpretune.adapters.sae_lens.config import SAELensFromPretrainedConfig

    refs = []
    for sae_cfg in getattr(it_cfg, "sae_cfgs", None) or ():
        if not isinstance(sae_cfg, SAELensFromPretrainedConfig):
            raise ValueError("the sae_lens delta pairs reproduce pretrained SAEs only")
        refs.append((sae_cfg.release, sae_cfg.sae_id))
    return tuple(refs)


def _nnsight_kwargs(it_cfg) -> dict[str, Any] | None:
    ns_cfg = getattr(it_cfg, "nnsight_cfg", None)
    if ns_cfg is None:
        return None
    return {"model_name": ns_cfg.model_name or it_cfg.model_name_or_path, **ns_cfg.get_nnsight_kwargs()}


def _upstream_spec(
    phase: str,
    steps: int,
    precision: str | None,
    adapter_ctx: tuple[str, ...] | None = None,
    cfg_extras: dict[str, Any] | None = None,
) -> _UpstreamSpec:
    """Realize interpretune's session for this composition on the CPU and read off the configuration to reproduce.

    The session is set up on the CPU and discarded before the arm's measured run starts, so none of its state reaches
    the accelerator. Batches are materialized here (host memory) so both arms consume identical inputs.
    """
    from interpretune.base.call import it_init

    with _tmpdir() as tmp:
        cfg = _core_cfg(phase, steps, "cpu", precision, _ctx(*adapter_ctx) if adapter_ctx else None, cfg_extras)
        session = _session(cfg, tmp)
        it_init(**session)
        module, dm = session.module, session.datamodule
        loader = dm.train_dataloader() if phase == "train" else dm.test_dataloader()
        batches = [dict(b) for b in itertools.islice(loader, steps)]
        hf_cfg = module.it_cfg.hf_from_pretrained_cfg
        gen_cfg = module.it_cfg.generative_step_cfg
        # a TransformerLens session holds a bridge over the HF model; the bridge keeps the HF config as `config`
        # (a sae_lens session's wrapper is a bridge subclass, so it counts as one here too)
        tl_bridge = bool({"transformer_lens", "sae_lens"} & set(adapter_ctx or ()))
        spec = _UpstreamSpec(
            model_class=_import(hf_cfg.model_head) if tl_bridge else type(module.model),
            model_name=module.it_cfg.model_name_or_path,
            config=module.model.config,
            generation_config=getattr(module.model, "generation_config", None),
            dtype=next(module.model.parameters()).dtype,
            # the TransformerLens module's `set_input_require_grads` is a no-op, so its arm never enables them
            enable_input_require_grads=bool(getattr(hf_cfg, "enable_input_require_grads", False)) and not tl_bridge,
            activation_checkpointing=bool(getattr(hf_cfg, "activation_checkpointing", False)),
            optimizer_init=dict(module.it_cfg.optimizer_init),
            lr_scheduler_init=dict(module.it_cfg.lr_scheduler_init) if module.it_cfg.lr_scheduler_init else None,
            pl_lrs_cfg=dict(getattr(module.it_cfg, "pl_lrs_cfg", None) or {}),
            label_token_ids=_label_token_ids(module.it_cfg),
            generate_kwargs=dict(gen_cfg.lm_generation_cfg.generate_kwargs) if gen_cfg.enabled else {},
            generative=bool(gen_cfg.enabled),
            batches=batches,
            tokenizer=dm.tokenizer,
            tl_bridge=tl_bridge,
            nnsight_kwargs=_nnsight_kwargs(module.it_cfg),
            sae_refs=_sae_refs(module.it_cfg),
        )
        del session, module, dm, loader
    return spec


class _SaeRoot(nn.Module):
    """A native SAETransformerBridge with its spliced SAEs registered, so one optimizer sees what training them
    takes.

    ``SAETransformerBridge.add_sae`` keeps a spliced SAE outside the bridge's module tree, so its parameters are in no
    ``parameters()`` and nothing native steps them; anyone training SAEs without interpretune has to register them
    somewhere, as interpretune's ``attached_saes`` does. The same SAE objects are held here, so the splice and the
    optimizer act on one copy.
    """

    def __init__(self, bridge: nn.Module, saes: list[nn.Module]) -> None:
        super().__init__()
        self.bridge = bridge
        self.saes = nn.ModuleList(saes)

    def forward(self, *args, **kwargs):
        return self.bridge(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.bridge.generate(*args, **kwargs)


def _upstream_model(spec: _UpstreamSpec, dev: torch.device) -> nn.Module:
    if spec.nnsight_kwargs is not None:
        return _nnsight_model(spec, dev)
    if spec.sae_refs:
        from sae_lens import SAE, SAETransformerBridge

        # SAELens's own entry points: its bridge factory and `SAE.from_pretrained`, spliced with `add_sae`
        bridge = SAETransformerBridge.boot_transformers(spec.model_name, device=dev, dtype=spec.dtype)
        saes = []
        for release, sae_id in spec.sae_refs:
            sae = SAE.from_pretrained(release, sae_id, device=str(dev))
            bridge.add_sae(sae)
            saes.append(sae)
        return _SaeRoot(bridge, saes)
    if spec.tl_bridge:
        from transformer_lens.model_bridge import TransformerBridge

        # TransformerLens's own entry point, rather than interpretune's factory path, so that path is in the delta
        return TransformerBridge.boot_transformers(spec.model_name, device=dev, dtype=spec.dtype)
    model = spec.model_class.from_pretrained(spec.model_name, config=spec.config, dtype=spec.dtype)
    if spec.generation_config is not None:
        model.generation_config = spec.generation_config
    if spec.activation_checkpointing:
        model.gradient_checkpointing_enable()
    if spec.enable_input_require_grads:
        model.enable_input_require_grads()
    return model.to(dev)


def _nnsight_model(spec: _UpstreamSpec, dev: torch.device):
    """A bare nnsight ``LanguageModel`` of the same checkpoint, loaded the native way: nnsight's defaults,
    dispatched to the measured device."""
    from nnsight import LanguageModel

    return LanguageModel(spec.nnsight_kwargs["model_name"], device_map=dev.type, dispatch=True)


def _nnsight_answer_loss(lm, batch: dict[str, Any], label_ids: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    """The same answer loss through nnsight's native idiom: trace the batch and save the output logits."""
    labels = torch.take(label_ids, torch.as_tensor(batch.pop("labels"), device=label_ids.device))
    with lm.trace(batch):
        logits = lm.output.logits.save()
    return loss_fn(torch.squeeze(logits[:, -1, :], dim=1), labels)


def _params_root(model) -> nn.Module:
    """The torch module holding the parameters: nnsight wrappers keep it as ``_model``."""
    inner = getattr(model, "_model", None)
    return inner if isinstance(inner, nn.Module) else model


def _accepted(fn: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The kwargs ``fn`` names: a native bridge ``generate`` takes no Hugging Face-only flags."""
    import inspect

    names = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in names}


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
    out = model(**batch)
    logits = out if isinstance(out, torch.Tensor) else out.logits  # a TransformerLens bridge returns the tensor
    return loss_fn(torch.squeeze(logits[:, -1, :], dim=1), labels)


def core_hf_upstream_arm(
    recorder: StepRecorder,
    steps: int,
    seed: int,
    phase: str,
    precision: str | None = None,
    adapter_ctx: tuple[str, ...] | None = None,
    cfg_extras: dict[str, Any] | None = None,
) -> None:
    spec = _upstream_spec(phase, steps, precision, adapter_ctx, cfg_extras)
    torch.manual_seed(seed)
    dev = _device()
    model = _upstream_model(spec, dev)
    label_ids = spec.label_token_ids.to(dev) if spec.label_token_ids is not None else None
    root = _params_root(model)
    if phase == "train":
        root.train()
        opt = _import(spec.optimizer_init["class_path"])(root.parameters(), **spec.optimizer_init.get("init_args", {}))
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
                step_loss = _nnsight_answer_loss if spec.nnsight_kwargs is not None else _answer_loss
                loss = step_loss(model, batch, label_ids, loss_fn)
            loss.backward()
            opt.step()
            recorder.step_end()
            recorder.note_structure(structure(root, opt))
        del opt, sched
    else:
        root.eval()
        for batch in spec.batches:
            recorder.note_input(batch)
            with torch.inference_mode():
                batch = _to(batch, dev)
                batch.pop("labels")
                with _autocast(dev, spec.dtype):
                    if spec.generative and spec.tl_bridge:
                        # filter against the bridge's own signature: `_SaeRoot.generate` only forwards `**kwargs`
                        gen_kwargs = _accepted(getattr(model, "bridge", model).generate, spec.generate_kwargs)
                        model.generate(batch.pop("input"), **gen_kwargs, **batch)
                    elif spec.generative:
                        model.generate(**batch, **spec.generate_kwargs)
                    else:
                        model(**batch)
            recorder.step_end()
    del model, root, spec


########################################################################################################################
# Lightning: interpretune's Lightning composition versus a vanilla LightningModule, under the same Trainer
########################################################################################################################


def _trainer(steps: int, epochs: int = 1):
    from lightning.pytorch import Trainer

    accelerator = "cpu" if _device().type == "cpu" else "gpu"
    return Trainer(
        default_root_dir=tempfile.mkdtemp(prefix="it_profdelta_l_"),
        devices=1,
        accelerator=accelerator,
        max_epochs=epochs,
        precision="32-true",
        num_sanity_val_steps=0,
        limit_train_batches=-(-steps // epochs),
        limit_val_batches=0,
        limit_test_batches=steps,
        max_steps=steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def lightning_it_arm(
    recorder: StepRecorder,
    steps: int,
    seed: int,
    phase: str,
    epochs: int = 1,
    adapter_ctx: tuple[str, ...] = ("lightning",),
) -> None:
    from lightning.fabric.utilities.seed import seed_everything

    seed_everything(seed)
    with _tmpdir() as tmp:
        cfg = _core_cfg(phase, steps, _device().type, None, adapter_ctx=_ctx(*adapter_ctx))
        session = _session(cfg, tmp)
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
        trainer = _trainer(steps, epochs)
        handle = register_optimizer_step_post_hook(_recording_hook(recorder, module))
        try:
            (trainer.fit if phase == "train" else trainer.test)(**session)
        finally:
            handle.remove()
            delattr(module, step_fn_name)
        del trainer, session


def _vanilla_lightning_module(spec: _UpstreamSpec, recorder: StepRecorder):
    from lightning.pytorch import LightningModule

    class VanillaHF(LightningModule):
        """The same HF model, loss and generation as the interpretune arm, in a plain LightningModule."""

        def __init__(self) -> None:
            super().__init__()
            # the Trainer moves the model, as it moves interpretune's; an nnsight arm registers the LanguageModel's
            # torch module so the Trainer moves and optimizes it, and steps through the LanguageModel's own trace
            self.lm = _upstream_model(spec, torch.device("cpu")) if spec.nnsight_kwargs is not None else None
            self.model = _params_root(self.lm) if self.lm is not None else _upstream_model(spec, torch.device("cpu"))
            self.loss_fn = nn.CrossEntropyLoss()
            self.register_buffer("label_ids", spec.label_token_ids.clone(), persistent=False)

        def training_step(self, batch, batch_idx):
            recorder.note_input(batch)
            if self.lm is not None:
                return _nnsight_answer_loss(self.lm, batch, self.label_ids, self.loss_fn)
            labels = torch.take(self.label_ids, torch.as_tensor(batch.pop("labels"), device=self.device))
            logits = self.model(**batch).logits
            return self.loss_fn(torch.squeeze(logits[:, -1, :], dim=1), labels)

        def test_step(self, batch, batch_idx):
            recorder.note_input(batch)
            batch.pop("labels")
            if spec.generative:
                self.model.generate(**batch, **spec.generate_kwargs)
            else:
                self.model(**batch)
            recorder.step_end()

        def configure_optimizers(self):
            init = spec.optimizer_init
            opt = _import(init["class_path"])(self.model.parameters(), **init.get("init_args", {}))
            if not spec.lr_scheduler_init:
                return opt
            sched_init = spec.lr_scheduler_init
            sched = _import(sched_init["class_path"])(opt, **sched_init.get("init_args", {}))
            return [opt], [{"scheduler": sched, **spec.pl_lrs_cfg}]

    return VanillaHF()


def lightning_upstream_arm(
    recorder: StepRecorder,
    steps: int,
    seed: int,
    phase: str,
    epochs: int = 1,
    adapter_ctx: tuple[str, ...] | None = None,
) -> None:
    from lightning.fabric.utilities.seed import seed_everything
    from torch.utils.data import DataLoader

    # the spec is read from the core-loop twin of the composition: same model, data and optimizer configuration
    core_ctx = tuple("core" if a == "lightning" else a for a in adapter_ctx) if adapter_ctx else None
    spec = _upstream_spec(phase, steps, None, core_ctx)
    seed_everything(seed)
    module = _vanilla_lightning_module(spec, recorder)
    # the batches are already collated: batch_size=None hands each one to the step unchanged
    loader = DataLoader(spec.batches, batch_size=None)
    trainer = _trainer(steps, epochs)
    handle = register_optimizer_step_post_hook(_recording_hook(recorder, module))
    try:
        if phase == "train":
            trainer.fit(module, train_dataloaders=loader)
        else:
            trainer.test(module, dataloaders=loader)
    finally:
        handle.remove()
    del trainer, module, spec


########################################################################################################################
# Core loop, Hugging Face model, phased by finetuning-scheduler's TuningScheduler
########################################################################################################################

# Three phases thawing from the output end of gpt2, one epoch each: two transitions, each adding parameter groups and,
# at the first step after the thaw, their AdamW state. Patterns are anchored at the interpretune module's own names
# (`model.<hf name>`), which the upstream arm reproduces by holding the HF model under a `model` attribute.
FTS_SCHEDULE = {
    0: {"params": [r"model\.transformer\.ln_f\..*"], "max_transition_epoch": 1},
    1: {"params": [r"model\.transformer\.h\.11\..*"], "max_transition_epoch": 2},
    2: {"params": [r"model\.transformer\.h\.10\..*"]},
}
FTS_PHASES = len(FTS_SCHEDULE)
FTS_STEPS_PER_PHASE = 6


def fts_available() -> bool:
    try:
        import finetuning_scheduler.core.tuning_scheduler  # noqa: F401
        import interpretune.tuning.core_loop  # noqa: F401
    except ImportError:
        return False
    return True


def structure(root: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Exact, noise-free invariants of a phased run, taken at a step end.

    A frozen parameter holding a gradient, optimizer state for a frozen parameter, or a frozen parameter in a param
    group is a leak a byte delta would first have to clear the noise floor to see; counted here it needs no threshold.
    """
    in_groups = {id(p) for g in optimizer.param_groups for p in g["params"]}
    params = list(root.parameters())
    state = [(p, st) for p, st in optimizer.state.items()]
    return {
        "thawed": sum(p.requires_grad for p in params),
        "frozen_with_grad": sum((not p.requires_grad) and p.grad is not None for p in params),
        "frozen_in_groups": sum((not p.requires_grad) and id(p) in in_groups for p in params),
        "state_params": len(state),
        "state_frozen": sum(not p.requires_grad for p, _ in state),
        "state_bytes": sum(
            t.numel() * t.element_size() for _, st in state for t in st.values() if isinstance(t, torch.Tensor)
        ),
    }


def _fts_steps(steps: int) -> int:
    if steps != FTS_PHASES * FTS_STEPS_PER_PHASE:
        raise ValueError(f"the FTS compositions run {FTS_PHASES} x {FTS_STEPS_PER_PHASE} steps, not {steps}")
    return FTS_STEPS_PER_PHASE


def core_hf_fts_it_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from lightning.fabric.utilities.seed import seed_everything

    from interpretune.config import SessionRunnerCfg
    from interpretune.runners.core import SessionRunner
    from interpretune.tuning.core_loop import FTSCoreCallback

    per_phase = _fts_steps(steps)
    seed_everything(seed)
    with _tmpdir() as tmp:
        cfg = _core_cfg("train", per_phase, _device().type, None)
        session = _session(cfg, tmp)
        fts = FTSCoreCallback(FTS_SCHEDULE)
        runner = SessionRunner(
            SessionRunnerCfg(
                it_session=session,
                max_epochs=FTS_PHASES,
                limit_train_batches=per_phase,
                limit_val_batches=0,
                callbacks=[fts],
            )
        )
        module = session.module
        training_step = module.training_step

        def recorded_step(batch, *args, **kwargs):
            recorder.note_input(batch)
            recorder.set_phase(fts.tuning_scheduler.curr_depth)
            return training_step(batch, *args, **kwargs)

        def on_step(optimizer, *_):
            recorder.step_end()
            recorder.note_structure(structure(module, optimizer))

        module.training_step = recorded_step
        handle = register_optimizer_step_post_hook(on_step)
        try:
            runner.train()
        finally:
            handle.remove()
            delattr(module, "training_step")
        del runner, session


class _Rooted(nn.Module):
    """Holds the HF model under ``model`` so the upstream tuning scheduler sees interpretune's parameter names."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


def core_hf_fts_upstream_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from finetuning_scheduler.core.tuning_scheduler import TuningScheduler

    per_phase = _fts_steps(steps)
    spec = _upstream_spec("train", per_phase, None)
    torch.manual_seed(seed)
    dev = _device()
    root = _Rooted(_upstream_model(spec, dev))
    root.train()
    sched_init = spec.lr_scheduler_init
    ctrl = TuningScheduler(
        root,
        FTS_SCHEDULE,
        optimizer=partial(_import(spec.optimizer_init["class_path"]), **spec.optimizer_init.get("init_args", {})),
        lr_scheduler=partial(_import(sched_init["class_path"]), **sched_init.get("init_args", {}))
        if sched_init
        else None,
    )
    ctrl.setup()
    label_ids = spec.label_token_ids.to(dev)
    loss_fn = nn.CrossEntropyLoss()
    for _ in ctrl.epochs(FTS_PHASES):
        for batch in spec.batches:
            # same order as the core loop: gradients are released before the step function (where the IT arm's
            # phase boundary lands), so both arms reset their phase peaks at the same allocator state
            ctrl.optimizer.zero_grad()
            recorder.note_input(batch)
            recorder.set_phase(ctrl.curr_depth)
            batch = _to(batch, dev)
            _answer_loss(root.model, batch, label_ids, loss_fn).backward()
            ctrl.optimizer.step()
            recorder.step_end()
            recorder.note_structure(structure(root, ctrl.optimizer))
            ctrl.step_lr_schedulers("step")
            ctrl.step_end()
        ctrl.step_lr_schedulers("epoch", {})
    ctrl.train_end()
    del ctrl, root, spec


########################################################################################################################
# Circuit-tracer attribution: interpretune's circuit_tracer composition versus a bare ReplacementModel
########################################################################################################################

# The smallest model/transcoder pair the circuit-tracer notebooks run, and the one the test registry's
# circuit-tracer composition uses (google/gemma-3-1b-pt with the width_16k gemma-scope transcoders). An
# attribution is a step. Replacement-model construction dominates both arms' retained bytes, which is the
# point: the native arm subtracts it out, leaving interpretune's attribution orchestration.
CT_MODEL_NAME = "google/gemma-3-1b-pt"
CT_TRANSCODER_SET = "mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine"
CT_PROMPT = "Fact: the capital of the state containing Dallas is"
CT_ATTRIBUTION_KWARGS: dict[str, Any] = {
    "attribution_targets": ["▁Dallas", "▁Austin"],
    "max_n_logits": 10,
    "desired_logit_prob": 0.95,
    "batch_size": 256,
    "max_feature_nodes": 8192,
    "offload": "cpu",
    "verbose": False,
}


def _ct_note_input(recorder: StepRecorder, tokenizer) -> None:
    recorder.note_input({"input_ids": tokenizer(CT_PROMPT, return_tensors="pt")["input_ids"]})


def circuit_tracer_it_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from lightning.fabric.utilities.seed import seed_everything
    from transformers import AutoTokenizer

    from tests.conftest import FixtPhase, session_fixture_hook_exec
    from tests.configuration import config_modules
    from tests.core.cfg_aliases import CircuitTracerNNsightGemma3

    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(CT_MODEL_NAME)
    with _tmpdir() as tmp:
        cfg = CircuitTracerNNsightGemma3(device_type=_device().type)
        session = config_modules(cfg, "profiling_delta_ct", {}, tmp)
        session_fixture_hook_exec(session, FixtPhase.setup)
        module = session.module
        for _ in range(steps):
            _ct_note_input(recorder, tokenizer)
            module.generate_attribution_graph(CT_PROMPT, verbose=False)
            recorder.step_end()
            # the module retains every graph it generates; the native arm discards each, so match that or the
            # retained graphs read as a leak in the interpretune arm alone
            module.attribution_graphs.clear()
        del session, module


def circuit_tracer_upstream_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from circuit_tracer import ReplacementModel, attribute
    from transformers import AutoTokenizer

    from tests.core.cfg_aliases import CircuitTracerNNsightGemma3

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(CT_MODEL_NAME)
    # circuit-tracer's own factory with the kwargs the interpretune composition passes it, read from its config:
    # the dtype (bfloat16, not float32) and the lazy-encoder setting (on when offloading) move gigabytes
    cfg = CircuitTracerNNsightGemma3().circuit_tracer_cfg
    assert cfg is not None and cfg.transcoder_set == CT_TRANSCODER_SET
    model = ReplacementModel.from_pretrained(
        CT_MODEL_NAME,
        CT_TRANSCODER_SET,
        backend=cfg.backend,
        dtype=cfg.dtype,
        lazy_encoder=cfg.lazy_encoder if cfg.lazy_encoder is not None else (cfg.offload == "cpu"),
        lazy_decoder=cfg.lazy_decoder,
        device_map=0,
        device=str(_device()),
    )
    for _ in range(steps):
        _ct_note_input(recorder, tokenizer)
        graph = attribute(prompt=CT_PROMPT, model=model, **CT_ATTRIBUTION_KWARGS)
        recorder.step_end()
        del graph
    del model


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

COMPOSITIONS["tl_train"] = Composition(
    name="tl_train",
    description="core session with the TransformerLens adapter (a TransformerBridge over gpt2) versus a native "
    "TransformerBridge.boot_transformers bridge in a plain torch loop",
    it_arm=partial(core_hf_it_arm, phase="train", adapter_ctx=("core", "transformer_lens")),
    upstream_arm=partial(core_hf_upstream_arm, phase="train", adapter_ctx=("core", "transformer_lens")),
)
COMPOSITIONS["tl_test"] = Composition(
    name="tl_test",
    description="TransformerLens test (generative classification through the bridge) versus a native bridge's generate",
    it_arm=partial(core_hf_it_arm, phase="test", adapter_ctx=("core", "transformer_lens")),
    upstream_arm=partial(core_hf_upstream_arm, phase="test", adapter_ctx=("core", "transformer_lens")),
)
COMPOSITIONS["nnsight_train"] = Composition(
    name="nnsight_train",
    description="core session with the NNsight adapter (a local LanguageModel over gpt2) versus a native LanguageModel "
    "trace of the same checkpoint in a plain torch loop",
    it_arm=partial(core_hf_it_arm, phase="train", adapter_ctx=("core", "nnsight")),
    upstream_arm=partial(core_hf_upstream_arm, phase="train", adapter_ctx=("core", "nnsight")),
)


def _sae_cfg_extras() -> dict[str, Any]:
    """One gpt2-small-res-jb SAE spliced in at init, as the sae_lens pairs measure (interpretune#825)."""
    from interpretune.adapters.sae_lens.config import SAELensFromPretrainedConfig

    return {
        "add_saes_on_init": True,
        "sae_cfgs": [SAELensFromPretrainedConfig(release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre")],
    }


for _phase in ("train", "test"):
    COMPOSITIONS[f"sae_lens_{_phase}"] = Composition(
        name=f"sae_lens_{_phase}",
        description="core session with the sae_lens adapter (SAEs spliced at init on gpt2) versus SAELens's own "
        "SAETransformerBridge with the same SAEs attached through add_sae",
        it_arm=partial(core_hf_it_arm, phase=_phase, adapter_ctx=("core", "sae_lens"), cfg_extras=_sae_cfg_extras),
        upstream_arm=partial(
            core_hf_upstream_arm, phase=_phase, adapter_ctx=("core", "sae_lens"), cfg_extras=_sae_cfg_extras
        ),
    )
COMPOSITIONS["nnsight_test"] = Composition(
    name="nnsight_test",
    description="core session with the NNsight adapter, test_step (generative classification through the Envoy) versus "
    "the same call on a native LanguageModel",
    it_arm=partial(core_hf_it_arm, phase="test", adapter_ctx=("core", "nnsight")),
    upstream_arm=partial(core_hf_upstream_arm, phase="test", adapter_ctx=("core", "nnsight")),
)
COMPOSITIONS["circuit_tracer_test"] = Composition(
    name="circuit_tracer_test",
    description="core session with the circuit_tracer adapter (NNsight backend, gemma-3-1b-pt) attributing one "
    "prompt per step versus a bare circuit-tracer ReplacementModel attributing the same prompt",
    it_arm=circuit_tracer_it_arm,
    upstream_arm=circuit_tracer_upstream_arm,
    # an attribution is far heavier than a training step: one warmup attribution and a 3-attribution window
    warmup=1,
    window=3,
)
COMPOSITIONS["lightning_train"] = Composition(
    name="lightning_train",
    description="interpretune's Lightning composition versus a vanilla LightningModule on the same HF model, under "
    "the same Trainer configuration (Lightning's own cost cancels)",
    it_arm=partial(lightning_it_arm, phase="train", epochs=2),
    upstream_arm=partial(lightning_upstream_arm, phase="train", epochs=2),
    # both arms settle over Lightning's first training steps (measured on gpt2: the vanilla arm is flat only from step
    # 6 on CUDA and step 8 on ROCm), so they warm up longer, over two epochs of the task's 10 examples at batch size 1
    warmup=8,
    window=12,
)
COMPOSITIONS["nnsight_lightning_train"] = Composition(
    name="nnsight_lightning_train",
    description="interpretune's Lightning + NNsight composition versus a vanilla LightningModule stepping a native "
    "LanguageModel trace, under the same Trainer (the NNsight + Lightning footprint once skipped as ~2x)",
    it_arm=partial(lightning_it_arm, phase="train", epochs=2, adapter_ctx=("lightning", "nnsight")),
    upstream_arm=partial(lightning_upstream_arm, phase="train", epochs=2, adapter_ctx=("lightning", "nnsight")),
    warmup=8,
    window=12,
)
COMPOSITIONS["lightning_test"] = Composition(
    name="lightning_test",
    description="Lightning test (generative classification) versus a vanilla LightningModule's generate",
    it_arm=partial(lightning_it_arm, phase="test"),
    upstream_arm=partial(lightning_upstream_arm, phase="test"),
)

if fts_available():
    COMPOSITIONS["core_hf_fts_train"] = Composition(
        name="core_hf_fts_train",
        description="core loop + FTSCoreCallback over a 3-phase schedule versus a plain loop driving the same "
        "TuningScheduler (its own cost cancels; per-phase slopes, transition steps and structure)",
        it_arm=core_hf_fts_it_arm,
        upstream_arm=core_hf_fts_upstream_arm,
        window=FTS_PHASES * FTS_STEPS_PER_PHASE - 2,
    )


def with_plant(composition: Composition, plant: dict[str, int]) -> Composition:
    """The synthetic composition with a defect planted in its interpretune arm only."""
    if composition.name != "synthetic":
        raise ValueError("defects are planted only in the synthetic composition")
    return replace(composition, it_arm=partial(synthetic_arm, plant=plant))
