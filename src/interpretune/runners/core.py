from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Any, TYPE_CHECKING
import logging
from functools import partialmethod
import torch

from interpretune.config import SessionRunnerCfg
from interpretune.base import _call_itmodule_hook, it_init, it_session_end, ITDataModule
from interpretune.protocol import Optimizable, CorePhases

if TYPE_CHECKING:
    from interpretune.adapters import ITModule


log = logging.getLogger(__name__)


def core_train_loop(
    module: ITModule,
    datamodule: ITDataModule,
    limit_train_batches: int,
    limit_val_batches: int,
    max_epochs: int,
    *args,
    **kwargs,
):
    assert hasattr(datamodule, "train_dataloader"), "Datamodule is expected to have a train dataloader"
    train_dataloader = datamodule.train_dataloader()  # type: ignore[attr-defined]
    val_dataloader = datamodule.val_dataloader() if hasattr(datamodule, "val_dataloader") else None  # type: ignore[attr-defined]  # duck typing
    # TODO: add optimizers property setter to corehelperattributes
    assert module.optimizers, "Module has no optimizers configured"
    optim = module.optimizers[0]
    train_ctx = {"module": module, "optimizer": optim}
    for epoch_idx in range(max_epochs):
        assert module.model is not None, "Model must be initialized before training"
        module.model.train()
        module.current_epoch = epoch_idx
        _call_itmodule_hook(module, hook_name="on_train_epoch_start", hook_msg="Running train epoch start hooks")
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= limit_train_batches >= 0:
                break
            run_step(step_fn="training_step", batch=batch, batch_idx=batch_idx, **train_ctx)
        if val_dataloader is not None:
            module.model.eval()
            for batch_idx, batch in enumerate(val_dataloader):
                with torch.inference_mode():
                    if batch_idx >= limit_val_batches >= 0:
                        break
                    run_step(step_fn="validation_step", batch=batch, batch_idx=batch_idx, **train_ctx)
        module.model.train()
        _call_itmodule_hook(module, hook_name="on_train_epoch_end", hook_msg="Running train epoch end hooks")


def core_test_loop(module: ITModule, datamodule: ITDataModule, limit_test_batches: int, *args, **kwargs):
    if not hasattr(datamodule, "test_dataloader"):
        raise AssertionError("Datamodule is expected to have a test dataloader")
    dataloader = datamodule.test_dataloader()  # type: ignore[attr-defined]  # duck typing, checked above
    test_ctx = {}
    module._it_state._current_epoch = 0
    assert module.model is not None, "Model must be initialized before testing"
    module.model.eval()
    for batch_idx, batch in enumerate(dataloader):
        with torch.inference_mode():
            if batch_idx >= limit_test_batches >= 0:
                break
            run_step(step_fn="test_step", module=module, batch=batch, batch_idx=batch_idx, **test_ctx)
    _call_itmodule_hook(module, hook_name="on_test_epoch_end", hook_msg="Running test epoch end hooks")


def run_step(step_fn, module, batch, batch_idx, optimizer: Optimizable | None = None, as_generator: bool = False):
    batch = module.batch_to_device(batch)
    step_func = getattr(module, step_fn)
    if module.global_step == 0 and step_fn in ("training_step", "test_step"):
        _call_itmodule_hook(
            module,
            hook_name="_on_test_or_train_batch_start",
            hook_msg="Running custom test or train batch start hook",
            batch=batch,
            batch_idx=batch_idx,
        )
    if step_fn == "training_step" and optimizer is not None:
        optimizer.zero_grad()
    if module.torch_dtype == torch.bfloat16:
        with torch.autocast(device_type=module.device.type, dtype=module.torch_dtype):
            output = step_func(batch, batch_idx)
    else:
        output = step_func(batch, batch_idx)
    if step_fn == "training_step" and optimizer is not None:
        _call_itmodule_hook(
            module,
            hook_name="on_train_batch_end",
            hook_msg="Running custom on_train_batch end hook",
            outputs=output,
            batch=batch,
            batch_idx=batch_idx,
        )
        output.backward()
        optimizer.step()
    module.global_step += 1

    if as_generator:
        # yield from output
        def generator():
            yield from output

        return generator()  # Return a generator object
    else:
        return output


class SessionRunner:
    """A barebones trainer that can be used to orchestrate training when no adapter is specified during ITSession
    composition."""

    def __init__(self, run_cfg: SessionRunnerCfg | dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_cfg = run_cfg if isinstance(run_cfg, SessionRunnerCfg) else SessionRunnerCfg(**run_cfg)
        # Only training and testing commands are supported in SessionRunner
        self.supported_commands = (None, "train", "test")
        self.it_init()

    @property
    def phase(self) -> CorePhases | None:
        return self._current_phase

    @phase.setter
    def phase(self, phase: CorePhases):
        self._current_phase = phase

    def it_init(self):
        # Unless overridden we dispatch the trainer-independent `it_init`
        assert self.run_cfg.it_session is not None, "Expected ITSession object"
        it_init(**self.run_cfg.it_session)

    def it_session_end(self):
        """Dispatch any phase-specific session end hooks."""
        assert self.run_cfg.it_session is not None, "Expected ITSession object"
        assert self.phase is not None, "Expected phase to be set"
        it_session_end(session_type=self.phase, **self.run_cfg.it_session)

    def _run(self, phase, loop_fn, *args: Any, **kwargs: Any) -> Any | None:
        self.phase = CorePhases[phase]
        phase_artifacts = loop_fn(**self.run_cfg.__dict__)
        self.it_session_end()
        return phase_artifacts

    test = partialmethod(_run, phase="test", loop_fn=core_test_loop)
    train = partialmethod(_run, phase="train", loop_fn=core_train_loop)
