from typing import Any, Optional, Dict
import logging
from functools import partialmethod
from dataclasses import dataclass

import torch

from interpretune.base.config.shared import CorePhase
from interpretune.base.datamodules import ITDataModule
from interpretune.adapters.core import ITModule
from interpretune.base.contract.protocol import ITModuleProtocol, ITDataModuleProtocol
from interpretune.base.contract.session import ITSession
from interpretune.base.call import _call_itmodule_hook, it_init, it_session_end
from interpretune.utils.types import Optimizable
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


def core_train_loop(module: ITModule, datamodule: ITDataModule, limit_train_batches: int, limit_val_batches: int,
    max_epochs: int, *args, **kwargs):
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # TODO: add optimizers property setter to corehelperattributes
    optim = module.optimizers[0]
    train_ctx = {"module": module, "optimizer": optim}
    for epoch_idx in range(max_epochs):
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
    dataloader = datamodule.test_dataloader()
    test_ctx = {"module": module}
    module._it_state._current_epoch = 0
    module.model.eval()
    for batch_idx, batch in enumerate(dataloader):
        with torch.inference_mode():
            if batch_idx >= limit_test_batches >= 0:
                break
            run_step(step_fn="test_step", batch=batch, batch_idx=batch_idx, **test_ctx)
    _call_itmodule_hook(module, hook_name="on_test_epoch_end", hook_msg="Running test epoch end hooks")

def run_step(step_fn, module, batch, batch_idx, optimizer: Optional[Optimizable] = None):
    batch = module.batch_to_device(batch)
    step_func = getattr(module, step_fn)
    if module.global_step == 0 and step_fn != "validation_step":
        _call_itmodule_hook(module, hook_name="_on_test_or_train_batch_start",
                            hook_msg="Running custom test or train batch start hook", batch=batch, batch_idx=batch_idx)
    if step_fn == "training_step":
        optimizer.zero_grad()
    if module.torch_dtype == torch.bfloat16:
        with torch.autocast(device_type=module.device.type, dtype=module.torch_dtype):
            loss = step_func(batch, batch_idx)
    else:
        loss = step_func(batch, batch_idx)
    if step_fn == "training_step":
        _call_itmodule_hook(module, hook_name="on_train_batch_end",
                            hook_msg="Running custom on_train_batch end hook", outputs=loss,
                            batch=batch, batch_idx=batch_idx)
        loss.backward()
        optimizer.step()
    module.global_step += 1


@dataclass(kw_only=True)
class BasicTrainerCfg:
    it_session: Optional[ITSession] = None
    module: Optional[ITModuleProtocol] = None
    datamodule: Optional[ITDataModuleProtocol] = None
    limit_train_batches: int = -1
    limit_val_batches: int = -1
    limit_test_batches: int = -1
    max_steps: int = -1
    max_epochs: int = -1

    def __post_init__(self):
        if self.it_session is not None:
            self._session_validation()
        else:
            if not all((self.module, self.datamodule)):
                raise MisconfigurationException("If not providing `it_session`, must provide both a `datamodule` and"
                                                " `module`")

    def _session_validation(self):
        if any((self.module, self.datamodule)):
            rank_zero_warn("`module`/`datamodule` should only be specified if not providing `it_session`. Attempting to"
                           " use the `module`/`datamodule` handles from `it_session`.")
        self.module = self.it_session.module
        self.datamodule = self.it_session.datamodule


class BasicTrainer:
    """A barebones trainer that can be used to orchestrate training when no adapter is specified during ITSession
    composition."""
    def __init__(self, trainer_cfg: BasicTrainerCfg | Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        super().__init__(*args, **kwargs)
        self.trainer_cfg = trainer_cfg if isinstance(trainer_cfg, BasicTrainerCfg) else BasicTrainerCfg(**trainer_cfg)
        self._current_phase = None
        self.supported_commands = (None, "train", "test")
        self.it_init()

    @property
    def phase(self) -> Optional[CorePhase]:
        return self._current_phase

    @phase.setter
    def phase(self, phase: CorePhase):
        self._current_phase = phase

    def it_init(self):
        # unless overridden we dispatch the trainer-independent `it_init`
        it_init(**self.trainer_cfg.it_session)

    def it_session_end(self):
        """Used to dispatch any phase-specific session end hooks."""
        # unless overridden we dispatch the trainer-independent `it_session_end`
        it_session_end(session_type=self.phase, **self.trainer_cfg.it_session)

    def _run(self, phase, loop_fn, *args: Any, **kwargs: Any) -> Any:
        self.phase = CorePhase[phase]
        loop_fn(**self.trainer_cfg.__dict__)
        self.it_session_end()

    test = partialmethod(_run, phase="test", loop_fn=core_test_loop)
    train = partialmethod(_run, phase="train", loop_fn=core_train_loop)
