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
# Initially based on https://bit.ly/3oQ8Vqf
from pathlib import Path
from typing import Optional, Tuple
from collections import defaultdict
from unittest import mock

import torch
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.base.config.shared import CorePhases
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import ITModule
from interpretune.base.call import _call_itmodule_hook, it_init, it_session_end
from interpretune.utils.types import Optimizable
from tests.configuration import config_modules
from tests.utils.lightning import to_device, move_data_to_device


if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
else:
    Trainer = object
    ModelCheckpoint = object


################################################################################
# Core Train/Test Orchestration
################################################################################

########################################################################################################################
# NOTE: [Parity Testing Approach]
# - We use a single set of results but separate tests for core/lightning parity tests since Lightning is not a required
#   dependency for Interpretune and we want to mark at the test-level for greater clarity and flexibility (we want to
#   signal clearly when either diverges from the expected benchmark so aren't testing relative values only)
# - The configuration space for parity tests is sampled rather than exhaustively testing all framework configuration
#   combinations due to resource constraints
# - Note that while we could access test_alias using the request fixture (`request.node.callspec.id`), this approach
#   using dataclass encapsulation allows us to flexibly define test ids, configurations, marks and expected outputs
#   together
# - We always check for basic exact match on device type, precision and dataset state
# - Our result mapping function uses these shared results for all supported parity test suffixes (e.g. '_l')
# - Set `state_log_mode=True` by setting the environmental variable `IT_GLOBAL_STATE_LOG_MODE` to `1` during development
#   to generate/dump state logs for tests rather than testing the relevant assertions (this can be manually overridden
#   on a test-by-test basis as well)
########################################################################################################################

def parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode: bool = False):
    datamodule, module = config_modules(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    if test_cfg.lightning:
        _ = run_lightning(module, datamodule, test_cfg, tmp_path)
    else:
        run_it(module, datamodule, test_cfg)

def run_it(module: ITModule, datamodule: ITDataModule, test_cfg: Tuple):
    it_init(module=module, datamodule=datamodule)
    if test_cfg.loop_type == "test":
        core_test_loop(module=module, datamodule=datamodule, device_type=test_cfg.device_type,
                       test_steps=test_cfg.test_steps)
    elif test_cfg.loop_type == "train":
        core_train_loop(module=module, datamodule=datamodule, device_type=test_cfg.device_type,
                        train_steps=test_cfg.train_steps, val_steps=test_cfg.val_steps)
    else:
        raise ValueError("Unsupported loop type, loop_type must be 'test' or 'train'")
    if test_cfg.loop_type=="train":
        session_type = CorePhases.train
    else:
        session_type = CorePhases.test
    it_session_end(module=module, datamodule=datamodule, session_type=session_type)

def core_train_loop(
    module: ITModule,
    datamodule: ITDataModule,
    device_type: str,
    train_steps: int = 1,
    epochs: int = 1,
    val_steps: int = 1,
):
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader() if val_steps > 0 else None
    optim = module.it_optimizers[0]
    train_ctx = {"module": module, "optimizer": optim, "device_type": device_type}
    for epoch_idx in range(epochs):
        module.model.train()
        module.current_epoch = epoch_idx
        iterator = iter(train_dataloader)
        _call_itmodule_hook(module, hook_name="on_train_epoch_start", hook_msg="Running train epoch start hooks")
        for batch_idx in range(train_steps):
            run_step(step_fn="training_step", iterator=iterator, batch_idx=batch_idx, **train_ctx)
        if val_steps > 0:
            module.model.eval()
            iterator = iter(val_dataloader)
            for batch_idx in range(val_steps):
                with torch.inference_mode():
                    run_step(step_fn="validation_step", iterator=iterator, batch_idx=batch_idx, **train_ctx)
        module.model.train()
        _call_itmodule_hook(module, hook_name="on_train_epoch_end", hook_msg="Running train epoch end hooks")

def core_test_loop(
    module: ITModule,
    datamodule: ITDataModule,
    device_type: str,
    test_steps: int = 1
):
    dataloader = datamodule.test_dataloader()
    test_ctx = {"module": module, "device_type": device_type}
    module._current_epoch = 0
    module.model.eval()
    iterator = iter(dataloader)
    for batch_idx in range(test_steps):
        with torch.inference_mode():
            run_step(step_fn="test_step", iterator=iterator, batch_idx=batch_idx, **test_ctx)
    _call_itmodule_hook(module, hook_name="on_test_epoch_end", hook_msg="Running test epoch end hooks")

def run_step(step_fn, module, iterator, batch_idx, device_type, optimizer: Optional[Optimizable] = None):
    batch = fetch_batch(iterator, module)
    step_func = getattr(module, step_fn)
    if module.global_step == 0 and step_fn != "validation_step":
        module.sampled_fwd_inputs = module.datamodule.sample_step_input(batch)
    if step_fn == "training_step":
        optimizer.zero_grad()
    if module.torch_dtype == torch.bfloat16:
        with torch.autocast(device_type=device_type, dtype=module.torch_dtype):
            loss = step_func(batch, batch_idx)
    else:
        loss = step_func(batch, batch_idx)
    if step_fn == "training_step":
        module.epoch_losses[module.current_epoch] = loss.item()
        loss.backward()
        optimizer.step()
    module.global_step += 1

def fetch_batch(iterator, module) -> BatchEncoding:
    batch = next(iterator)
    if hasattr(module, 'tl_cfg'):  # ensure the input is on the same device as TranformerLens assigns to layer 0
        move_data_to_device(batch, module.input_device)
    else:
        to_device(module.device, batch)
    return batch


################################################################################
# Lightning Train/Test Orchestration
################################################################################

def run_lightning(module: ITModule, datamodule: ITDataModule, test_cfg: Tuple, tmp_path: Path) -> Trainer:
    accelerator = "cpu" if test_cfg.device_type == "cpu" else "gpu"
    trainer_steps = {"limit_train_batches": test_cfg.train_steps, "limit_val_batches": test_cfg.val_steps,
                     "limit_test_batches": test_cfg.test_steps, "limit_predict_batches": 1,
                     "max_steps": test_cfg.train_steps}
    trainer = Trainer(default_root_dir=tmp_path, devices=1, deterministic=True, accelerator=accelerator, max_epochs=1,
                      precision=lightning_prec_alias(test_cfg.precision), num_sanity_val_steps=0, **trainer_steps)
    lightning_func = trainer.fit if test_cfg.loop_type == "train" else trainer.test
    with mock.patch.object(ModelCheckpoint, "_save_checkpoint"):  # do not save checkpoints for lightning tests
        lightning_func(datamodule=datamodule, model=module)
    return trainer

def lightning_prec_alias(precision: str):
    # TODO: update this and get_model_input_dtype() to use a shared set of supported alias mappings once more
    # types are tested
    return "bf16-true" if precision == "bf16" else "32-true"

################################################################################
# Simple Utility Functions
################################################################################

def dummy_step(*args, **kwargs) -> None:
    ...

def nones(num_n) -> Tuple:  # to help dedup config
    return (None,) * num_n

def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)
