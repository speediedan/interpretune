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
# TODO: fill in this placeholder with actual core helper functions
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Any, Union, Dict, NamedTuple
from collections import defaultdict
from copy import deepcopy

import pytest
import torch
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.base.config_classes import ITDataModuleConfig, ITConfig, CorePhases
from interpretune.base.it_datamodule import ITDataModule
from interpretune.base.it_module import ITModule, BaseITModule
from interpretune.base.call import _call_itmodule_hook, it_init, it_session_end
from interpretune.utils.types import Optimizable
from tests.helpers.cfg_aliases import (RUNIF_ALIASES, test_datamodule_kwargs, test_shared_config, test_it_module_base,
                                       test_it_module_optim, MemProfResult, test_dataset_state)
from tests.helpers.runif import RunIf
from tests.helpers.utils import make_deterministic
from tests.helpers.lightning_utils import cuda_reset, to_device
from tests.helpers.modules import (TestITDataModule, TestITDataModuleFullDataset, TestITLightningDataModule,
                                   TestITLightningDataModuleFullDataset, TestITModule, TestITLightningModule)

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import seed_everything, Trainer
else:
    seed_everything = object
    Trainer = object


################################################################################
# Core test generation and encapsulation
################################################################################

@dataclass(kw_only=True)
class TestCfg():
    alias: str
    cfg: Optional[Dict | Tuple] = None
    marks: Optional[Tuple] = None
    expected: Optional[Dict] = None
    result_gen: Optional[Callable] = None

    def __post_init__(self):
        if self.expected is None and self.result_gen is not None:
            assert callable(self.result_gen), "result_gen must be callable"
            self.expected = self.result_gen(self.alias)
        if self.cfg is None and self.cfg_gen is not None:
            assert callable(self.cfg_gen), "cfg_gen must be callable"
            self.cfg = self.cfg_gen(self.alias)
        elif isinstance(self.cfg, Dict):
            self.cfg = self.cfg[self.alias]

def get_marks(marks: Union[Dict, str]) -> RunIf:
    # support RunIf aliases
    if isinstance(marks, Dict):
        return RunIf(**marks)
    elif isinstance(marks, str):
        return RunIf(**RUNIF_ALIASES[marks])
    else:
        raise ValueError(f"Unexpected marks type (should be Dict or str): {type(marks)}")

def pytest_param_factory(test_configs: List[TestCfg], unpack: bool = True) -> List:
    return [pytest.param(
            config.alias,
            *config.cfg if unpack else (config.cfg,),
            id=config.alias,
            marks=get_marks(config.marks) if config.marks else tuple(),
        )
        for config in test_configs
    ]


################################################################################
# Expected result generation and encapsulation
################################################################################

class TestResult(NamedTuple):
    result_alias: Optional[str] = None  # N.B. diff test aliases may map to the same result alias (e.g. parity tests)
    exact_results: Optional[Tuple] = None
    close_results: Optional[Tuple] = None
    mem_results: Optional[Tuple] = None
    tolerance_map: Optional[Dict[str, float]] = None

def mem_results(results: Tuple):
    """Result generation function for memory profiling tests."""
    # See NOTE [Memprofiler Key Format]
    # snap keys are src.rank.phase.epoch_idx.step_idx.step_ctx
    loop_type, src, test_values = results
    mem_keys = MemProfResult.cuda_mem_keys if src == "cuda" else ('rss_diff',)
    step_key = f'{src}.{MemProfResult.train_keys[src]}' if loop_type == 'train' else f'{src}.{MemProfResult.test_key}'
    # default tolerance of rtol=0.05, atol=0 for all keys unless overridden with an explicit `tolerance_map`
    tolerance_map = {'tolerance_map': {k: (0.05, 0) for k in mem_keys}}
    return {**tolerance_map, 'expected_memstats': (step_key, mem_keys, test_values)}

def close_results(close_map: Tuple):
    """Result generation function that packages expected close results with a provided tolerance dict or generates
    a default one based upon the test_alias."""
    expected_close = defaultdict(dict)
    close_keys = set()
    for e, k, v in close_map:
        expected_close[e][k] = v
        close_keys.add(k)
    closestats_tol = {'tolerance_map': {k: (0.1, 0) for k in close_keys}}
    return {**closestats_tol, 'expected_close': expected_close}

def exact_results(expected_exact: Tuple):
    """Result generation function that packages expected close results with a provided tolerance dict or generates
    a default one based upon the test_alias."""
    return {'expected_exact': expected_exact}

def def_results(device_type: str, precision: Union[int, str]):

    # wrap result dict such that only the first epoch is checked
    return {0: {"device_type": device_type, "precision": get_model_input_dtype(precision),
                "dataset_state": test_dataset_state},}

RESULT_TYPE_MAPPING = {
    "exact_results": exact_results,
    "close_results": close_results,
    "mem_results": mem_results,
}

def parity_normalize(test_alias) -> str:
    parity_suffixes = ("_l",)  # likely will add other parity suffixes in the future
    for ps in parity_suffixes:
        if test_alias.endswith(ps):
            test_alias = test_alias[:-len(ps)]
            break  # don't support multiple suffixes at once
    return test_alias

def collect_results(result_map: Dict[str, Tuple], test_alias: str):
    test_alias = parity_normalize(test_alias)
    test_result: TestResult = result_map[test_alias]
    collected_results = defaultdict(dict)
    for rtype, rfunc in RESULT_TYPE_MAPPING.items():
        if rattr := getattr(test_result, rtype):
            collected_results.update(rfunc(rattr))
    if exp_tol := test_result.tolerance_map:
        collected_results['tolerance_map'].update(exp_tol)
    return collected_results


################################################################################
# Configuration composition
################################################################################

def get_model_input_dtype(precision):
    if precision in ("float16", "16-true", "16-mixed", "16", 16):
        return torch.float16
    if precision in ("bfloat16", "bf16-true", "bf16-mixed", "bf16"):
        return torch.bfloat16
    if precision in ("64-true", "64", 64):
        return torch.double
    return torch.float32

def get_itdm_cfg(dm_override_cfg: Optional[Dict] = None, **kwargs) -> ITConfig:
    test_it_datamodule_cfg = deepcopy(test_datamodule_kwargs)
    if dm_override_cfg:
        test_it_datamodule_cfg.update(dm_override_cfg)
    return ITDataModuleConfig(**test_shared_config, **test_it_datamodule_cfg)

def get_it_cfg(test_cfg: TestCfg, core_log_dir: Optional[str| os.PathLike] = None) -> ITConfig:
    if test_cfg.loop_type == "test":
        test_it_module_cfg = deepcopy(test_it_module_base)
    elif test_cfg.loop_type == "train":
        test_it_module_cfg = deepcopy(test_it_module_optim)
    if test_cfg.act_ckpt:
        test_it_module_cfg.update({"activation_checkpointing": True})
    if test_cfg.from_pretrained_cfg:
        test_it_module_cfg["from_pretrained_cfg"].update(test_cfg.from_pretrained_cfg)
    if test_cfg.memprofiling_cfg:
        test_it_module_cfg['memprofiler_cfg'] = test_cfg.memprofiling_cfg
    if test_cfg.cust_fwd_kwargs:
        test_it_module_cfg['cust_fwd_kwargs'].update(test_cfg.cust_fwd_kwargs)
    if core_log_dir:
        test_it_module_cfg.update({'core_log_dir': core_log_dir})
    test_it_module_cfg = configure_device_precision(test_it_module_cfg, test_cfg.device_type, test_cfg.precision)
    return ITConfig(**test_it_module_cfg)

def configure_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> Dict[str, Any]:
    cfg['from_pretrained_cfg'].update({'torch_dtype': get_model_input_dtype(precision)})
    if device_type == "cuda":
        cfg['from_pretrained_cfg'].update({'device_map': 0})
    return cfg

def datamodule_factory(test_cfg: TestCfg, force_prepare_data: bool = False) -> ITDataModule:
    itdm_cfg = get_itdm_cfg(dm_override_cfg=test_cfg.dm_override_cfg)
    if test_cfg.lightning:
        datamodule_class = TestITLightningDataModuleFullDataset if test_cfg.full_dataset else TestITLightningDataModule
    else:
        datamodule_class = TestITDataModuleFullDataset if test_cfg.full_dataset else TestITDataModule
    return datamodule_class(itdm_cfg=itdm_cfg, force_prepare_data=force_prepare_data)

def config_modules(test_cfg, test_alias, expected_results, tmp_path,
                   state_log_mode: bool = False) -> Tuple[ITDataModule, BaseITModule]:
    fill_uninitialized_memory = True
    if test_cfg.lightning:
        seed_everything(1, workers=True)
        # maximally align our core PyTorch reproducibility settings w/ Lightning
        # (may remove this in the future if Lightning adds this option to seed everything)
        torch._C._set_deterministic_fill_uninitialized_memory(fill_uninitialized_memory)
    else:
        make_deterministic(warn_only=True, fill_uninitialized_memory=fill_uninitialized_memory)
    cuda_reset()
    torch.set_printoptions(precision=12)
    datamodule = datamodule_factory(test_cfg)
    it_cfg = get_it_cfg(test_cfg, core_log_dir=tmp_path)
    module_class = TestITLightningModule if test_cfg.lightning else TestITModule
    module = module_class(it_cfg=it_cfg, test_alias=test_alias,
                          state_log_dir=tmp_path if state_log_mode else None,  # optionally enable expected state logs
                          **expected_results,)
    return datamodule, module


################################################################################
# Core Train/Test Orchestration
################################################################################

def run_it(module: ITModule, datamodule: ITDataModule, test_cfg: TestCfg):
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
        module._current_epoch = epoch_idx
        iterator = iter(train_dataloader)
        _call_itmodule_hook(module, hook_name="on_train_epoch_start", hook_msg="Running train epoch start hooks")
        for batch_idx in range(train_steps):
            run_step(step_fn="training_step", iterator=iterator, batch_idx=batch_idx, **train_ctx)
            _call_itmodule_hook(module, hook_name="on_train_epoch_end", hook_msg="Running train epoch end hooks")
        if val_steps > 0:
            module.model.eval()
            iterator = iter(val_dataloader)
            for batch_idx in range(val_steps):
                with torch.inference_mode():
                    run_step(step_fn="validation_step", iterator=iterator, batch_idx=batch_idx, **train_ctx)
        module.model.train()

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

def fetch_batch(iterator, module) -> BatchEncoding:
    batch = next(iterator)
    to_device(module.model.device, batch)
    return batch


################################################################################
# Lightning Train/Test Orchestration
################################################################################

def run_lightning(module: ITModule, datamodule: ITDataModule, test_cfg: TestCfg, tmp_path: Path) -> Trainer:
    accelerator = "cpu" if test_cfg.device_type == "cpu" else "gpu"
    trainer_steps = {"limit_train_batches": test_cfg.train_steps, "limit_val_batches": test_cfg.val_steps,
                     "limit_test_batches": test_cfg.test_steps, "limit_predict_batches": 1,
                     "max_steps": test_cfg.train_steps}
    trainer = Trainer(default_root_dir=tmp_path, devices=1, deterministic=True, accelerator=accelerator, max_epochs=1,
                      precision=lightning_prec_alias(test_cfg.precision), num_sanity_val_steps=0, **trainer_steps)
    lightning_func = trainer.fit if test_cfg.loop_type == "train" else trainer.test
    lightning_func(datamodule=datamodule, model=module)
    return trainer

def lightning_prec_alias(precision: str):
    # TODO: update this and get_model_input_dtype() to use a shared set of supported alias mappings once more
    # types are tested
    return "bf16-true" if precision == "bf16" else "32-true"


# TODO: add/test usage of this fixture that inits a basic module/datamodule pair to used for all tests at a function
# (or maybe module) level that use this fixure
#@pytest.fixture(scope="function")
# def test_module_datamodule_base_cuda() -> Tuple[TestITModule,TestITDataModule]:
#     """A fixture that generates a 'best' and 'kth' checkpoint to be used in scheduled fine-tuning resumption
#     testing."""
#     it_cfg = TestITConfigBase
#     torch_dtype = get_model_input_dtype(32)
#     # TODO: update to only override torch_dtype in from_pretrained_cfg only once torch_dtype is a module property
#     torch_dtype_attrs = ("from_pretrained_cfg['torch_dtype']", "torch_dtype")
#     for attr in torch_dtype_attrs:
#         setattr(it_cfg, attr, torch_dtype)
#     it_cfg.from_pretrained_cfg.update({'device_map': 0})
#     cuda_reset()
#     datamodule = TestITDataModule(itdm_cfg=TestITDataModuleConfig)
#     module = TestITModule(it_cfg=it_cfg)
#     it_init(module=module, datamodule=datamodule)
#     return module, datamodule

# while this fixture works to avoid reinitilizing the fulldataset datamodule for each test, most of the resource
# consumption is in the setup/init which is not shared so we use a factory function to get the datamodule for
# each configuration instead of a fixture to bolster flexibility at the expense of some additional initialization cost
# @pytest.fixture(scope="module")
# def fulldataset_datamodule() -> TestITDataModule:
#     return TestITDataModuleFullDataset(itdm_cfg=get_itdm_cfg())

# TODO: add a test using this loop validating memprofile_ctx as a context manager like this (vs decorator invocation)
# def run_profile_ctx_train_step(module, optimizer, iterator, batch_idx, device_type, profiler, epoch_idx = 0):
#     with ProfilerHooksMixin.memprofile_ctx(module.memprofiler, epoch_idx=epoch_idx, step_idx=batch_idx,
#                                             phase="train"):
#         batch = fetch_batch(iterator, module)
#         optimizer.zero_grad()
#         if module.torch_dtype == torch.bfloat16:
#             with torch.autocast(device_type=device_type, dtype=module.torch_dtype):
#                 loss = module.training_step(batch, batch_idx)
#         else:
#             loss = module.training_step(batch, batch_idx)
#         loss.backward()
#         optimizer.step()
