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
from typing import List, Optional, Tuple, Callable, Any, Union, Dict, NamedTuple
from collections import defaultdict
from copy import deepcopy

import pytest
import torch

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.base.config.module import ITConfig
from interpretune.plugins.transformer_lens import ITLensFromPretrainedConfig, ITLensConfig
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import BaseITModule
from base.cfg_aliases import (test_core_datamodule_kwargs, test_core_it_module_base, test_core_it_module_optim,
                              test_core_shared_config, test_dataset_state_core, expected_first_fwd_ids, MemProfResult,)
from tests.plugins.transformer_lens.cfg_aliases import (test_tl_it_module_base, test_tl_it_module_optim,
                                                        test_tl_datamodule_kwargs, test_tl_shared_config,
                                                        test_dataset_state_tl)
from tests.utils.runif import RunIf, RUNIF_ALIASES
from tests.utils.lightning import cuda_reset
from tests.plugins.transformer_lens.modules import TestITLensModule,TestITLensLightningModule
from base.modules import (TestITDataModule, TestITDataModuleFullDataset, TestITLightningDataModule, TestITModule,
                          TestITLightningDataModuleFullDataset, TestITLightningModule,)

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import seed_everything
else:
    seed_everything = object


################################################################################
# Core test generation and encapsulation
################################################################################

@dataclass(kw_only=True)
class TestCfg():
    alias: str
    cfg: Optional[Tuple] = None
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

class ParityCfg(NamedTuple):
    loop_type: str = "train"
    device_type: str = "cpu"
    precision: str | int = 32
    full_dataset: bool = False
    act_ckpt: bool = False
    lightning: bool = False
    transformerlens: bool = False
    train_steps: Optional[int] = 1
    val_steps: Optional[int] = 1
    test_steps: Optional[int] = 1
    dm_override_cfg: Optional[Dict] = None
    memprofiler_cfg: Optional[Dict] = None
    auto_model_cfg: Optional[Dict] = None
    tl_from_pretrained_cfg: Optional[Dict] = None
    from_pretrained_cfg: Optional[Dict] = None
    cust_fwd_kwargs: Optional[Dict] = None
    force_prepare_data: bool = False

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
    # snap keys are rank.phase.epoch_idx.step_idx.step_ctx
    loop_type, src, test_values = results
    mem_keys = MemProfResult.cuda_mem_keys if src == "cuda" else MemProfResult.cpu_mem_keys[loop_type]
    step_key = f'{MemProfResult.train_keys[src]}' if loop_type == 'train' else f'{MemProfResult.test_key}'
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
    """Result generation function that packages."""
    return {'expected_exact': expected_exact}

def def_results(device_type: str, precision: Union[int, str], dataset_type: Optional[str] = "core",
                ds_cfg: Optional[str] = "train_prof"):
    # wrap result dict such that only the first epoch is checked
    test_dataset_state = test_dataset_state_core if dataset_type == "core" else test_dataset_state_tl
    test_dataset_state = test_dataset_state + expected_first_fwd_ids[ds_cfg]
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

TEST_DATAMODULE_MAPPING = {
    # (lightning, full_dataset)
    (False, False): TestITDataModule,
    (True, False): TestITLightningDataModule,
    (False,True): TestITDataModuleFullDataset,
    (True, True): TestITLightningDataModuleFullDataset,
}

TEST_MODULE_MAPPING = {
    # (lightning, transformerlens)
    (False, False): TestITModule,
    (True, False): TestITLightningModule,
    (False, True): TestITLensModule,
    (True, True): TestITLensLightningModule,
}

def get_model_input_dtype(precision):
    if precision in ("float16", "16-true", "16-mixed", "16", 16):
        return torch.float16
    if precision in ("bfloat16", "bf16-true", "bf16-mixed", "bf16"):
        return torch.bfloat16
    if precision in ("64-true", "64", 64):
        return torch.double
    return torch.float32

def get_itdm_cfg(test_cfg: Tuple, dm_override_cfg: Optional[Dict] = None, **kwargs) -> ITConfig:
    if test_cfg.transformerlens:
        default_itdm_kwargs = test_tl_datamodule_kwargs
        shared_config = test_tl_shared_config
    else:
        default_itdm_kwargs = test_core_datamodule_kwargs
        shared_config = test_core_shared_config
    test_it_datamodule_cfg = deepcopy(default_itdm_kwargs)
    if dm_override_cfg:
        test_it_datamodule_cfg.update(dm_override_cfg)
    return ITDataModuleConfig(**shared_config, **test_it_datamodule_cfg)

def get_it_cfg(test_cfg: Tuple, core_log_dir: Optional[str| os.PathLike] = None) -> ITConfig:
    test_cfg_override_attrs = ["from_pretrained_cfg", "memprofiler_cfg", "cust_fwd_kwargs", "auto_model_cfg",
                               "tl_from_pretrained_cfg"]
    if test_cfg.loop_type == "test":
        target_test_it_module_cfg = test_tl_it_module_base if test_cfg.transformerlens else test_core_it_module_base
    elif test_cfg.loop_type == "train":
        target_test_it_module_cfg = test_tl_it_module_optim if test_cfg.transformerlens else test_core_it_module_optim
    test_it_module_cfg = deepcopy(target_test_it_module_cfg)
    if test_cfg.act_ckpt:
        test_it_module_cfg.update({"activation_checkpointing": True})
    for attr in test_cfg_override_attrs:
        if getattr(test_cfg, attr):
            test_it_module_cfg.update({attr: getattr(test_cfg, attr)})
    if core_log_dir:
        test_it_module_cfg.update({'core_log_dir': core_log_dir})
    test_it_module_cfg = configure_device_precision(test_it_module_cfg, test_cfg.device_type, test_cfg.precision)
    if test_cfg.transformerlens:
        test_it_module_cfg['tl_from_pretrained_cfg'] = \
            ITLensFromPretrainedConfig(**test_it_module_cfg['tl_from_pretrained_cfg'])
        return ITLensConfig(**test_it_module_cfg)
    return ITConfig(**test_it_module_cfg)

def configure_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> Dict[str, Any]:
    cfg['from_pretrained_cfg'].update({'torch_dtype': get_model_input_dtype(precision)})
    if device_type == "cuda":
        cfg['from_pretrained_cfg'].update({'device_map': 0})
    if cfg.get('tl_from_pretrained_cfg', None):
        cfg['tl_from_pretrained_cfg'].update({'dtype': get_model_input_dtype(precision), 'device': device_type})
    return cfg

def datamodule_factory(test_cfg: Tuple) -> ITDataModule:
    itdm_cfg = get_itdm_cfg(test_cfg=test_cfg, dm_override_cfg=test_cfg.dm_override_cfg)
    datamodule_class = TEST_DATAMODULE_MAPPING[(test_cfg.lightning, test_cfg.full_dataset)]
    return datamodule_class(itdm_cfg=itdm_cfg, force_prepare_data=test_cfg.force_prepare_data)

def config_modules(test_cfg, test_alias, expected_results, tmp_path,
                   state_log_mode: bool = False) -> Tuple[ITDataModule, BaseITModule]:
    if test_cfg.lightning: # allow Lightning to set env vars
        seed_everything(1, workers=True)
    cuda_reset()
    torch.set_printoptions(precision=12)
    datamodule = datamodule_factory(test_cfg)
    it_cfg = get_it_cfg(test_cfg, core_log_dir=tmp_path)
    module_class = TEST_MODULE_MAPPING[(test_cfg.lightning, test_cfg.transformerlens)]
    module = module_class(it_cfg=it_cfg, test_alias=test_alias,
                          state_log_dir=tmp_path if state_log_mode else None,  # optionally enable expected state logs
                          **expected_results,)
    return datamodule, module
