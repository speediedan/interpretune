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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Any, Union, Dict, NamedTuple, Type
from collections import defaultdict
from copy import deepcopy
import os

import pytest
import torch

from interpretune.base.config.datamodule import ITDataModuleConfig
from it_examples.experiments.rte_boolq.config import RTEBoolqConfig, RTEBoolqTLConfig
from interpretune.base.config.module import ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import BaseITModule
from interpretune.plugins.transformer_lens import ITLensFromPretrainedConfig, ITLensCustomConfig
from interpretune.base.contract.session import InterpretunableSessionConfig
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.utils.types import StrOrPath
from tests.base.cfg_aliases import (test_core_datamodule_kwargs, test_core_it_module_base, test_core_it_module_optim,
                              test_core_shared_config, test_dataset_state_core, expected_first_fwd_ids, MemProfResult,)
from tests.plugins.transformer_lens.cfg_aliases import (
    test_tl_pretrained_it_module_base, test_tl_cust_it_module_base, test_tl_pretrained_it_module_optim,
    test_tl_cust_it_module_optim, test_tl_datamodule_kwargs, test_tl_shared_config, test_dataset_state_tl)
from tests.utils.runif import RunIf, RUNIF_ALIASES
from tests.utils.lightning import cuda_reset
from tests.modules import (RTETestITLensModule, RTETestITLensLightningModule, TestITDataModule,
                           TestITDataModuleFullDataset, TestITLightningDataModule, RTETestITModule,
                           TestITLightningDataModuleFullDataset, RTETestITLightningModule,)

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import seed_everything
else:
    seed_everything = object


IT_GLOBAL_STATE_LOG_MODE = os.environ.get("IT_GLOBAL_STATE_LOG_MODE", "0") == "1"

################################################################################
# Core test generation and encapsulation
################################################################################

@dataclass(kw_only=True)
class TestCfg:
    alias: str
    cfg: Optional[Tuple] = None
    marks: Optional[Dict] = None  # test instance-specific marks
    expected: Optional[Dict] = None
    result_gen: Optional[Callable] = None
    function_marks: Dict[str, Any] = field(default_factory=dict)  # marks applied at test function level

    def __post_init__(self):
        if self.expected is None and self.result_gen is not None:
            assert callable(self.result_gen), "result_gen must be callable"
            self.expected = self.result_gen(self.alias)
        if self.cfg is None and self.cfg_gen is not None:
            assert callable(self.cfg_gen), "cfg_gen must be callable"
            self.cfg = self.cfg_gen(self.alias)
        elif isinstance(self.cfg, Dict):
            self.cfg = self.cfg[self.alias]
        if self.marks or self.function_marks:
            self.marks = self._get_marks(self.marks, self.function_marks)

    def _get_marks(self, marks: Optional[Dict | str], function_marks: Dict) -> Optional[RunIf]:
        # support RunIf aliases
        if marks:
            if isinstance(marks, Dict):
                function_marks.update(marks)
            elif isinstance(marks, str):
                function_marks.update(RUNIF_ALIASES[marks])
            else:
                raise ValueError(f"Unexpected marks input type (should be Dict, str or None): {type(marks)}")
        if function_marks:
            return RunIf(**function_marks)

def pytest_param_factory(test_configs: List[TestCfg], unpack: bool = True) -> List:
    return [pytest.param(
            config.alias,
            *config.cfg if unpack else (config.cfg,),
            id=config.alias,
            marks=config.marks or tuple(),
        )
        for config in test_configs
    ]

@dataclass(kw_only=True)
class ParityCfg:
    loop_type: str = "train"
    device_type: str = "cpu"
    model_key: str = "rte"  # "real-model"-based acceptance/parity testing/profiling
    precision: str | int = 32
    full_dataset: bool = False
    act_ckpt: bool = False
    lightning: bool = False
    plugin: Optional[str] = None
    plugin_cfg_key: Optional[str] = None
    train_steps: Optional[int] = 1
    val_steps: Optional[int] = 1
    test_steps: Optional[int] = 1
    dm_override_cfg: Optional[Dict] = None
    memprofiler_cfg: Optional[Dict] = None
    #auto_model_cfg: Optional[Dict] = None
    tl_cfg: Optional[Dict] = None
    #hf_from_pretrained_cfg: Optional[Dict] = None
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

def def_results(device_type: str, precision: Union[int, str], dataset_type: str = "core",
                ds_cfg: str = "train_prof"):
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


########################################################################################################################
# Configuration composition
########################################################################################################################

TEST_DATAMODULE_BASE_CONFIGS = {
    "core": (test_core_shared_config, test_core_datamodule_kwargs),
    "transformerlens": (test_tl_shared_config, test_tl_datamodule_kwargs),
}

TEST_MODULE_BASE_CONFIGS = {
    # (loop_type, plugin_key, plugin_cfg_key)
    ("test", None, None): test_core_it_module_base,
    ("train", None, None): test_core_it_module_optim,
    ("test", "transformerlens", "pretrained"): test_tl_pretrained_it_module_base,
    ("train", "transformerlens", "pretrained"): test_tl_pretrained_it_module_optim,
    ("test", "transformerlens", "cust"): test_tl_cust_it_module_base,
    ("train", "transformerlens", "cust"): test_tl_cust_it_module_optim,
}

TEST_DATAMODULE_MAPPING = {
    # (lightning, full_dataset)
    (False, False): TestITDataModule,
    (True, False): TestITLightningDataModule,
    (False,True): TestITDataModuleFullDataset,
    (True, True): TestITLightningDataModuleFullDataset,
}

TEST_MODULE_MAPPING = {
    # TODO: only using rte module right now for "real model"-based parity/acceptance testing/profiling but will expand
    #to use different/toy model types for unit testing in the future
    # (model_key, lightning, plugin_key)
    ("rte", False, None): RTETestITModule,
    ("rte", True, None): RTETestITLightningModule,
    ("rte", False, "transformerlens"): RTETestITLensModule,
    ("rte", True, "transformerlens"): RTETestITLensLightningModule,
}

MODULE_CONFIG_MAPPING = {
    # (model_key, plugin_key)
    ("rte", None): RTEBoolqConfig,
    ("rte", "transformerlens"): RTEBoolqTLConfig
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
    shared_config, default_itdm_kwargs  = TEST_DATAMODULE_BASE_CONFIGS[test_cfg.plugin or "core"]
    test_it_datamodule_cfg = deepcopy(default_itdm_kwargs)
    if dm_override_cfg:
        test_it_datamodule_cfg.update(dm_override_cfg)
    return ITDataModuleConfig(**shared_config, **test_it_datamodule_cfg)

def init_plugin_cfg(test_cfg: Tuple, test_it_module_cfg: Dict):
    if test_cfg.plugin == "transformerlens":
        if test_cfg.plugin_cfg_key == "pretrained":
            test_it_module_cfg['tl_cfg'] = ITLensFromPretrainedConfig(**test_it_module_cfg['tl_cfg'])
        elif test_cfg.plugin_cfg_key == "cust":
            test_it_module_cfg['tl_cfg'] = ITLensCustomConfig(**test_it_module_cfg['tl_cfg'])
        else:
            raise ValueError(f"Unknown plugin_cfg_key: {test_cfg.plugin_cfg_key}")
    else:  # See NOTE [Interpretability Plugins]
        raise ValueError(f"Unknown plugin type: {test_cfg.plugin}")

def get_it_cfg(test_cfg: Tuple, core_log_dir: Optional[StrOrPath] = None) -> ITConfig:
    test_cfg_override_attrs = ["memprofiler_cfg", "cust_fwd_kwargs", "tl_cfg"]
    target_test_it_module_cfg = TEST_MODULE_BASE_CONFIGS[(test_cfg.loop_type, test_cfg.plugin, test_cfg.plugin_cfg_key)]
    test_it_module_cfg = deepcopy(target_test_it_module_cfg)
    if test_cfg.act_ckpt:
        test_it_module_cfg['hf_from_pretrained_cfg'].activation_checkpointing = True
    for attr in test_cfg_override_attrs:
        if getattr(test_cfg, attr):
            test_it_module_cfg.update({attr: getattr(test_cfg, attr)})
    if core_log_dir:
        test_it_module_cfg.update({'core_log_dir': core_log_dir})
    test_it_module_cfg = configure_device_precision(test_it_module_cfg, test_cfg.device_type, test_cfg.precision)
    if test_cfg.plugin:
        init_plugin_cfg(test_cfg, test_it_module_cfg)
    config_class = MODULE_CONFIG_MAPPING[(test_cfg.model_key, test_cfg.plugin)]
    return config_class(**test_it_module_cfg)

def configure_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> Dict[str, Any]:
    if cfg.get('hf_from_pretrained_cfg', None) is not None:
        cfg['hf_from_pretrained_cfg'].pretrained_kwargs.update({'torch_dtype': get_model_input_dtype(precision)})
        if device_type == "cuda":
            # note that with TLens this should be overridden by the tl plugin but we want to test that functionality
            cfg['hf_from_pretrained_cfg'].pretrained_kwargs.update({'device_map': 0})
    if cfg.get('tl_cfg', None) is not None:
        dev_prec_override = {'dtype': get_model_input_dtype(precision), 'device': device_type}
        if cfg['tl_cfg'].get('cfg', None) is not None:  # TL custom model config
            cfg['tl_cfg']['cfg'].update(dev_prec_override)
        else:  # TL from pretrained config, we set directly in addition to pretrained above to verify sync behavior
            cfg['tl_cfg'].update(dev_prec_override)
    return cfg

def datamodule_config_factory(test_cfg: Tuple) -> Tuple[Type[ITDataModule], ITDataModuleConfig]:
    itdm_cfg = get_itdm_cfg(test_cfg=test_cfg, dm_override_cfg=test_cfg.dm_override_cfg)
    datamodule_class = TEST_DATAMODULE_MAPPING[(test_cfg.lightning, test_cfg.full_dataset)]
    return datamodule_class, itdm_cfg

def module_config_factory(test_cfg: Tuple, core_log_dir: StrOrPath) -> Tuple[Type[BaseITModule], ITConfig]:
    it_cfg = get_it_cfg(test_cfg=test_cfg, core_log_dir=core_log_dir)
    module_class = TEST_MODULE_MAPPING[(test_cfg.model_key, test_cfg.lightning, test_cfg.plugin)]
    return module_class, it_cfg

def config_session(core_cfg, test_cfg, test_alias, expected, state_log_dir):
    session_type = {'lightning': test_cfg.lightning, 'plugin': test_cfg.plugin}
    dm_kwargs = {'dm_kwargs': {'force_prepare_data': test_cfg.force_prepare_data}}
    module_kwargs = {'module_kwargs': {'test_alias': test_alias, 'state_log_dir': state_log_dir, **expected}}
    session_cfg = InterpretunableSessionConfig(**core_cfg, **session_type, **dm_kwargs, **module_kwargs)
    return session_cfg

def config_modules(test_cfg, test_alias, expected_results, tmp_path,
                   state_log_mode: bool = False) -> Tuple[ITDataModule, BaseITModule]:
    if test_cfg.lightning: # allow Lightning to set env vars
        seed_everything(1, workers=True)
    cuda_reset()
    torch.set_printoptions(precision=12)
    dm_cls, itdm_cfg = datamodule_config_factory(test_cfg)
    module_cls, it_cfg = module_config_factory(test_cfg, core_log_dir=tmp_path)
    core_cfg = {'datamodule_cfg': itdm_cfg, 'datamodule_cls': dm_cls, 'module_cfg': it_cfg, 'module_cls': module_cls}
    state_log_dir = tmp_path if state_log_mode else None
    it_session = config_session(core_cfg, test_cfg, test_alias, expected_results, state_log_dir)
    return it_session.datamodule, it_session.module
