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
from typing import List, Optional, Tuple, Callable, Any, Union, Dict, NamedTuple
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
import os
from functools import reduce

import pytest
import torch

from interpretune.base.config.datamodule import ITDataModuleConfig
from it_examples.experiments.rte_boolq.config import RTEBoolqConfig, RTEBoolqTLConfig
from interpretune.base.config.module import ITConfig
from interpretune.base.config.mixins import HFFromPretrainedConfig, ZeroShotClassificationConfig
from interpretune.adapters.transformer_lens import ITLensFromPretrainedConfig, ITLensCustomConfig
from interpretune.base.contract.session import ITSessionConfig, ITSession
from interpretune.adapters.registration import Adapter
from interpretune.adapters import ADAPTER_REGISTRY
from interpretune.extensions.memprofiler import MemProfilerCfg
from interpretune.extensions.debug_generation import DebugLMConfig
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.utils.types import StrOrPath
from tests.parity_acceptance.adapters.lightning.cfg_aliases import (
    expected_first_fwd_ids, MemProfResult, core_gpt2_datamodule_kwargs, core_cust_datamodule_kwargs,
    test_core_cust_it_module_base, core_cust_shared_config, test_core_cust_it_module_optim,
    test_core_gpt2_it_module_base, test_core_gpt2_it_module_optim, core_gpt2_shared_config, core_llama2_shared_config,
    core_llama2_datamodule_kwargs, test_dataset_state_core_gpt2, test_dataset_state_core_cust,
    test_dataset_state_core_llama2, test_core_llama2_it_module_base, test_core_llama2_it_module_optim)
from tests.parity_acceptance.adapters.transformer_lens.cfg_aliases import (
    test_tl_gpt2_it_module_base, test_tl_cust_it_module_base, test_tl_gpt2_it_module_optim,
    test_tl_cust_it_module_optim, test_tl_datamodule_kwargs, test_tl_gpt2_shared_config, test_dataset_state_tl)
from tests.utils.runif import RunIf, RUNIF_ALIASES
from tests.modules import TestITDataModule, TestITModule, Llama2TestITDataModule

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import seed_everything
else:
    seed_everything = object

# We use the same datamodule and module for all parity test contexts to ensure cross-adapter compatibility
# Default test modules
TEST_IT_DATAMODULE = TestITDataModule
TEST_IT_MODULE = TestITModule
CORE_SESSION_CFG = {'datamodule_cls': TEST_IT_DATAMODULE, 'module_cls': TEST_IT_MODULE}
DEFAULT_TEST_DATAMODULES = ('cust', 'gpt2')

# N.B. Some unit tests may require slightly modified/subclassed test datamodules to accommodate testing of functionality
# not compatible with the default GPT2-based test datamodule
TEST_IT_DATAMODULE_MAPPING = {'llama2': Llama2TestITDataModule}

IT_GLOBAL_STATE_LOG_MODE = os.environ.get("IT_GLOBAL_STATE_LOG_MODE", "0") == "1"


################################################################################
# Core test generation and encapsulation
################################################################################

@dataclass(kw_only=True)
class BaseAugTest:
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

def pytest_param_factory(test_configs: List[BaseAugTest], unpack: bool = True) -> List:
    return [pytest.param(
            config.alias,
            *config.cfg if unpack else (config.cfg,),
            id=config.alias,
            marks=config.marks or tuple(),
        )
        for config in test_configs
    ]

@dataclass(kw_only=True)
class BaseCfg:
    phase: str = "train"
    device_type: str = "cpu"
    model_key: str = "rte"  # "real-model"-based acceptance/parity testing/profiling
    precision: str | int = 32
    adapter_ctx: Iterable[Adapter | str] = (Adapter.core,)
    model_src_key: Optional[str] = None
    limit_train_batches: Optional[int] = 1
    limit_val_batches: Optional[int] = 1
    limit_test_batches: Optional[int] = 1
    dm_override_cfg: Optional[Dict] = None
    zero_shot_cfg: Optional[ZeroShotClassificationConfig] = None
    hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = None
    memprofiler_cfg: Optional[MemProfilerCfg] = None
    debug_lm_cfg: Optional[DebugLMConfig] = None
    model_cfg: Optional[Dict] = None
    tl_cfg: Optional[Dict] = None
    max_epochs: Optional[int] = 1
    cust_fwd_kwargs: Optional[Dict] = None
    # used when adding a new test dataset or changing a test model to force re-caching of test datasets
    force_prepare_data: bool = False

    def __post_init__(self):
        self.adapter_ctx = ADAPTER_REGISTRY.canonicalize_composition(self.adapter_ctx)

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
    phase, src, test_values = results
    mem_keys = MemProfResult.cuda_mem_keys if src == "cuda" else MemProfResult.cpu_mem_keys[phase]
    step_key = f'{MemProfResult.train_keys[src]}' if phase == 'train' else f'{MemProfResult.test_key}'
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

def def_results(device_type: str, precision: Union[int, str], dataset_type: str = "cust",
                ds_cfg: str = "train_prof"):
    match dataset_type:
        case "cust":
            test_dataset_state = test_dataset_state_core_cust
        case "gpt2":
            test_dataset_state = test_dataset_state_core_gpt2
        case "llama2":
            test_dataset_state = test_dataset_state_core_llama2
        case "tl":
            test_dataset_state = test_dataset_state_tl
        case _:
            raise ValueError(f"Unexpected dataset_type: {dataset_type}")
    test_dataset_state = test_dataset_state + expected_first_fwd_ids[ds_cfg]
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


########################################################################################################################
# Configuration composition
########################################################################################################################

# useful for manipulating segments of nested dictionaries (e.g. generating config file sets for CLI composition tests)
def set_nested(chained_keys: List | str, orig_dict: Optional[Dict] = None):
    orig_dict = {} if orig_dict is None else orig_dict
    chained_keys = chained_keys if isinstance(chained_keys, list) else chained_keys.split(".")
    reduce(lambda d, k: d.setdefault(k, {}), chained_keys, orig_dict)
    return orig_dict

def get_nested(target: Dict, chained_keys: List | str):
    chained_keys = chained_keys if isinstance(chained_keys, list) else chained_keys.split(".")
    return reduce(lambda d, k: d.get(k), chained_keys, target)

TEST_DATAMODULE_BASE_CONFIGS = {
    # TODO: make this dict a more robust registry if the number of tested models profilerates
    # TODO: pull module/datamodule configs from model-keyed test config dict (fake lightweight registry)
    # (dm_adapter_ctx, model_src_key)
    (Adapter.core, "gpt2"): (core_gpt2_shared_config, core_gpt2_datamodule_kwargs),
    (Adapter.core, "llama2"): (core_llama2_shared_config, core_llama2_datamodule_kwargs),
    (Adapter.core, "cust"): (core_cust_shared_config, core_cust_datamodule_kwargs),
    (Adapter.transformer_lens, "any"): (test_tl_gpt2_shared_config, test_tl_datamodule_kwargs),
}

TEST_MODULE_BASE_CONFIGS = {
    # (phase, adapter_mod_cfg_key, model_src_key)
    ("test", None, "gpt2"): test_core_gpt2_it_module_base,
    ("train", None, "gpt2"): test_core_gpt2_it_module_optim,
    ("test", None, "llama2"): test_core_llama2_it_module_base,
    ("train", None, "llama2"): test_core_llama2_it_module_optim,
    ("predict", None, "cust"): test_core_cust_it_module_base,
    ("test", None, "cust"): test_core_cust_it_module_base,
    ("train", None, "cust"): test_core_cust_it_module_optim,
    ("test", Adapter.transformer_lens, "gpt2"): test_tl_gpt2_it_module_base,
    ("train", Adapter.transformer_lens, "gpt2"): test_tl_gpt2_it_module_optim,
    ("test", Adapter.transformer_lens, "cust"): test_tl_cust_it_module_base,
    ("train", Adapter.transformer_lens, "cust"): test_tl_cust_it_module_optim,
}


MODULE_CONFIG_MAPPING = {
    # (model_key,  adapter_mod_cfg_key)
    ("rte", None): RTEBoolqConfig,
    ("rte", Adapter.transformer_lens): RTEBoolqTLConfig
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
    dm_base_cfg_key = (Adapter.transformer_lens, "any") if Adapter.transformer_lens in test_cfg.adapter_ctx else \
        (Adapter.core, test_cfg.model_src_key or "gpt2")
    shared_config, default_itdm_kwargs  = TEST_DATAMODULE_BASE_CONFIGS[dm_base_cfg_key]
    test_it_datamodule_cfg = deepcopy(default_itdm_kwargs)
    if dm_override_cfg:
        test_it_datamodule_cfg.update(dm_override_cfg)
    return ITDataModuleConfig(**shared_config, **test_it_datamodule_cfg)

def init_adapter_mod_cfg(test_cfg: Tuple, test_it_module_cfg: Dict):
    if Adapter.transformer_lens in test_cfg.adapter_ctx:
        if test_cfg.model_src_key == "gpt2":
            test_it_module_cfg['tl_cfg'] = ITLensFromPretrainedConfig(**test_it_module_cfg['tl_cfg'])
        elif test_cfg.model_src_key == "cust":
            test_it_module_cfg['tl_cfg'] = ITLensCustomConfig(**test_it_module_cfg['tl_cfg'])
        else:
            raise ValueError(f"Unknown model_src_key: {test_cfg.model_src_key}")
    else:  # See NOTE [Interpretability Adapters]
        raise ValueError(f"Unknown adapter type: {test_cfg.adapter_ctx}")

def get_it_cfg(test_cfg: Tuple, core_log_dir: Optional[StrOrPath] = None) -> ITConfig:
    test_cfg_override_attrs = ["memprofiler_cfg", "debug_lm_cfg", "cust_fwd_kwargs", "tl_cfg", "model_cfg",
                               "hf_from_pretrained_cfg", "zero_shot_cfg"]
    adapter_mod_cfg_key = Adapter.transformer_lens if Adapter.transformer_lens in test_cfg.adapter_ctx else None
    target_test_it_module_cfg = TEST_MODULE_BASE_CONFIGS[(test_cfg.phase, adapter_mod_cfg_key, test_cfg.model_src_key)]
    test_it_module_cfg = deepcopy(target_test_it_module_cfg)
    for attr in test_cfg_override_attrs:
        if getattr(test_cfg, attr):
            test_it_module_cfg.update({attr: getattr(test_cfg, attr)})
    if core_log_dir:
        test_it_module_cfg.update({'core_log_dir': core_log_dir})
    test_it_module_cfg = configure_device_precision(test_it_module_cfg, test_cfg.device_type, test_cfg.precision)
    if adapter_mod_cfg_key:
        init_adapter_mod_cfg(test_cfg, test_it_module_cfg)
    config_class = MODULE_CONFIG_MAPPING[(test_cfg.model_key, adapter_mod_cfg_key)]
    return config_class(**test_it_module_cfg)

def configure_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> Dict[str, Any]:
    # TODO: As we accommodate many different device/precision settting sources at the moment, it may make sense
    # to refactor hf and tl support via additional adapter functions and only test adherence to the
    # common Interpretune protocol here (testing the adapter functions separately with smaller unit tests)
    if cfg.get('model_cfg', None) is not None:
        cfg['model_cfg'].update({'dtype': get_model_input_dtype(precision), 'device': device_type})
    if cfg.get('hf_from_pretrained_cfg', None) is not None:
        cfg['hf_from_pretrained_cfg'].pretrained_kwargs.update({'torch_dtype': get_model_input_dtype(precision)})
        if device_type == "cuda":
            # note that with TLens this should be overridden by the tl adapter but we want to test that functionality
            cfg['hf_from_pretrained_cfg'].pretrained_kwargs.update({'device_map': 0})
    if cfg.get('tl_cfg', None) is not None:
        dev_prec_override = {'dtype': get_model_input_dtype(precision), 'device': device_type}
        if cfg['tl_cfg'].get('cfg', None) is not None:  # TL custom model config
            cfg['tl_cfg']['cfg'].update(dev_prec_override)
        else:  # TL from pretrained config, we set directly in addition to pretrained above to verify sync behavior
            cfg['tl_cfg'].update(dev_prec_override)
    return cfg

def config_session(core_cfg, test_cfg, test_alias, expected_results, state_log_dir, prewrapped_modules) \
    -> ITSessionConfig:
    session_ctx = {'adapter_ctx': test_cfg.adapter_ctx}
    dm_kwargs = {'dm_kwargs': {'force_prepare_data': test_cfg.force_prepare_data}}
    module_kwargs = {'module_kwargs': {'test_alias': test_alias, 'state_log_dir': state_log_dir, **expected_results}}
    session_cfg = ITSessionConfig(**core_cfg, **session_ctx, **dm_kwargs, **module_kwargs, **prewrapped_modules)
    return session_cfg

def gen_session_cfg(test_cfg, test_alias, expected_results, tmp_path, prewrapped_modules,
                    state_log_mode: bool = False) -> ITSessionConfig:
    itdm_cfg = get_itdm_cfg(test_cfg=test_cfg, dm_override_cfg=test_cfg.dm_override_cfg)
    it_cfg = get_it_cfg(test_cfg=test_cfg, core_log_dir=tmp_path)
    core_cfg = {'datamodule_cfg': itdm_cfg, 'module_cfg': it_cfg, **CORE_SESSION_CFG}
    if test_cfg.model_src_key not in DEFAULT_TEST_DATAMODULES:
        core_cfg['datamodule_cls'] = TEST_IT_DATAMODULE_MAPPING[test_cfg.model_src_key]
    state_log_dir = tmp_path if state_log_mode else None
    it_session_cfg = config_session(core_cfg, test_cfg, test_alias, expected_results, state_log_dir, prewrapped_modules)
    return it_session_cfg

def config_modules(test_cfg, test_alias, expected_results, tmp_path,
                   prewrapped_modules: Optional[Dict[str, Any]] = None, state_log_mode: bool = False) -> ITSession:
    if Adapter.lightning in test_cfg.adapter_ctx:  # allow Lightning to set env vars
        seed_everything(1, workers=True)
    cuda_reset()
    torch.set_printoptions(precision=12)
    prewrapped = prewrapped_modules or {}
    it_session_cfg = gen_session_cfg(test_cfg, test_alias, expected_results, tmp_path, prewrapped, state_log_mode)
    it_session = ITSession(it_session_cfg)
    return it_session

################################################################################
# CUDA utils
################################################################################

def _clear_cuda_memory() -> None:
    # strangely, the attribute function be undefined when torch.compile is used
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        # https://github.com/pytorch/pytorch/issues/95668
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()

def cuda_reset():
    if torch.cuda.is_available():
        _clear_cuda_memory()
        torch.cuda.reset_peak_memory_stats()
