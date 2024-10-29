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
import os
from typing import Optional, Any, Union, Dict
from copy import deepcopy

import torch

from interpretune.adapters.transformer_lens import ITLensFromPretrainedConfig, ITLensCustomConfig
from interpretune.adapters.registration import Adapter
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.base.contract.session import ITSessionConfig, ITSession
from interpretune.utils.types import StrOrPath
from tests import seed_everything
from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY
from tests.utils import get_model_input_dtype
from base_defaults import  BaseCfg


# # We use the same datamodule and module for all parity test contexts to ensure cross-adapter compatibility
# # Default test modules
# TEST_IT_DATAMODULE = TestITDataModule
# TEST_IT_MODULE = TestITModule
# CORE_SESSION_CFG = {'datamodule_cls': TEST_IT_DATAMODULE, 'module_cls': TEST_IT_MODULE}
# DEFAULT_TEST_DATAMODULES = ('cust', 'gpt2')

# # N.B. Some unit tests may require slightly modified/subclassed test datamodules to accommodate testing of
#  functionality
# # not compatible with the default GPT2-based test datamodule
# TEST_IT_DATAMODULE_MAPPING = {'llama3': LlamaTestITDataModule}

IT_GLOBAL_STATE_LOG_MODE = os.environ.get("IT_GLOBAL_STATE_LOG_MODE", "0") == "1"

# ################################################################################
# # Core test generation and encapsulation
# ################################################################################

# @dataclass(kw_only=True)
# class BaseAugTest:
#     alias: str
#     cfg: Optional[Tuple] = None
#     marks: Optional[Dict] = None  # test instance-specific marks
#     expected: Optional[Dict] = None
#     result_gen: Optional[Callable] = None
#     function_marks: Dict[str, Any] = field(default_factory=dict)  # marks applied at test function level

#     def __post_init__(self):
#         if self.expected is None and self.result_gen is not None:
#             assert callable(self.result_gen), "result_gen must be callable"
#             self.expected = self.result_gen(self.alias)
#         if self.cfg is None and self.cfg_gen is not None:
#             assert callable(self.cfg_gen), "cfg_gen must be callable"
#             self.cfg = self.cfg_gen(self.alias)
#         elif isinstance(self.cfg, Dict):
#             self.cfg = self.cfg[self.alias]
#         if self.marks or self.function_marks:
#             self.marks = self._get_marks(self.marks, self.function_marks)

#     def _get_marks(self, marks: Optional[Dict | str], function_marks: Dict) -> Optional[RunIf]:
#         # support RunIf aliases applied to function level
#         if marks:
#             if isinstance(marks, Dict):
#                 function_marks.update(marks)
#             elif isinstance(marks, str):
#                 function_marks.update(RUNIF_ALIASES[marks])
#             else:
#                 raise ValueError(f"Unexpected marks input type (should be Dict, str or None): {type(marks)}")
#         if function_marks:
#             return RunIf(**function_marks)

# def pytest_param_factory(test_configs: List[BaseAugTest], unpack: bool = True) -> List:
#     return [pytest.param(
#             config.alias,
#             *config.cfg if unpack else (config.cfg,),
#             id=config.alias,
#             marks=config.marks or tuple(),
#         )
#         for config in test_configs
#     ]

# @dataclass(kw_only=True)
# class BaseCfg:
#     phase: str = "train"
#     device_type: str = "cpu"
#     model_key: str = "rte"  # "real-model"-based acceptance/parity testing/profiling
#     precision: str | int = 32
#     adapter_ctx: Iterable[Adapter | str] = (Adapter.core,)
#     model_src_key: Optional[str] = None
#     limit_train_batches: Optional[int] = 1
#     limit_val_batches: Optional[int] = 1
#     limit_test_batches: Optional[int] = 1
#     dm_override_cfg: Optional[Dict] = None
#     zero_shot_cfg: Optional[ZeroShotClassificationConfig] = None
#     hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = None
#     memprofiler_cfg: Optional[MemProfilerCfg] = None
#     debug_lm_cfg: Optional[DebugLMConfig] = None
#     model_cfg: Optional[Dict] = None
#     tl_cfg: Optional[Dict] = None
#     sl_cfg: Optional[Dict] = None
#     max_epochs: Optional[int] = 1
#     cust_fwd_kwargs: Optional[Dict] = None
#     # used when adding a new test dataset or changing a test model to force re-caching of test datasets
#     force_prepare_data: bool = False  # TODO: make this settable via an env variable as well
#     max_steps: Optional[int] = None
#     save_checkpoints: bool = False

#     def __post_init__(self):
#         self.adapter_ctx = ADAPTER_REGISTRY.canonicalize_composition(self.adapter_ctx)
#         self.max_steps = self.max_steps or self.limit_train_batches


########################################################################################################################
# Configuration composition
########################################################################################################################

def apply_itdm_test_cfg(base_itdm_cfg: ITDataModuleConfig, test_cfg: BaseCfg, **kwargs) -> ITConfig:
    # TODO: NEXT: inspect relevant adapter_ctx used for a sampling of various tests and verify all existing combinations
    #       are properly handled after switching from current TEST_DATAMODULE_BASE_CONFIGS and TEST_MODULE_BASE_CONFIGS
    #       references to a single MODULE_EXAMPLE_REGISTRY.get call
    # dm_base_cfg_key = (Adapter.transformer_lens, "any") if Adapter.transformer_lens in test_cfg.adapter_ctx else \
    #     (Adapter.core, test_cfg.model_src_key or "gpt2")
    # # myconfigs = MODULE_EXAMPLE_REGISTRY.get(("gpt2", "rte", "test", (Adapter.core,)))
    # test_it_datamodule_cfg = deepcopy(TEST_DATAMODULE_BASE_CONFIGS[dm_base_cfg_key])
    test_itdm_cfg = deepcopy(base_itdm_cfg)
    if test_cfg.dm_override_cfg:
        test_itdm_cfg.__dict__.update(test_cfg.dm_override_cfg)
    return test_itdm_cfg

def apply_it_test_cfg(base_it_cfg: ITConfig, test_cfg: BaseCfg, core_log_dir: Optional[StrOrPath] = None) -> ITConfig:
    test_cfg_override_attrs = ["memprofiler_cfg", "debug_lm_cfg", "cust_fwd_kwargs", "tl_cfg", "model_cfg",
                               "hf_from_pretrained_cfg", "zero_shot_cfg"]
    # adapter_mod_cfg_key = Adapter.transformer_lens if Adapter.transformer_lens in test_cfg.adapter_ctx else None
    # target_test_it_module_cfg = TEST_MODULE_BASE_CONFIGS[(test_cfg.phase, adapter_mod_cfg_key,
    #                                                      test_cfg.model_src_key)]
    test_it_cfg = deepcopy(base_it_cfg)
    for attr in test_cfg_override_attrs:
        if getattr(test_cfg, attr):
            test_it_cfg.__dict__.update({attr: getattr(test_cfg, attr)})
    if core_log_dir:
        test_it_cfg.__dict__.update({'core_log_dir': core_log_dir})
    return configure_device_precision(test_it_cfg, test_cfg.device_type, test_cfg.precision)

def configure_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> Dict[str, Any]:
    # TODO: As we accommodate many different device/precision setting sources at the moment, it may make sense
    # to refactor hf and tl support via additional adapter functions and only test adherence to the
    # common Interpretune protocol here (testing the adapter functions separately with smaller unit tests)
    # if we've already initialized config, we need to manually update (`_torch_dtype`)
    if hasattr(cfg, '_torch_dtype'):
        cfg._torch_dtype = get_model_input_dtype(precision)
    if cfg.model_cfg:
        cfg.model_cfg.update({'dtype': get_model_input_dtype(precision), 'device': device_type})
    if cfg.hf_from_pretrained_cfg:
        cfg.hf_from_pretrained_cfg.pretrained_kwargs.update({'torch_dtype': get_model_input_dtype(precision)})
        if device_type == "cuda":
            # note that with TLens this should be overridden by the tl adapter but we want to test that functionality
            cfg.hf_from_pretrained_cfg.pretrained_kwargs.update({'device_map': 0})
    if getattr(cfg, 'tl_cfg', None) is not None:  # if we're using a TL subclass of ITConfig
        _update_tl_cfg_device_precision(cfg, device_type, precision)
    return cfg

def _update_tl_cfg_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> None:
        dev_prec_override = {'dtype': get_model_input_dtype(precision), 'device': device_type}
        if isinstance(cfg.tl_cfg, ITLensCustomConfig):  # initialized TL custom model config
            cfg.tl_cfg.cfg.__dict__.update(dev_prec_override)
        elif isinstance(cfg.tl_cfg, ITLensFromPretrainedConfig):
            # TL from pretrained config, we set directly in addition to pretrained above to verify sync behavior
            cfg.tl_cfg.__dict__.update(dev_prec_override)
        else:  # likely uninitialized TL custom model config, may want to remove this branch and check
            assert cfg.tl_cfg.get('cfg', None)
            cfg.tl_cfg['cfg'].update(dev_prec_override)

def config_session(core_cfg, test_cfg, test_alias, expected_results, state_log_dir, prewrapped_modules) \
    -> ITSessionConfig:
    session_ctx = {'adapter_ctx': test_cfg.adapter_ctx}
    dm_kwargs = {'dm_kwargs': {'force_prepare_data': test_cfg.force_prepare_data}}
    module_kwargs = {'module_kwargs': {'test_alias': test_alias, 'state_log_dir': state_log_dir, **expected_results}}
    session_cfg = ITSessionConfig(**core_cfg, **session_ctx, **dm_kwargs, **module_kwargs, **prewrapped_modules)
    return session_cfg

def gen_session_cfg(test_cfg, test_alias, expected_results, tmp_path, prewrapped_modules,
                    state_log_mode: bool = False) -> ITSessionConfig:
    #test_example_key = (test_cfg.model_src_key, test_cfg.model_key, test_cfg.phase, test_cfg.adapter_ctx)
    base_itdm_cfg, base_it_cfg, dm_cls, m_cls = MODULE_EXAMPLE_REGISTRY.get(test_cfg)
    itdm_cfg = apply_itdm_test_cfg(base_itdm_cfg=base_itdm_cfg, test_cfg=test_cfg)
    it_cfg = apply_it_test_cfg(base_it_cfg=base_it_cfg, test_cfg=test_cfg, core_log_dir=tmp_path)
    core_cfg = {'datamodule_cls': dm_cls, 'module_cls': m_cls, 'datamodule_cfg': itdm_cfg, 'module_cfg': it_cfg}
    # core_cfg = {'datamodule_cfg': itdm_cfg, 'module_cfg': it_cfg, **CORE_SESSION_CFG}
    # if test_cfg.model_src_key not in DEFAULT_TEST_DATAMODULES:
    #     core_cfg['datamodule_cls'] = TEST_IT_DATAMODULE_MAPPING[test_cfg.model_src_key]
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
