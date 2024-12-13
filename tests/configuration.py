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
from typing import Optional, Any, Union, Dict, Sequence
from copy import deepcopy

import torch

from interpretune.adapters.transformer_lens import ITLensFromPretrainedConfig, ITLensCustomConfig
from interpretune.adapters.sae_lens import SAELensFromPretrainedConfig, SAELensCustomConfig
from interpretune.base.config.shared import Adapter
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.base.contract.session import ITSessionConfig, ITSession
from interpretune.utils.types import StrOrPath
from tests import seed_everything
from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY
from tests.utils import get_model_input_dtype
from base_defaults import  BaseCfg

IT_GLOBAL_STATE_LOG_MODE = os.environ.get("IT_GLOBAL_STATE_LOG_MODE", "0") == "1"

########################################################################################################################
# Configuration composition
########################################################################################################################

def apply_itdm_test_cfg(base_itdm_cfg: ITDataModuleConfig, test_cfg: BaseCfg, **kwargs) -> ITConfig:
    test_itdm_cfg = deepcopy(base_itdm_cfg)
    if test_cfg.dm_override_cfg:
        test_itdm_cfg.__dict__.update(test_cfg.dm_override_cfg)
    return test_itdm_cfg

def apply_it_test_cfg(base_it_cfg: ITConfig, test_cfg: BaseCfg, core_log_dir: Optional[StrOrPath] = None) -> ITConfig:
    test_cfg_override_attrs = ["memprofiler_cfg", "debug_lm_cfg", "cust_fwd_kwargs", "tl_cfg", "model_cfg", "sae_cfgs",
                               "hf_from_pretrained_cfg", "generative_step_cfg", "add_saes_on_init"]
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
        if getattr(cfg, 'sae_cfgs', None) is not None:  # if we're using a SL subclass of ITConfig, also requires TL
            if not isinstance(cfg.sae_cfgs, Sequence):
                cfg.sae_cfgs = [cfg.sae_cfgs]
            for sae_cfg in cfg.sae_cfgs:
                _update_sae_cfg_device_precision(sae_cfg, device_type, precision)
    return cfg

def _update_tl_cfg_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> None:
        dev_prec_override = {'dtype': get_model_input_dtype(precision), 'device': device_type}
        if isinstance(cfg.tl_cfg, ITLensCustomConfig):  # initialized TL custom model config
            cfg.tl_cfg.cfg.__dict__.update(dev_prec_override)
        elif isinstance(cfg.tl_cfg, ITLensFromPretrainedConfig):
            # TL from pretrained config, we set directly in addition to pretrained above to verify sync behavior
            cfg.tl_cfg.__dict__.update(dev_prec_override)
        else:  # likely uninitialized TL custom model config, may want to remove this branch/check
            assert cfg.tl_cfg.get('cfg', None)

def _update_sae_cfg_device_precision(sae_cfg: SAELensCustomConfig | SAELensFromPretrainedConfig, device_type: str,
                                     precision: Union[int, str]) -> None:
    dev_prec_override = {'dtype': precision, 'device': device_type}  # SAEConfig currently requires strings
    if isinstance(sae_cfg, SAELensCustomConfig):
        sae_cfg.cfg.__dict__.update(dev_prec_override)
    elif isinstance(sae_cfg, SAELensFromPretrainedConfig):
        # TODO: SAE.from_pretrained() ctor doesn't support dtype a the moment. It could be added for specific loaders
        # via cfg_overrides dict read from config_overrides defined in ``pretrained_saes.yaml```
        dev_prec_override.pop('dtype')
        sae_cfg.__dict__.update(dev_prec_override)
    else: # likely uninitialized SL custom model config, may want to remove this branch/check
        assert sae_cfg.get('cfg', None)
        sae_cfg['cfg'].update(dev_prec_override)

def config_session(core_cfg, test_cfg, test_alias, expected_results, state_log_dir, prewrapped_modules) \
    -> ITSessionConfig:
    session_ctx = {'adapter_ctx': test_cfg.adapter_ctx}
    dm_kwargs = {'dm_kwargs': {'force_prepare_data': test_cfg.force_prepare_data}}
    module_kwargs = {'module_kwargs': {'test_alias': test_alias, 'state_log_dir': state_log_dir,
                                       'req_grad_mask': test_cfg.req_grad_mask, **expected_results}}
    session_cfg = ITSessionConfig(**core_cfg, **session_ctx, **dm_kwargs, **module_kwargs, **prewrapped_modules)
    return session_cfg

def gen_session_cfg(test_cfg, test_alias, expected_results, tmp_path, prewrapped_modules,
                    state_log_mode: bool = False) -> ITSessionConfig:
    base_itdm_cfg, base_it_cfg, dm_cls, m_cls = MODULE_EXAMPLE_REGISTRY.get(test_cfg)
    itdm_cfg = apply_itdm_test_cfg(base_itdm_cfg=base_itdm_cfg, test_cfg=test_cfg)
    it_cfg = apply_it_test_cfg(base_it_cfg=base_it_cfg, test_cfg=test_cfg, core_log_dir=tmp_path)
    core_cfg = {'datamodule_cls': dm_cls, 'module_cls': m_cls, 'datamodule_cfg': itdm_cfg, 'module_cfg': it_cfg}
    state_log_dir = tmp_path if state_log_mode else None
    it_session_cfg = config_session(core_cfg, test_cfg, test_alias, expected_results, state_log_dir, prewrapped_modules)
    return it_session_cfg

def config_modules(test_cfg, test_alias, expected_results, tmp_path,
                   prewrapped_modules: Optional[Dict[str, Any]] = None, state_log_mode: bool = False,
                   cfg_only: bool = False) -> ITSession:
    if Adapter.lightning in test_cfg.adapter_ctx:  # allow Lightning to set env vars
        seed_everything(1, workers=True)
    cuda_reset()
    torch.set_printoptions(precision=12)
    prewrapped = prewrapped_modules or {}
    it_session_cfg = gen_session_cfg(test_cfg, test_alias, expected_results, tmp_path, prewrapped, state_log_mode)
    if cfg_only:
        return it_session_cfg
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
