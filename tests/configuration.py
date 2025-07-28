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
from datasets import Dataset
from sae_lens import SAE
from transformers import BatchEncoding

from interpretune.config import (ITConfig, ITDataModuleConfig, AnalysisCfg, SAELensFromPretrainedConfig,
                                 SAELensCustomConfig, ITLensFromPretrainedConfig, ITLensCustomConfig)
from interpretune.session import ITSessionConfig, ITSession
from interpretune.protocol import StrOrPath, Adapter
from interpretune.analysis import AnalysisBatch, AnalysisStore
from interpretune.runners.analysis import maybe_init_analysis_cfg
from tests import seed_everything
from tests.base_defaults import BaseCfg
from tests.data_generation import gen_or_validate_input_data
from tests.utils import get_model_input_dtype, cuda_reset
from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY


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
    # TODO: for attributes that don't actually belong to ITConfig (and existing subclasses), we should avoid adding them
    # e.g. right now, `sae_analysis_targets` is the only one that doesn't belong to ITConfig or defined subclasses
    test_cfg_override_attrs = ["memprofiler_cfg", "debug_lm_cfg", "cust_fwd_kwargs", "tl_cfg", "model_cfg", "sae_cfgs",
                               "hf_from_pretrained_cfg", "generative_step_cfg", "add_saes_on_init", "auto_comp_cfg",
                               "sae_analysis_targets", "analysis_cfgs", "sae_cfgs"]
    test_it_cfg = deepcopy(base_it_cfg)
    for attr in test_cfg_override_attrs:
        if hasattr(test_cfg, attr) and getattr(test_cfg, attr) is not None:
            test_it_cfg.__dict__.update({attr: getattr(test_cfg, attr)})
    if core_log_dir:
        test_it_cfg.__dict__.update({'core_log_dir': core_log_dir})
    it_cfg = configure_device_precision(test_it_cfg, test_cfg.device_type, test_cfg.precision)
    it_cfg.__post_init__()  # re-execute post-init logic since we may have changed the constituent cfgs manually
    return it_cfg

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

def cfg_op_env(request, session_fixture, op_to_test, deepcopy_session_fixt: bool, input_data=None, batches=1,
               generate_required_only: bool = True, override_req_cols: Optional[tuple] = None, ) -> tuple:
    """Set up a test environment for an operation using a real model.

    Args:
        request: pytest request fixture
        session_fixture: name of the session fixture to use
        op_to_test: the operation to test (e.g., it.model_forward)
        input_data: dictionary of input data needed for the op's input_schema
        batches: number of batches to process (default: 1)
        override_req_cols: Tuple of field names to override the required_only behavior
                          (see gen_or_validate_input_data docstring)

    Returns:
        If batches=1: tuple of (it_session, batch, analysis_batch)
        If batches>1: tuple of (it_session, [batch1, batch2, ...], [analysis_batch1, analysis_batch2, ...])
    """
    # Get the fixture and extract session
    fixture = request.getfixturevalue(session_fixture)
    it_session = get_deepcopied_session(fixture.it_session) if deepcopy_session_fixt else fixture.it_session

    # Configure the analysis
    analysis_cfg = AnalysisCfg(
        target_op=op_to_test,
        ignore_manual=True,
        save_tokens=True,
        sae_analysis_targets=fixture.test_cfg().sae_analysis_targets,
    )

    # Initialize analysis config on the module
    maybe_init_analysis_cfg(it_session.module, analysis_cfg)

    # Extract input schema
    input_schema = getattr(analysis_cfg.op, 'input_schema', None)

    # Get a sample batch to extract shapes
    batch_shapes = {}
    if input_schema:
        # Create test dataloader early to inspect shapes
        dataloader = it_session.datamodule.test_dataloader()
        sample_batch = next(iter(dataloader))

        # Extract shape information from the batch
        if isinstance(sample_batch, (BatchEncoding, dict)):
            for key, value in sample_batch.items():
                if hasattr(value, 'shape'):
                    batch_shapes[f'{key}_shape'] = value.shape

            # Calculate actual sequence length if input_ids are present
            # Determine the dynamic sequence length key dynamically
            model_input_name = it_session.datamodule.tokenizer.model_input_names[0]
            if model_input_name in sample_batch:
                target_seq_len = sample_batch[model_input_name].shape[-1]
                batch_shapes['target_seq_len'] = target_seq_len

        # Build features spec for inputs and handle multiple batches
        regular_data, intermediate_data = gen_or_validate_input_data(
            module=it_session.module,
            input_schema=input_schema,
            batch_shapes=batch_shapes,
            input_data=input_data,
            num_batches=batches,
            required_only=generate_required_only,
            override_req_cols=override_req_cols,
            predefined_indices=True  # Always use predefined indices for testing
        )
    else:
        regular_data, intermediate_data = {}, {}

    # Create the analysis store with regular input fields only
    input_store = None
    if regular_data:
        schema_source = it_session.module.analysis_cfg.op.input_schema
        serializable_col_cfg = {k: v.to_dict() for k, v in schema_source.items()} if schema_source else {}
        it_format_kwargs = dict(col_cfg=serializable_col_cfg)
        dataset = Dataset.from_dict(regular_data).with_format("interpretune", **it_format_kwargs)
        input_store = AnalysisStore(dataset=dataset, it_format_kwargs=it_format_kwargs)

    analysis_cfg.input_store = input_store

    # Reset dataloader iterator to start from the beginning
    dataloader = it_session.datamodule.test_dataloader()
    dataloader_iter = iter(dataloader)

    # If only one batch is requested (original behavior)
    if batches == 1:
        batch = next(dataloader_iter)

        if input_store is not None:
            analysis_batch = AnalysisBatch(input_store[0])
        else:
            analysis_batch = AnalysisBatch()

        # Manually add intermediate_only values to the analysis batch
        for field, values in intermediate_data.items():
            if values and len(values) > 0:
                setattr(analysis_batch, field, values[0])

        return it_session, batch, analysis_batch

    # If multiple batches are requested
    else:
        batch_list = []
        analysis_batch_list = []

        for i in range(batches):
            try:
                batch = next(dataloader_iter)
                batch_list.append(batch)

                if input_store is not None and i < len(input_store.dataset):
                    analysis_batch = AnalysisBatch(input_store[i])
                else:
                    analysis_batch = AnalysisBatch()

                # Manually add intermediate_only values to the analysis batch
                for field, values in intermediate_data.items():
                    if values and i < len(values):
                        setattr(analysis_batch, field, values[i])

                analysis_batch_list.append(analysis_batch)
            except StopIteration:
                # If we run out of data, break the loop
                break

        return it_session, batch_list, analysis_batch_list

def config_modules(test_cfg, test_alias, expected_results, tmp_path,
                   prewrapped_modules: Optional[Dict[str, Any]] = None, state_log_mode: bool = False,
                   cfg_only: bool = False) -> ITSessionConfig | ITSession:
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

def get_deepcopied_session(it_session_fixt: ITSession):
    """Deepcopy the provided ITSession fixture with potential special handling for certain attributes. This is
    necessary in some contexts to avoid issues with shared references in the original session.

    Args:
        it_session_fixt: The ITSession fixture to be deepcopied.

    Returns:
        A deepcopy of the provided ITSession fixture patched to handle edge cases.
    """
    it_session = deepcopy(it_session_fixt)

    # TODO: since there are other method-bound closures that we don't patch in it_session encapsulated objects, we
    # should probably refactor our it_session fixture strategy and possibly create a custom __deepcopy__ method for
    # it_session that handles all of these cases.
    # Note: Currently, the only context that requires special handling is when SAEs have already been instantiated.
    # This modification should be removed once the SAELens code is refactored to avoid this issue.
    sae_handles = getattr(it_session.module, "sae_handles", None)
    # as `reshape_fn_in` is considered an immutable, atomic object by deepcopy, we need to recreate the relevant
    # method-bound closure `turn_on_forward_pass_hook_z_reshaping` creates to rebind it to our new SAE instance(s)
    if sae_handles and any(isinstance(handle, SAE) for handle in sae_handles):
        for handle in sae_handles:
            if handle.cfg.metadata.hook_name.endswith("_z"):
                handle.turn_on_forward_pass_hook_z_reshaping()
            else:
                handle.turn_off_forward_pass_hook_z_reshaping()
    return it_session
