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
from copy import deepcopy
from unittest.mock import patch
import operator
import os
import sys
from typing import Any

import pytest
import torch

from tests.runif import RunIf
from tests.utils import ablate_cls_attrs
from tests.warns import CORE_CTX_WARNS, unexpected_warns, unmatched_warns
from interpretune.session import ITSession
from interpretune.utils import (
    resolve_funcs, _get_rank, rank_zero_only, rank_zero_deprecation, instantiate_class, _resolve_torch_dtype,
    package_available,  MisconfigurationException, move_data_to_device, to_device, module_available, compare_version)
from interpretune.config import ITExtension, SessionRunnerCfg
from interpretune.extensions import MemProfilerHooks, DefaultMemHooks


class TestClassUtils:

    @RunIf(min_cuda_gpus=1)
    @pytest.mark.parametrize(
        "w_expected",
        ["Unable to patch `get_cud.*", None],
        ids=["cuda_loading_warn_unpatched", "cuda_loading_patched"],
    )
    def test_get_cuda_loading_patch(self, recwarn, get_it_session_cfg__core_cust, w_expected):
        sess_cfg = deepcopy(get_it_session_cfg__core_cust)
        orig_patch_mgr = None
        if w_expected:
            orig_patch_mgr = sys.modules['interpretune.utils.logging'].__dict__.pop('patch_torch_env_logging_fn')
        it_session = ITSession(sess_cfg)
        if w_expected:
            sys.modules['interpretune.utils.logging'].__dict__['patch_torch_env_logging_fn'] = orig_patch_mgr
        collected_cuda_loading_config = it_session.module._it_state._init_hparams['env_info']['cuda_module_loading']
        assert isinstance(collected_cuda_loading_config, str)
        if w_expected:
            assert collected_cuda_loading_config != "not inspected"
            unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=w_expected)
            assert not unmatched
        else:
            assert collected_cuda_loading_config == "not inspected"
            unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=CORE_CTX_WARNS)
            assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_rank_zero_utils(self):
        with patch.dict(os.environ, {"RANK": "42"}):
            test_rank = _get_rank()
            assert test_rank == 42
        with patch.object(rank_zero_only, "rank", "13"):
            def rank_zero_default(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
                pass
            rank_zero_default = rank_zero_only(rank_zero_default, default="test success")
            assert rank_zero_default() == "test success"
        with patch.object(rank_zero_only, "rank", None):
            @rank_zero_only
            def rank_zero_errtest(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
                pass
            with pytest.raises(RuntimeError, match="needs to be set before use"):
                rank_zero_errtest()
        with pytest.warns(DeprecationWarning, match="Test deprecation msg"):
            rank_zero_deprecation("Test deprecation msg.")

    def test_fn_instantiate_class(self):
        short_circuit_path = "ITExtension"
        ext_init = {"class_path": short_circuit_path, "init_args": {'ext_attr': 'test_ext', 'ext_cls_fqn': 'some.loc',
                                                                    'ext_cfg_fqn': 'another.loc'}}
        sys.modules['interpretune.utils.import_utils'].instantiate_class.__globals__[short_circuit_path] = ITExtension
        test_ext = instantiate_class(init=ext_init)
        sys.modules['interpretune.utils.import_utils'].instantiate_class.__globals__.pop(short_circuit_path)
        assert test_ext.ext_cls_fqn == 'some.loc'
        del ext_init['class_path']
        with pytest.raises(MisconfigurationException, match="A class_path was not included"):
            test_ext = instantiate_class(init=ext_init)

    def test_fn_resolve_funcs(self):
        memory_hooks_cfg = MemProfilerHooks(pre_forward_hooks=DefaultMemHooks.pre_forward.value,
                                            post_forward_hooks=[], reset_state_hooks=[])
        resolved_single_hook = resolve_funcs(cfg_obj=memory_hooks_cfg, func_type='pre_forward_hooks')
        assert callable(resolved_single_hook[0])
        memory_hooks_cfg = MemProfilerHooks(pre_forward_hooks='interpretune.utils.warnings.unexpected_state_msg_suffix',
                                            post_forward_hooks=[], reset_state_hooks=[])
        with pytest.raises(MisconfigurationException, match="is not callable"):
            resolved_single_hook = resolve_funcs(cfg_obj=memory_hooks_cfg, func_type='pre_forward_hooks')
        memory_hooks_cfg = MemProfilerHooks(pre_forward_hooks='notfound.analysis.memprofiler._hook_npp_pre_forward',
                                            post_forward_hooks=[], reset_state_hooks=[])
        with pytest.raises(MisconfigurationException, match="Unable to import and resolve specified function"):
            resolved_single_hook = resolve_funcs(cfg_obj=memory_hooks_cfg, func_type='pre_forward_hooks')

    def test_fn_resolve_dtype(self):
        resolved_dtype = _resolve_torch_dtype(dtype="torch.float32")
        assert isinstance(resolved_dtype, torch.dtype)

    def test_package_module_available(self):
        assert not package_available("notgoingtofind.this.package")
        assert not module_available("missingtoplevelpackage.analysis.memprofiler")
        assert not module_available("interpretune.extensions.missingmodule")

    def test_compare_version(self):
        assert not compare_version("torchnotfound", operator.ge, "2.2.0", use_base_version=True)
        with ablate_cls_attrs(torch, "__version__"):
            assert compare_version("torch", operator.ge, "2.0.0", use_base_version=True)
        with patch.object(torch, "__version__", lambda x: x + 1):  # allow TypeError for Sphinx mocks
            assert compare_version("torch", operator.ge, "2.0.0", use_base_version=True)

    @RunIf(min_cuda_gpus=1)
    def test_to_device(self):
        mod = torch.nn.Linear(4, 8)
        mod = to_device("cuda", mod)
        assert mod.bias.device.type == "cuda"

    @RunIf(min_cuda_gpus=1)
    def test_move_data_to_device(self):
        batch = torch.ones([2, 3])
        batch = move_data_to_device(batch, "cuda")
        assert batch.device.type == "cuda"
        orig_to = torch._tensor.Tensor.to
        def degen_to(aten, *args, **kwargs):
            aten = orig_to(aten, *args, **kwargs)
            assert aten.device.type == 'cpu'  # forget to return self
        with patch('torch._tensor.Tensor.to', degen_to):
            batch = move_data_to_device(batch, "cpu")
        assert batch.device.type == "cuda"  # still on cuda because of improperly implemented `to`

    def test_basic_trainer_warns(self, get_it_session__core_cust__setup):
        test_cfg = get_it_session__core_cust__setup.fixt_test_cfg()
        test_cfg_overrides = {k: v for k,v in test_cfg.__dict__.items() if k in SessionRunnerCfg.__dict__.keys()}
        with pytest.raises(MisconfigurationException, match="If not providing `it_session`"):
            _ = SessionRunnerCfg(module=get_it_session__core_cust__setup.module, datamodule=None, **test_cfg_overrides)
        assert test_cfg
        with pytest.warns(UserWarning, match="should only be specified if not providing `it_session`"):
            trainer_config = SessionRunnerCfg(module=get_it_session__core_cust__setup.module,
                                it_session=get_it_session__core_cust__setup, **test_cfg_overrides)
        assert trainer_config
