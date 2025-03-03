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

import pytest
import torch

from interpretune.base import _call_itmodule_hook, CoreHelperAttributes
from interpretune.utils import MisconfigurationException
from interpretune.protocol import LRSchedulerConfig, LRScheduler
from tests.warns import unmatched_warns


class TestClassCoreModule:

    def test_lr_scheduler_confs(self, get_it_module__core_cust__setup):
        core_cust_it_m = get_it_module__core_cust__setup
        _call_itmodule_hook(core_cust_it_m, hook_name="configure_optimizers",
                            hook_msg="initializing optimizers and schedulers",
                            connect_output=True)
        assert isinstance(core_cust_it_m.lr_scheduler_configs[0], LRSchedulerConfig)
        # validate property mapping
        assert isinstance(core_cust_it_m.lr_schedulers, LRScheduler)
        core_cust_it_m._it_state._it_lr_scheduler_configs += core_cust_it_m.lr_scheduler_configs
        assert len(core_cust_it_m.lr_schedulers) == len(core_cust_it_m.lr_scheduler_configs) == 2
        core_cust_it_m._it_state._it_lr_scheduler_configs = None
        assert core_cust_it_m.lr_schedulers is None

    def test_optim_conf(self, get_it_module__core_cust__setup):
        def mock_optim_confs(base_optim, base_lr_scheduler):
            """Optional because it is not mandatory in the context of core IT modules (required for some adapter
            modules)."""
            tuple_lists = base_optim, base_lr_scheduler
            single_optim = base_optim[0]
            single_dict = {"optimizer": base_optim, "lr_scheduler": base_lr_scheduler}
            multi_dict = {"optimizer": base_optim, "lr_scheduler": base_lr_scheduler}, {"optimizer": None}
            single_tuple = base_optim[0], deepcopy(base_optim[0])
            mon_warn = {"optimizer": base_optim, "monitor": "unsupp_val", "lr_scheduler": base_lr_scheduler}
            unsupported_cfg = {base_optim[0], base_optim[0]}  # test warning assuming set remains unsupported
            return tuple_lists, single_optim, single_dict, multi_dict, single_tuple, mon_warn, unsupported_cfg

        core_cust_it_m = get_it_module__core_cust__setup
        base_config = _call_itmodule_hook(core_cust_it_m, hook_name="configure_optimizers", hook_msg="optim conf test")
        *optim_confs, mon_warn, unsupp = mock_optim_confs(base_config[0], base_config[1])
        for optim_conf in optim_confs:
            core_cust_it_m._it_init_optimizers_and_schedulers(optim_conf)
        with pytest.warns(UserWarning, match="does not support `monitor`"):
            core_cust_it_m._it_init_optimizers_and_schedulers(mon_warn)
        with pytest.raises(MisconfigurationException):
            core_cust_it_m._it_init_optimizers_and_schedulers(unsupp)
        core_cust_it_m._it_init_optimizers_and_schedulers(None)  # validate graceful handling with empty config

    def test_property_dispatch_warns(self, recwarn, get_it_module__core_cust__setup):
        core_cust_it_m = get_it_module__core_cust__setup
        _call_itmodule_hook(core_cust_it_m, hook_name="setup",
                            hook_msg="warning on setup hook that does not support connected output",
                            connect_output=True)
        EXPECTED_PROP_WARNS = ("Could not find a device reference", "Output received for hook")
        core_cust_it_m.CORE_TO_FRAMEWORK_ATTRS_MAP["_current_epoch"] = ("foo.attr", 42, "FooAttr not set yet.")
        assert core_cust_it_m.current_epoch == 42  # validate unset c2f property mapping handling
        core_cust_it_m._it_state._device = None
        delattr(core_cust_it_m.model, 'device')
        assert core_cust_it_m.device is None  # validate device unset handling generates warning
        core_cust_it_m.it_cfg._torch_dtype = None
        assert core_cust_it_m.torch_dtype == torch.float32  # validate backup dtype introspection
        delattr(core_cust_it_m, 'it_cfg')  # ensure unexpected torch_dtype resolution failure is handled gracefully
        assert core_cust_it_m.torch_dtype is None
        w_expected = EXPECTED_PROP_WARNS
        if w_expected:
            unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=w_expected)
            assert not unmatched

    def test_degen_core_attributes(self):
        with pytest.raises(MisconfigurationException, match="CoreHelperAttributes requires an ITConfig"):
            _ = CoreHelperAttributes()

    def test_config_adapter(self, get_it_session__tl_cust__initonly):
        it_session = get_it_session__tl_cust__initonly
        curr_mod = it_session.module
        curr_config = curr_mod.it_cfg.tl_cfg.cfg
        curr_config = curr_mod._make_config_serializable(curr_config, 'dtype')
        assert isinstance(curr_config.dtype, str)
