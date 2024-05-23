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
from unittest.mock import patch
import pytest
from copy import deepcopy

from interpretune.adapters.core import ITModule
from interpretune.base.call import _call_itmodule_hook
from tests.orchestration import ablate_cls_attrs
from interpretune.base.contract.session import ITSession
from tests.utils.warns import CORE_CTX_WARNS, unexpected_warns


class TestClassBaseMisc:

    def test_dm_attribute_access(self, get_it_session__core_cust__setup):
        core_cust_it_dm = get_it_session__core_cust__setup.datamodule
        assert isinstance(core_cust_it_dm.module, ITModule)
        with ablate_cls_attrs(core_cust_it_dm, "_module"), pytest.warns(UserWarning, match="Could not find module"):
            assert core_cust_it_dm.module is None
        with pytest.warns(UserWarning, match="Output received for hook"):
            _call_itmodule_hook(core_cust_it_dm, hook_name="setup",
                                hook_msg="warning on setup hook that does not support connected output",
                                connect_output=True)

    @pytest.mark.parametrize("remove_unused", [True, False], ids=['remove_unused', 'no_remove_unused'])
    def test_dm_force_prepare(self, get_it_session__core_cust_force_prepare__initonly, remove_unused):
        datamodule = get_it_session__core_cust_force_prepare__initonly.datamodule
        with patch.multiple(datamodule.itdm_cfg, signature_columns=[], remove_unused_columns=remove_unused):
            _call_itmodule_hook(datamodule, hook_name="prepare_data", hook_msg="Preparing data",
                                target_model=get_it_session__core_cust_force_prepare__initonly.module.model)


    def test_m_no_before_it_cfg_init(self, recwarn, get_core_cust_it_session_cfg):
        sess_cfg = deepcopy(get_core_cust_it_session_cfg)
        # we ablate _before_it_cfg_init twich to walk up the object hierarchy to the default BaseITModule version
        with ablate_cls_attrs(sess_cfg.module_cls, '_before_it_cfg_init'), \
            ablate_cls_attrs(sess_cfg.module_cls, '_before_it_cfg_init'):
            _ = ITSession(sess_cfg)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=CORE_CTX_WARNS)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
