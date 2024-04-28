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
from contextlib import contextmanager

import pytest

from tests.utils.warns import CORE_CTX_WARNS, unexpected_warns, unmatched_warns
from tests.orchestration import ablate_cls_attrs
from interpretune.base.contract.session import ITSession, ITMeta
from interpretune.base.contract.protocol import ITDataModuleProtocol, ITModuleProtocol


class TestClassContract:

    @contextmanager
    @staticmethod
    def invalid_mods_ctx(session_cfg, invalidate_dm: bool = True):
        try:
            dm_cls = session_cfg.datamodule_cls
            m_cls = session_cfg.module_cls
            missing_dm_funcs = ("train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader") \
                if invalidate_dm else ()
            missing_m_funcs = ("training_step", "validation_step", "test_step", "predict_step")
            with ablate_cls_attrs(dm_cls, missing_dm_funcs), ablate_cls_attrs(m_cls, missing_m_funcs):
                yield
        finally:
            pass

    def test_it_session_warns(self, recwarn, get_tl_it_session_cfg):
        expected_warns = (".*TestITDataModule.* is not", ".*TestITModule.* is not")
        with TestClassContract.invalid_mods_ctx(get_tl_it_session_cfg):
            _ = ITSession(get_tl_it_session_cfg)
        unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=expected_warns)
        assert not unmatched

    def test_it_session_validation_errors(self, recwarn, get_tl_it_session_cfg):
        sess_cfg = get_tl_it_session_cfg
        build_ctx = {'framework_ctx': sess_cfg.framework_ctx, 'plugin_ctx': sess_cfg.plugin_ctx}
        m_cls = ITMeta('InterpretunableModule', (), {}, component='m', input=sess_cfg.module_cls, ctx=build_ctx)
        with TestClassContract.invalid_mods_ctx(sess_cfg, invalidate_dm=False):
            with pytest.raises(ValueError, match="must be provided"):
                dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm', input=sess_cfg.datamodule_cls)
            with pytest.raises(ValueError, match="Specified component was"):
                dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='oops', input=sess_cfg.datamodule_cls,
                                ctx=build_ctx)
            with pytest.raises(ValueError, match="is not a callable"):
                dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm', input="oops", ctx=build_ctx)
            with pytest.raises(ValueError, match="desired class enrichment"):
                dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm', input=sess_cfg.datamodule_cls,
                                ctx="oops")
            dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm', input=sess_cfg.datamodule_cls,
                                ctx=build_ctx)
            tmp_dm = dm_cls(itdm_cfg=sess_cfg.datamodule_cfg, *sess_cfg.dm_args, **sess_cfg.dm_kwargs)
            # if we are manually preparing this particular module, we need to manually implement the attribute handle
            # sharing that session auto-composition would provide us via `ITSession._set_dm_handles_for_instantiation`
            with patch.object(sess_cfg.module_cfg, 'tokenizer', new=getattr(tmp_dm, 'tokenizer')):
                ready_inval_m = m_cls(it_cfg=sess_cfg.module_cfg, *sess_cfg.module_args, **sess_cfg.module_kwargs)
            with patch.object(sess_cfg, 'module', new=ready_inval_m):
                with pytest.raises(ValueError, match="enable auto-composition"):
                    _ = ITSession(sess_cfg)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=CORE_CTX_WARNS)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    @pytest.mark.parametrize(
        "dm_precomposed, m_precomposed",
        [pytest.param(True, True), pytest.param(True, False), pytest.param(False, True)],
        ids=["both_dm_m_precomposed", "dm_precomposed", "m_precomposed"],
    )
    def test_it_session_precomposed(self, recwarn, get_core_cust_it_session_cfg, dm_precomposed, m_precomposed):
        expected_warns = (*CORE_CTX_WARNS, "Since no datamodule.*")
        sess_cfg = get_core_cust_it_session_cfg
        build_ctx = {'framework_ctx': sess_cfg.framework_ctx, 'plugin_ctx': sess_cfg.plugin_ctx}
        if dm_precomposed:
            dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm', input=sess_cfg.datamodule_cls,
                                    ctx=build_ctx)
            ready_dm = dm_cls(itdm_cfg=sess_cfg.datamodule_cfg, *sess_cfg.dm_args, **sess_cfg.dm_kwargs)
        if m_precomposed:
            m_cls = ITMeta('InterpretunableModule', (), {}, component='m', input=sess_cfg.module_cls, ctx=build_ctx)
            ready_m = m_cls(it_cfg=sess_cfg.module_cfg, *sess_cfg.module_args, **sess_cfg.module_kwargs)
        if dm_precomposed and m_precomposed:
            # we patch *_cls attributes to `None` so we can reuse the same session cfg object across multiple tests
            with patch.object(sess_cfg, 'datamodule', new=ready_dm), patch.object(sess_cfg, 'module', new=ready_m), \
                 patch.object(sess_cfg, 'datamodule_cls', None), patch.object(sess_cfg, 'module_cls', None):
                it_session = ITSession(sess_cfg)
        elif dm_precomposed:
            with patch.object(sess_cfg, 'datamodule', new=ready_dm), patch.object(sess_cfg, 'datamodule_cls', None):
                it_session = ITSession(sess_cfg)
        elif m_precomposed:
            with patch.object(sess_cfg, 'module', new=ready_m), patch.object(sess_cfg, 'module_cls', None):
                it_session = ITSession(sess_cfg)
        assert isinstance(it_session.datamodule, ITDataModuleProtocol)
        assert isinstance(it_session.module, ITModuleProtocol)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warns)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_it_session_cfg_sanitize(self, get_tl_it_session_cfg):
        it_session = ITSession(get_tl_it_session_cfg)
        assert len(it_session) == 2
        assert repr(it_session.module) == 'InterpretunableModule(TestITModule)'
        assert repr(it_session.datamodule) == 'InterpretunableDataModule(TestITDataModule)'

    def test_session_min_dep_installed(self):
        import sys
        from importlib import reload
        del sys.modules['interpretune.base.contract.session']
        with patch('interpretune.utils.import_utils._LIGHTNING_AVAILABLE', False):
            from interpretune.base.contract.session import LightningModule, LightningDataModule
            assert LightningModule.__module__ == 'builtins'
            assert LightningDataModule.__module__ == 'builtins'
        reload(sys.modules['interpretune.base.contract.session'])
        from interpretune.base.contract.session import LightningModule, LightningDataModule
        assert LightningModule.__module__ == 'lightning.pytorch.core.module'
        assert LightningDataModule.__module__ == 'lightning.pytorch.core.datamodule'
