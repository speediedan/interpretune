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
from copy import deepcopy

import pytest

from interpretune.config import ITSharedConfig
from interpretune.session import ITSession, ITMeta, ITSessionConfig
from interpretune.protocol import ITDataModuleProtocol, ITModuleProtocol
from tests.warns import CORE_CTX_WARNS, unexpected_warns, unmatched_warns
from tests.utils import ablate_cls_attrs, platform_normalize_str


class TestClassContract:
    @contextmanager
    @staticmethod
    def invalid_mods_ctx(session_cfg, invalidate_dm: bool = True):
        try:
            dm_cls = session_cfg.datamodule_cls
            m_cls = session_cfg.module_cls
            missing_dm_funcs = (
                ("train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader") if invalidate_dm else ()
            )
            missing_m_funcs = ("training_step", "validation_step", "test_step", "predict_step")
            with ablate_cls_attrs(dm_cls, missing_dm_funcs), ablate_cls_attrs(m_cls, missing_m_funcs):
                yield
        finally:
            pass

    def test_it_session_warns(self, recwarn, get_it_session_cfg__tl_cust):
        expected_warns = (".*TestITDataModule.* is not", ".*TestITModule.* is not")
        with TestClassContract.invalid_mods_ctx(get_it_session_cfg__tl_cust):
            _ = ITSession(get_it_session_cfg__tl_cust)
        unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=expected_warns)
        assert not unmatched

    def test_it_session_validation_errors(self, recwarn, get_it_session_cfg__tl_cust):
        sess_cfg = deepcopy(get_it_session_cfg__tl_cust)
        m_cls = ITMeta(
            "InterpretunableModule", (), {}, component="m", input=sess_cfg.module_cls, ctx=sess_cfg.adapter_ctx
        )
        with TestClassContract.invalid_mods_ctx(sess_cfg, invalidate_dm=False):
            with pytest.raises(ValueError, match="must be provided"):
                dm_cls = ITMeta("InterpretunableDataModule", (), {}, component="dm", input=sess_cfg.datamodule_cls)
            with pytest.raises(ValueError, match="Specified component was"):
                dm_cls = ITMeta(
                    "InterpretunableDataModule",
                    (),
                    {},
                    component="oops",
                    input=sess_cfg.datamodule_cls,
                    ctx=sess_cfg.adapter_ctx,
                )
            with pytest.raises(ValueError, match="is not a callable"):
                dm_cls = ITMeta(
                    "InterpretunableDataModule", (), {}, component="dm", input="oops", ctx=sess_cfg.adapter_ctx
                )
            with pytest.raises(ValueError, match="desired class enrichment"):
                dm_cls = ITMeta(
                    "InterpretunableDataModule", (), {}, component="dm", input=sess_cfg.datamodule_cls, ctx="oops"
                )
            dm_cls = ITMeta(
                "InterpretunableDataModule",
                (),
                {},
                component="dm",
                input=sess_cfg.datamodule_cls,
                ctx=sess_cfg.adapter_ctx,
            )
            tmp_dm = dm_cls(itdm_cfg=sess_cfg.datamodule_cfg, *sess_cfg.dm_args, **sess_cfg.dm_kwargs)
            # if we are manually preparing this particular module, we need to manually implement the attribute handle
            # sharing that session auto-composition would provide us via `ITSession._set_dm_handles_for_instantiation`
            with patch.object(sess_cfg.module_cfg, "tokenizer", new=getattr(tmp_dm, "tokenizer")):
                ready_inval_m = m_cls(it_cfg=sess_cfg.module_cfg, *sess_cfg.module_args, **sess_cfg.module_kwargs)
            with patch.object(sess_cfg, "module", new=ready_inval_m):
                with pytest.raises(ValueError, match="enable auto-composition"):
                    _ = ITSession(sess_cfg)
        expected_warns = (*CORE_CTX_WARNS, "Since no datamodule.*")
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warns)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    @pytest.mark.parametrize(
        "dm_precomposed, m_precomposed",
        [pytest.param(True, True), pytest.param(True, False), pytest.param(False, True)],
        ids=["both_dm_m_precomposed", "dm_precomposed", "m_precomposed"],
    )
    def test_it_session_precomposed(self, recwarn, get_it_session_cfg__core_cust, dm_precomposed, m_precomposed):
        sess_cfg = deepcopy(get_it_session_cfg__core_cust)
        if dm_precomposed:
            dm_cls = ITMeta(
                "InterpretunableDataModule",
                (),
                {},
                component="dm",
                input=sess_cfg.datamodule_cls,
                ctx=sess_cfg.adapter_ctx,
            )
            ready_dm = dm_cls(itdm_cfg=sess_cfg.datamodule_cfg, *sess_cfg.dm_args, **sess_cfg.dm_kwargs)
        if m_precomposed:
            m_cls = ITMeta(
                "InterpretunableModule", (), {}, component="m", input=sess_cfg.module_cls, ctx=sess_cfg.adapter_ctx
            )
            ready_m = m_cls(it_cfg=sess_cfg.module_cfg, *sess_cfg.module_args, **sess_cfg.module_kwargs)
        if dm_precomposed and m_precomposed:
            # we patch *_cls attributes to `None` so we can reuse the same session cfg object across multiple tests
            with (
                patch.object(sess_cfg, "datamodule", new=ready_dm),
                patch.object(sess_cfg, "module", new=ready_m),
                patch.object(sess_cfg, "datamodule_cls", None),
                patch.object(sess_cfg, "module_cls", None),
            ):
                it_session = ITSession(sess_cfg)
        elif dm_precomposed:
            with patch.object(sess_cfg, "datamodule", new=ready_dm), patch.object(sess_cfg, "datamodule_cls", None):
                it_session = ITSession(sess_cfg)
        elif m_precomposed:
            with patch.object(sess_cfg, "module", new=ready_m), patch.object(sess_cfg, "module_cls", None):
                it_session = ITSession(sess_cfg)
        assert isinstance(it_session.datamodule, ITDataModuleProtocol)
        assert isinstance(it_session.module, ITModuleProtocol)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=CORE_CTX_WARNS)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_it_session_cfg_sanitize(self, get_it_session_cfg__tl_cust):
        it_session = ITSession(get_it_session_cfg__tl_cust)
        assert len(it_session) == 2
        m_prefix = "Original module: TestITModule \nNow InterpretunableModule composing TestITModule with: \n  - ITLe"
        dm_prefix = "Original module: FingerprintTestITDataModule \nNow InterpretunableDataModule composing"
        # Normalize line endings for cross-platform compatibility
        actual_module_repr = platform_normalize_str(repr(it_session.module))
        expected_module_prefix = platform_normalize_str(m_prefix)
        assert actual_module_repr.startswith(expected_module_prefix)
        actual_dm_repr = platform_normalize_str(repr(it_session.datamodule))
        expected_dm_prefix = platform_normalize_str(dm_prefix)
        assert expected_dm_prefix in actual_dm_repr
        it_session.datamodule._module = None
        dm_no_attach_prefix = "Attached module: No module yet attached"
        assert dm_no_attach_prefix in repr(it_session.datamodule)

    def test_it_session_cfg_from_module_cls_fqn(self, get_it_session_cfg__tl_cust):
        tmp_get_it_session_cfg = deepcopy(get_it_session_cfg__tl_cust)
        tmp_get_it_session_cfg.module_cls = "tests.modules.TestITModule"
        test_tmp_session_cfg = ITSessionConfig(
            module_cls=tmp_get_it_session_cfg.module_cls,
            adapter_ctx=tmp_get_it_session_cfg.adapter_ctx,
            module_cfg=tmp_get_it_session_cfg.module_cfg,
            datamodule_cls=tmp_get_it_session_cfg.datamodule_cls,
            datamodule_cfg=tmp_get_it_session_cfg.datamodule_cfg,
            module_kwargs=tmp_get_it_session_cfg.module_kwargs,
            dm_kwargs=tmp_get_it_session_cfg.dm_kwargs,
        )
        assert isinstance(test_tmp_session_cfg, ITSessionConfig)

    def test_it_session_cfg_override_warns(self, recwarn, get_it_session_cfg__tl_cust):
        it_session_cfg = deepcopy(get_it_session_cfg__tl_cust)
        default_shared_cfg = ITSharedConfig()
        for attr in ITSharedConfig.__dataclass_fields__:
            setattr(it_session_cfg.datamodule_cfg, attr, getattr(default_shared_cfg, attr))
        rebound_kwargs = dict(
            module_cls=it_session_cfg.module_cls,
            adapter_ctx=it_session_cfg.adapter_ctx,
            module_cfg=it_session_cfg.module_cfg,
            datamodule_cls=it_session_cfg.datamodule_cls,
            datamodule_cfg=it_session_cfg.datamodule_cfg,
            module_kwargs=it_session_cfg.module_kwargs,
            dm_kwargs=it_session_cfg.dm_kwargs,
            shared_cfg=default_shared_cfg.__dict__,
        )  # exercise dict ctor branch
        expected_override_msg = "Overriding `defer_model_init`"
        # assert that we warn about overriding datamodule/module configs from a shared config only when values change
        ITSessionConfig(**rebound_kwargs)
        assert not any(expected_override_msg in str(w.message) for w in recwarn.list)
        it_session_cfg.datamodule_cfg.defer_model_init = True
        with pytest.warns(UserWarning, match=expected_override_msg):
            ITSessionConfig(**rebound_kwargs)

    def test_session_min_dep_installed(self):
        import sys
        from importlib import reload

        modules_to_reload = ("interpretune.adapters.lightning", "interpretune.base.components.cli")
        for module_fqn in modules_to_reload:
            del sys.modules[module_fqn]
        with patch("interpretune.utils._LIGHTNING_AVAILABLE", False):
            from interpretune.adapters.lightning import LightningModule, LightningDataModule

            assert LightningModule.__module__ == "builtins"
            assert LightningDataModule.__module__ == "builtins"
            from interpretune.base.components.cli import l_cli_main

            assert l_cli_main.__module__ == "builtins"
        for module_fqn in modules_to_reload:
            reload(sys.modules[module_fqn])
        from interpretune.adapters.lightning import LightningModule, LightningDataModule

        assert LightningModule.__module__ == "lightning.pytorch.core.module"
        assert LightningDataModule.__module__ == "lightning.pytorch.core.datamodule"
        from interpretune.base.components.cli import l_cli_main

        assert l_cli_main.__module__ == "interpretune.base.components.cli"
