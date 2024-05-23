from copy import deepcopy

import pytest

from tests.parity_acceptance.adapters.lightning.cfg_aliases import test_core_gpt2_it_module_base
from parity_acceptance.adapters.lightning.test_interpretune_l import CoreCfg
from tests.configuration import get_itdm_cfg
from interpretune.base.config.module import ITConfig
from interpretune.base.config.mixins import HFFromPretrainedConfig

class TestClassBaseConfigs:

    def test_datamodule_cfg_defer_init(self):
        test_cfg = CoreCfg()
        test_cfg.dm_override_cfg = {"defer_model_init": True}
        itdm_cfg = get_itdm_cfg(test_cfg=test_cfg, dm_override_cfg=test_cfg.dm_override_cfg)
        assert itdm_cfg

    def test_hf_from_pretrained_cfg_validation(self):
        pretrained_kwargs = {"pretrained_kwargs":{"device_map": "cpu", "torch_dtype": "float32", "token": "strip-me"}}
        from_pretrained_cfg = HFFromPretrainedConfig(**pretrained_kwargs, model_head="transformers.GPT2LMHeadModel")
        assert from_pretrained_cfg.pretrained_kwargs.get('token', None) is None
        test_it_cfg = deepcopy(test_core_gpt2_it_module_base)
        test_it_cfg['hf_from_pretrained_cfg'].bitsandbytesconfig = True
        it_cfg = ITConfig(**test_it_cfg)
        assert it_cfg._torch_dtype is None
        with pytest.warns(UserWarning, match="attempting to proceed with `torch_dtype` unset"):
            it_cfg.hf_from_pretrained_cfg.pretrained_kwargs['torch_dtype'] = "unresolvable"
            it_cfg.hf_from_pretrained_cfg._torch_dtype_serde()
        # TODO: add testing for core IT object custom constructor/representers, e.g. for ITConfig here
        # cfg_file = tmp_path / "test_config.yaml"
        # with open(cfg_file, "w") as f:
        #     f.write(yaml.dump(it_cfg))
        # with open(cfg_file, "r") as f:
        #     cfg = yaml.load(f, Loader=yaml.FullLoader)
