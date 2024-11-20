from copy import deepcopy

import pytest

from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.base.config.mixins import HFFromPretrainedConfig
from tests.base_defaults import default_test_task


class TestClassBaseConfigs:

    core_gpt2_shared_config = dict(task_name=default_test_task,
        tokenizer_kwargs={"add_bos_token": True, "local_files_only": False, "padding_side": "left",
                          "model_input_names": ["input_ids", "attention_mask"]},
        model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})

    test_core_gpt2 = {**core_gpt2_shared_config,
                      "hf_from_pretrained_cfg": HFFromPretrainedConfig(pretrained_kwargs={
                          "device_map": "cpu", "torch_dtype": "float32"}, model_head="transformers.GPT2LMHeadModel")}

    def test_datamodule_cfg_defer_init(self):
        test_core_datamodule = {**TestClassBaseConfigs.core_gpt2_shared_config, "signature_columns": ["input_ids"]}
        itdm_cfg = deepcopy(test_core_datamodule) | {"defer_model_init": True}
        assert ITDataModuleConfig(**itdm_cfg)

    def test_hf_from_pretrained_cfg_validation(self):
        pretrained_kwargs = {"pretrained_kwargs":{"device_map": "cpu", "torch_dtype": "float32", "token": "strip-me"}}
        from_pretrained_cfg = HFFromPretrainedConfig(**pretrained_kwargs, model_head="transformers.GPT2LMHeadModel")
        assert from_pretrained_cfg.pretrained_kwargs.get('token', None) is None
        test_it_cfg = deepcopy(TestClassBaseConfigs.test_core_gpt2)
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
