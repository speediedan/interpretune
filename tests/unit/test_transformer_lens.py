from copy import deepcopy

import pytest
from torch import device

from interpretune.config import ITLensFromPretrainedConfig, ITLensConfig, ITLensCustomConfig
from interpretune.utils import MisconfigurationException
from tests.warns import unexpected_warns, TL_CTX_WARNS
from tests.utils import ablate_cls_attrs
from tests.base_defaults import default_test_task


class TestClassTransformerLens:

    tl_tokenizer_kwargs = {"add_bos_token": True, "local_files_only": False,  "padding_side": "left",
                            "model_input_names": ['input', 'attention_mask']}
    test_tl_signature_columns = ['input', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds',
                                     'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
    test_tl_gpt2_shared_config = dict(task_name=default_test_task, tokenizer_kwargs=tl_tokenizer_kwargs,
                                    model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})
    test_tl_cust_config = {"cfg": {"n_layers":1, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
                        "act_fn":'relu', "tokenizer_name": 'gpt2'}}
    test_tlens_gpt2 = {**test_tl_gpt2_shared_config, "tl_cfg": {}, "hf_from_pretrained_cfg": dict(
        pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"}, model_head="transformers.GPT2LMHeadModel")}
    test_tlens_cust = {**test_tl_gpt2_shared_config, "tl_cfg": test_tl_cust_config}

    def test_tl_session_exceptions(self, get_it_session__tl_cust__setup):
        tl_test_module = get_it_session__tl_cust__setup.module
        with ablate_cls_attrs(tl_test_module.model, 'cfg'), pytest.warns(UserWarning, match="Could not find a `Hooked"):
                _ = tl_test_module.tl_cfg
        with ablate_cls_attrs(tl_test_module._it_state, '_device'), ablate_cls_attrs(tl_test_module.tl_cfg, 'device'):
            with pytest.warns(UserWarning, match="Could not find a device reference"):
                _ = tl_test_module.device
            with pytest.warns(UserWarning, match="determining appropriate device for block"):
                 _ = tl_test_module.get_tl_device(0)
        tl_test_module.device = 'meta'
        assert isinstance(tl_test_module.device, device)
        assert tl_test_module.device.type == 'meta'

    def test_tl_session_cfg_exceptions(self):
        test_tl_cfg = deepcopy(TestClassTransformerLens.test_tlens_cust)
        test_tl_cfg.update({'tl_cfg': None})
        with pytest.raises(MisconfigurationException, match="Either a `ITLens"):
            _ = ITLensConfig(**test_tl_cfg)

    @pytest.mark.parametrize(
        "use_hf_pretrained, hf_pretrained_cfg, tl_cust_cfg, tl_pretrained_cfg, expected_warn, expected_error",
        [pytest.param(True, {}, None, None, "dtype was not provided. Setting", None),
         pytest.param(False, {'x': 2}, {'dtype': 'float32'}, None, "attributes will be ignore", None),
         pytest.param(True, {'pretrained_kwargs': {'device_map': {"unsupp": 0, "lm_head": 1}}}, None, None,
                      "mapping to multiple devices", None),
         pytest.param(True, {'dict unconvertible': 'to hf_cfg'}, None, None, None, "or a dict convertible"),
         pytest.param(True, {'pretrained_kwargs': {'torch_dtype': 'bfloat16'}}, None, {'dtype': 'float32'},
                      "does not match TL dtype", None),
                      ],
        ids=["no_hf_cfg_warn", "hf_cfg_ignore_warn", "multi_device_map_warn", "invalid_hf_cfg_error",
             "TL_HF_dtype_mismatch_warn"],
    )
    def test_tl_pretrained_cfgs(self, recwarn, use_hf_pretrained, hf_pretrained_cfg, tl_cust_cfg, tl_pretrained_cfg,
                                expected_warn, expected_error):
        test_tl_cfg = deepcopy(TestClassTransformerLens.test_tlens_gpt2 if use_hf_pretrained else \
                               TestClassTransformerLens.test_tlens_cust)
        if hf_pretrained_cfg is not None:
            test_tl_cfg['hf_from_pretrained_cfg'] = hf_pretrained_cfg
        if tl_cust_cfg is not None:
            test_tl_cfg['tl_cfg']['cfg'].update(tl_cust_cfg)
        if tl_pretrained_cfg is not None:
            test_tl_cfg['tl_cfg'].update(tl_pretrained_cfg)
        if use_hf_pretrained:
            test_tl_cfg['tl_cfg'] = ITLensFromPretrainedConfig(**test_tl_cfg['tl_cfg'])
        else:
            test_tl_cfg['tl_cfg'] = ITLensCustomConfig(**test_tl_cfg['tl_cfg'])
        if expected_warn:
            with pytest.warns(UserWarning, match=expected_warn):
                _ = ITLensConfig(**test_tl_cfg)
        if expected_error:
            with pytest.raises(MisconfigurationException, match=expected_error):
                _ = ITLensConfig(**test_tl_cfg)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=TL_CTX_WARNS)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
