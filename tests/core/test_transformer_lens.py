from copy import deepcopy

import pytest
from torch import device

from interpretune.config import ITLensFromPretrainedConfig, ITLensConfig, ITLensCustomConfig
from interpretune.utils import MisconfigurationException
from tests.warns import unexpected_warns, TL_CTX_WARNS
from tests.utils import ablate_cls_attrs
from tests.base_defaults import default_test_task


class TestClassTransformerLens:
    tl_tokenizer_kwargs = {
        "add_bos_token": True,
        "local_files_only": False,
        "padding_side": "left",
        "model_input_names": ["input", "attention_mask"],
    }
    test_tl_signature_columns = [
        "input",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "use_cache",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
    ]
    test_tl_gpt2_shared_config = dict(
        task_name=default_test_task,
        tokenizer_kwargs=tl_tokenizer_kwargs,
        model_name_or_path="gpt2",
        tokenizer_id_overrides={"pad_token_id": 50256},
    )
    test_tl_cust_config = {
        "cfg": {
            "n_layers": 1,
            "d_mlp": 10,
            "d_model": 10,
            "d_head": 5,
            "n_heads": 2,
            "n_ctx": 200,
            "act_fn": "relu",
            "tokenizer_name": "gpt2",
        }
    }
    test_tlens_gpt2 = {
        **test_tl_gpt2_shared_config,
        "tl_cfg": {},
        "hf_from_pretrained_cfg": dict(
            pretrained_kwargs={"device_map": "cpu", "dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        ),
    }
    test_tlens_cust = {**test_tl_gpt2_shared_config, "tl_cfg": test_tl_cust_config}

    def test_tl_session_exceptions(self, get_it_session__tl_cust__setup):
        fixture = get_it_session__tl_cust__setup
        tl_test_module = fixture.it_session.module
        with (
            ablate_cls_attrs(tl_test_module.model, "cfg"),
            pytest.warns(UserWarning, match="Could not find a TransformerLens config"),
        ):
            _ = tl_test_module.tl_cfg
        with ablate_cls_attrs(tl_test_module._it_state, "_device"), ablate_cls_attrs(tl_test_module.tl_cfg, "device"):
            with pytest.warns(UserWarning, match="Could not find a device reference"):
                _ = tl_test_module.device
            with pytest.warns(UserWarning, match="determining appropriate device from TransformerLens"):
                _ = tl_test_module.get_tl_device()
        tl_test_module.device = "meta"
        assert isinstance(tl_test_module.device, device)
        assert tl_test_module.device.type == "meta"

    def test_tl_session_cfg_exceptions(self):
        test_tl_cfg = deepcopy(TestClassTransformerLens.test_tlens_cust)
        test_tl_cfg.update({"tl_cfg": None})
        with pytest.raises(MisconfigurationException, match="Either a `ITLens"):
            _ = ITLensConfig(**test_tl_cfg)

    @pytest.mark.parametrize(
        "use_hf_pretrained, hf_pretrained_cfg, tl_cust_cfg, tl_pretrained_cfg, expected_warn, expected_error",
        [
            pytest.param(True, {}, None, None, "dtype was not provided. Setting", None),
            pytest.param(False, {"x": 2}, {"dtype": "float32"}, None, "attributes will be ignore", None),
            pytest.param(
                True,
                {"pretrained_kwargs": {"device_map": {"unsupp": 0, "lm_head": 1}}},
                None,
                None,
                "mapping to multiple devices",
                None,
            ),
            pytest.param(True, {"dict unconvertible": "to hf_cfg"}, None, None, None, "or a dict convertible"),
            pytest.param(
                True,
                {"pretrained_kwargs": {"dtype": "bfloat16"}},
                None,
                {"dtype": "float32"},
                "does not match TL dtype",
                None,
            ),
        ],
        ids=[
            "no_hf_cfg_warn",
            "hf_cfg_ignore_warn",
            "multi_device_map_warn",
            "invalid_hf_cfg_error",
            "TL_HF_dtype_mismatch_warn",
        ],
    )
    def test_tl_pretrained_cfgs(
        self,
        recwarn,
        use_hf_pretrained,
        hf_pretrained_cfg,
        tl_cust_cfg,
        tl_pretrained_cfg,
        expected_warn,
        expected_error,
    ):
        test_tl_cfg = deepcopy(
            TestClassTransformerLens.test_tlens_gpt2 if use_hf_pretrained else TestClassTransformerLens.test_tlens_cust
        )
        if hf_pretrained_cfg is not None:
            test_tl_cfg["hf_from_pretrained_cfg"] = hf_pretrained_cfg
        if tl_cust_cfg is not None:
            test_tl_cfg["tl_cfg"]["cfg"].update(tl_cust_cfg)
        if tl_pretrained_cfg is not None:
            test_tl_cfg["tl_cfg"].update(tl_pretrained_cfg)
        if use_hf_pretrained:
            test_tl_cfg["tl_cfg"] = ITLensFromPretrainedConfig(**test_tl_cfg["tl_cfg"])
        else:
            test_tl_cfg["tl_cfg"] = ITLensCustomConfig(**test_tl_cfg["tl_cfg"])
        if expected_warn:
            with pytest.warns(UserWarning, match=expected_warn):
                _ = ITLensConfig(**test_tl_cfg)
        if expected_error:
            with pytest.raises(MisconfigurationException, match=expected_error):
                _ = ITLensConfig(**test_tl_cfg)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=TL_CTX_WARNS)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_prune_tl_cfg_dict_warns(self, monkeypatch, get_it_session__tl_cust__setup):
        """Test that _prune_tl_cfg_dict warns when non-None values for 'hf_model' or 'tokenizer' are found."""
        fixture = get_it_session__tl_cust__setup
        tl_test_module = fixture.it_session.module

        # Create a mock config with non-None values for keys that should be pruned
        mock_config = deepcopy(tl_test_module.it_cfg.tl_cfg)
        mock_config.hf_model = "some_value"  # non-None value for hf_model
        mock_config.tokenizer = "another_value"  # non-None value for tokenizer

        # Monkeypatch the it_cfg.tl_cfg to use our mock config
        monkeypatch.setattr(tl_test_module.it_cfg, "tl_cfg", mock_config)

        # Verify warnings are raised when _prune_tl_cfg_dict is called
        with pytest.warns(UserWarning, match="Found non-None value for 'hf_model' in tl_cfg"):
            with pytest.warns(UserWarning, match="Found non-None value for 'tokenizer' in tl_cfg"):
                pruned_dict = tl_test_module._prune_tl_cfg_dict()

        # Verify the keys were actually removed despite having values
        assert "hf_model" not in pruned_dict
        assert "tokenizer" not in pruned_dict

    def test_tl_use_bridge_config(self):
        """Test that use_bridge configuration option is properly handled."""
        # Test with use_bridge=True (default, TransformerBridge)
        test_tl_cfg = deepcopy(TestClassTransformerLens.test_tlens_gpt2)
        test_tl_cfg["tl_cfg"]["use_bridge"] = True
        test_tl_cfg["tl_cfg"] = ITLensFromPretrainedConfig(**test_tl_cfg["tl_cfg"])
        it_cfg_bridge = ITLensConfig(**test_tl_cfg)
        assert it_cfg_bridge.tl_cfg.use_bridge is True

        # Test with use_bridge=False (legacy HookedTransformer)
        test_tl_cfg_legacy = deepcopy(TestClassTransformerLens.test_tlens_gpt2)
        test_tl_cfg_legacy["tl_cfg"]["use_bridge"] = False
        test_tl_cfg_legacy["tl_cfg"] = ITLensFromPretrainedConfig(**test_tl_cfg_legacy["tl_cfg"])
        it_cfg_legacy = ITLensConfig(**test_tl_cfg_legacy)
        assert it_cfg_legacy.tl_cfg.use_bridge is False

        # Test default (should be True)
        test_tl_cfg_default = deepcopy(TestClassTransformerLens.test_tlens_gpt2)
        test_tl_cfg_default["tl_cfg"] = ITLensFromPretrainedConfig(**test_tl_cfg_default["tl_cfg"])
        it_cfg_default = ITLensConfig(**test_tl_cfg_default)
        assert it_cfg_default.tl_cfg.use_bridge is False  # TODO: remove this switch after debugging upstream issue
        # assert it_cfg_default.tl_cfg.use_bridge is True  # default should be True

        # Test default for custom config: should default to False (HookedTransformer)
        test_tl_cfg_custom_default = deepcopy(TestClassTransformerLens.test_tlens_cust)
        test_tl_cfg_custom_default["tl_cfg"] = ITLensCustomConfig(**test_tl_cfg_custom_default["tl_cfg"])
        it_cfg_custom = ITLensConfig(**test_tl_cfg_custom_default)
        assert it_cfg_custom.tl_cfg.use_bridge is False

        # If user sets use_bridge=True in a custom config, warn and force to False
        test_tl_cfg_custom_override = deepcopy(TestClassTransformerLens.test_tlens_cust)
        test_tl_cfg_custom_override["tl_cfg"]["use_bridge"] = True
        test_tl_cfg_custom_override["tl_cfg"] = ITLensCustomConfig(**test_tl_cfg_custom_override["tl_cfg"])
        with pytest.warns(UserWarning, match="ITLensCustomConfig does not support TransformerBridge"):
            it_cfg_custom_override = ITLensConfig(**test_tl_cfg_custom_override)
        assert it_cfg_custom_override.tl_cfg.use_bridge is False
