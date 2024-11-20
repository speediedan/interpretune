from copy import deepcopy

import pytest

from interpretune.utils.exceptions import MisconfigurationException
from interpretune.adapters.sae_lens import SAELensFromPretrainedConfig, SAELensConfig, SAELensCustomConfig
from interpretune.adapters.transformer_lens import ITLensFromPretrainedConfig, ITLensCustomConfig
from tests.warns import unexpected_warns

from tests.utils import ablate_cls_attrs
from tests.runif import RunIf
from tests.warns import SL_LIGHTNING_CTX_WARNS
from tests.orchestration import run_lightning
from tests.base_defaults import default_test_task

class TestClassSAELens:

    sl_tokenizer_kwargs = {"add_bos_token": True, "local_files_only": False,  "padding_side": "left",
                            "model_input_names": ['input_ids', 'attention_mask']}
    test_sl_signature_columns = ['inputs', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds',
                                     'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
    test_tl_gpt2_shared_config = dict(task_name=default_test_task, tokenizer_kwargs=sl_tokenizer_kwargs,
                                    model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})
    test_tlens_gpt2 = {**test_tl_gpt2_shared_config, "tl_cfg": ITLensFromPretrainedConfig(),
                       "hf_from_pretrained_cfg": dict(
        pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"}, model_head="transformers.GPT2LMHeadModel")}
    test_sl_from_pretrained = SAELensFromPretrainedConfig(release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre",
                                                          device="cpu")

    # tiny custom tl model to attach custom SAE to
    test_tl_cust_config = {"n_layers":1, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
                        "act_fn":'relu', "tokenizer_name": 'gpt2'}

    test_sae_cust_config = dict(
        architecture="standard",
        d_in=10,
        d_sae=10 * 2,
        dtype="float32",
        device="cpu",
        model_name="cust",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",
        prepend_bos=True,
        context_size=200,
        dataset_path="test",
        dataset_trust_remote_code=True,
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        sae_lens_training_version=None,
        normalize_activations="none",
    )
    test_sl_cust_config = SAELensCustomConfig(cfg=test_sae_cust_config)

    test_sl_gpt2 = {**test_tl_gpt2_shared_config, **test_tlens_gpt2, "sae_cfg": test_sl_from_pretrained}
    test_sl_cust = {**test_tl_gpt2_shared_config, "tl_cfg": ITLensCustomConfig(cfg=test_tl_cust_config),
                    "sae_cfg": test_sl_cust_config}

    @RunIf(min_cuda_gpus=1)
    def test_basic(self, recwarn, get_it_session__l_sl_gpt2__initonly):
        expected_warnings = SL_LIGHTNING_CTX_WARNS
        sl_test_module = get_it_session__l_sl_gpt2__initonly.module
        with ablate_cls_attrs(sl_test_module.it_cfg, 'sae_cfg'), pytest.warns(UserWarning, match="Could not find a `S"):
            _ = sl_test_module.sae_cfg
        _ = run_lightning(it_session=get_it_session__l_sl_gpt2__initonly,
               test_cfg=get_it_session__l_sl_gpt2__initonly.fixt_test_cfg(),
               tmp_path=get_it_session__l_sl_gpt2__initonly.module.it_cfg.core_log_dir)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_sl_session_cfg(self):
        test_sae_cfg = deepcopy(TestClassSAELens.test_sl_cust)
        it_cfg = SAELensConfig(**test_sae_cfg)
        assert it_cfg

    def test_sl_session_cfg_exceptions(self):
        test_sl_cfg = deepcopy(TestClassSAELens.test_sl_cust)
        test_sl_cfg.update({'sae_cfg': None})
        with pytest.raises(MisconfigurationException, match="Either a `SAELens"):
            _ = SAELensConfig(**test_sl_cfg)
