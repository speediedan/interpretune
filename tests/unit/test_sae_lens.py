from copy import deepcopy

from sae_lens.sae import SAE

from interpretune.adapters.sae_lens import SAELensFromPretrainedConfig, SAELensConfig
from tests.warns import unexpected_warns

#from tests.utils import ablate_cls_attrs
from tests.runif import RunIf
from tests.warns import CORE_CTX_WARNS
from tests.orchestration import run_lightning

class TestClassSAELens:

    sl_tokenizer_kwargs = {"add_bos_token": True, "local_files_only": False,  "padding_side": "left",
                            "model_input_names": ['input_ids', 'attention_mask']}
    test_sl_signature_columns = ['inputs', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds',
                                     'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
    test_sl_gpt2_shared_config = dict(task_name="pytest_rte_hf", tokenizer_kwargs=sl_tokenizer_kwargs,
                                      model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})
    test_sl_from_pretrained = SAELensFromPretrainedConfig(release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre",
                                                          device="cpu")
    # test_sl_cust_config = {"cfg": {"n_layers":1, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
    #                     "act_fn":'relu', "tokenizer_name": 'gpt2'}}
    test_sl_gpt2 = {**test_sl_gpt2_shared_config, "sl_cfg": test_sl_from_pretrained}
    # test_tlens_cust = {**test_sl_gpt2_shared_config, "tl_cfg": test_tl_cust_config}

    # def test_tl_session_exceptions(self, get_it_session__tl_cust__setup):
    #     tl_test_module = get_it_session__tl_cust__setup.module
    #     with ablate_cls_attrs(tl_test_module.model, 'cfg'), pytest.warns(UserWarning,
    #                                                                      match="Could not find a `Hooked"):
    #             _ = tl_test_module.tl_cfg
    #     with ablate_cls_attrs(tl_test_module._it_state, '_device'), ablate_cls_attrs(tl_test_module.tl_cfg, 'device'):
    #         with pytest.warns(UserWarning, match="Could not find a device reference"):
    #             _ = tl_test_module.device
    #         with pytest.warns(UserWarning, match="determining appropriate device for block"):
    #              _ = tl_test_module.get_tl_device(0)
    #     tl_test_module.device = 'meta'
    #     assert isinstance(tl_test_module.device, device)
    #     assert tl_test_module.device.type == 'meta'

    # def test_tl_session_cfg_exceptions(self):
    #     test_tl_cfg = deepcopy(TestClassSAELens.test_tlens_cust)
    #     test_tl_cfg.update({'tl_cfg': None})
    #     with pytest.raises(MisconfigurationException, match="Either a `ITLens"):
    #         _ = ITLensConfig(**test_tl_cfg)

    @RunIf(min_cuda_gpus=1)
    def test_basic(self, recwarn, get_it_session__l_sl_gpt2__initonly):
        expected_warnings = CORE_CTX_WARNS
        _ = run_lightning(it_session=get_it_session__l_sl_gpt2__initonly,
               test_cfg=get_it_session__l_sl_gpt2__initonly.fixt_test_cfg(),
               tmp_path=get_it_session__l_sl_gpt2__initonly.module.it_cfg.core_log_dir)
        # run_it(it_session=get_it_session__l_sl_gpt2__initonly,
        #        test_cfg=get_it_session__l_sl_gpt2__initonly.fixt_test_cfg())
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_sl_session_cfg(self):
        test_sl_cfg = deepcopy(TestClassSAELens.test_sl_gpt2)
        it_cfg = SAELensConfig(**test_sl_cfg)
        model, original_cfg_dict, sparsity = SAE.from_pretrained(**it_cfg.sl_cfg.__dict__)
        pass
