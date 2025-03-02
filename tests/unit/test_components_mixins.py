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
from unittest import mock

import pytest
import torch

from interpretune.utils import MisconfigurationException
from interpretune.base import HFFromPretrainedMixin
from interpretune.config import ITConfig, ITExtensionsConfigMixin, HFFromPretrainedConfig, ITExtension
from tests.base_defaults import default_test_task
from tests.utils import disable_genclassif
from tests.runif import RunIf
from tests.warns import CORE_CTX_WARNS, unexpected_warns
from tests.orchestration import run_it


class TestClassMixins:

    core_gpt2_shared_config = dict(task_name=default_test_task,
        tokenizer_kwargs={"add_bos_token": True, "local_files_only": False, "padding_side": "left",
                          "model_input_names": ["input_ids", "attention_mask"]},
        model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})

    test_core_gpt2 = {**core_gpt2_shared_config,
                      "hf_from_pretrained_cfg": HFFromPretrainedConfig(pretrained_kwargs={
                          "device_map": "cpu", "torch_dtype": "float32"}, model_head="transformers.GPT2LMHeadModel")}

    @staticmethod
    def _get_hf_from_pretrained_mixin(test_it_cfg):
        it_cfg = ITConfig(**test_it_cfg)
        hf_from_pretrained_mixin = HFFromPretrainedMixin()
        hf_from_pretrained_mixin.it_cfg = it_cfg
        it_cfg.num_labels = 0
        hf_from_pretrained_mixin.torch_dtype = it_cfg._torch_dtype
        hf_from_pretrained_mixin._update_hf_pretrained_cfg()
        return hf_from_pretrained_mixin

    def test_hf_from_pretrained_config_clean(self):
        pretrained_kwargs = {"pretrained_kwargs":{"device_map": "cpu", "token": "strip-me"}}
        from_pretrained_cfg = HFFromPretrainedConfig(**pretrained_kwargs, model_head="transformers.GPT2LMHeadModel")
        assert from_pretrained_cfg.pretrained_kwargs.get('token', None) is None

    @pytest.mark.parametrize(
        "return_unused, tie_word_embeddings",
        [pytest.param(True, False)],
        ids=["return_unused_no_tie_embeddings",],
    )
    def test_hf_from_pretrained_hf_cust_config(self, return_unused, tie_word_embeddings):
        access_token = None
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg['hf_from_pretrained_cfg'].pretrained_kwargs['return_unused_kwargs'] = return_unused
        test_it_cfg['model_cfg'] = {'tie_word_embeddings': tie_word_embeddings}
        if return_unused:
            test_it_cfg['hf_from_pretrained_cfg'].pretrained_kwargs['give_it_back'] = True
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        cust_config, unused_kwargs = hf_from_pretrained_mixin._hf_gen_cust_config()
        assert cust_config.tie_word_embeddings == tie_word_embeddings
        if return_unused:
            assert 'give_it_back' in unused_kwargs
            hf_from_pretrained_mixin.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs.pop('give_it_back', None)
        model = hf_from_pretrained_mixin.hf_configured_model_init(cust_config, access_token)
        assert model.config.tie_word_embeddings == tie_word_embeddings

    @pytest.mark.parametrize(
        "head_configured, defer_init",
        [pytest.param(True, True), pytest.param(False, False), pytest.param(False, True)],
        ids=["head_config_defer_init", "no_head_config_no_defer_init", "no_head_config_defer_init"],
    )
    def test_hf_from_pretrained_hf_configured_model_init(self, head_configured, defer_init):
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg['defer_model_init'] = defer_init
        if not head_configured:
            test_it_cfg['hf_from_pretrained_cfg'].model_head = ''
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        if not head_configured and defer_init:
            with pytest.warns(UserWarning, match="`defer_model_init` not currently supported without `model_head`"):
                _ = hf_from_pretrained_mixin._hf_gen_cust_config()
        else:
            cust_config, _ = hf_from_pretrained_mixin._hf_gen_cust_config()
            _ = hf_from_pretrained_mixin.hf_configured_model_init(cust_config)

    @RunIf(min_cuda_gpus=1)
    def test_hf_from_pretrained_peft_init(self, get_it_session__core_gpt2_peft__initonly):
        it_m = get_it_session__core_gpt2_peft__initonly.module
        assert it_m.model.transformer.h[0].attn.c_proj.weight.quant_type == 'nf4'
        assert it_m.model.transformer.h[0].attn.c_proj.base_layer.compute_dtype == torch.bfloat16
        assert getattr(it_m.model.transformer.h[0].attn.c_proj, 'lora_A', None) is not None
        assert it_m.model.base_model.model.is_gradient_checkpointing

    @RunIf(min_cuda_gpus=1)
    @pytest.mark.parametrize(
        "phase, genclassif",
        [pytest.param('train', True), pytest.param('test', True), pytest.param('test', False)],
        ids=["train_genclassif", "test_genclassif", "test_no_genclassif"],
    )
    def test_peft(self, recwarn, get_it_session__core_gpt2_peft__initonly, phase, genclassif):
        expected_warnings = CORE_CTX_WARNS
        test_cfg = get_it_session__core_gpt2_peft__initonly.fixt_test_cfg()
        test_cfg.phase = phase
        if not genclassif:
            with disable_genclassif(get_it_session__core_gpt2_peft__initonly):
                run_it(it_session=get_it_session__core_gpt2_peft__initonly, test_cfg=test_cfg)
        else:
            run_it(it_session=get_it_session__core_gpt2_peft__initonly, test_cfg=test_cfg)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    @RunIf(min_cuda_gpus=1)
    def test_peft_seq_test(self, recwarn, get_it_session__core_gpt2_peft_seq__initonly):
        expected_warnings = CORE_CTX_WARNS
        run_it(it_session=get_it_session__core_gpt2_peft_seq__initonly,
               test_cfg=get_it_session__core_gpt2_peft_seq__initonly.fixt_test_cfg())
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_hf_from_pretrained_dynamic_module_load(self):
        pretrained_kwargs = {"pretrained_kwargs":{"device_map": "cpu", "torch_dtype": "float32"}}
        dynamic_module_cfg = {"config_class": "configuration_falcon.FalconConfig",
                              "model_class": "modeling_falcon.FalconForCausalLM"}
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg['model_name_or_path'] = "tiiuae/falcon-7b"
        hf_from_pretrained_cfg = HFFromPretrainedConfig(**pretrained_kwargs, dynamic_module_cfg=dynamic_module_cfg)
        test_it_cfg['hf_from_pretrained_cfg'] = hf_from_pretrained_cfg
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        hf_from_pretrained_mixin._hf_gen_cust_config()

    def test_degen_it_extension(self):
        degen_ext = ITExtension("not_here", "oops.not_found", "oops.I.did.it.again")
        ext_mixin = ITExtensionsConfigMixin()
        ext_mixin.DEFAULT_EXTENSIONS = (degen_ext,)
        with pytest.raises(MisconfigurationException):
            ext_mixin._detect_extensions()

    def test_it_generate_exception_handling(self, get_it_session__core_cust__initonly):
        core_cust_it_m = get_it_session__core_cust__initonly.module
        test_cfg = get_it_session__core_cust__initonly.fixt_test_cfg()
        test_cfg.phase = 'test'
        def generate(oops_no_matching_args):
            pass
        # we modify our generate function and avoid checking the batch inputs in order to generate our error feedback
        with mock.patch.object(core_cust_it_m.model, 'generate', generate), \
            mock.patch.object(core_cust_it_m, 'map_gen_inputs', lambda x: x):
            with pytest.warns(UserWarning, match="The following keys were found"), pytest.raises(Exception):
                run_it(it_session=get_it_session__core_cust__initonly, test_cfg=test_cfg)

    @pytest.mark.parametrize("tokenizer_id_overrides", [None, {'pad_token_id': 150}],
                             ids=["no_token_overrides", "new_token_overrides"],)
    def test_hf_from_pretrained_maybe_resize_token_embeddings(self, tokenizer_id_overrides):
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg['tokenizer_id_overrides'] = tokenizer_id_overrides
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        hf_from_pretrained_mixin.model = mock.Mock()
        hf_from_pretrained_mixin.model.base_model = mock.Mock()
        hf_from_pretrained_mixin.model.base_model.vocab_size = 100
        expected_calls = 0 if tokenizer_id_overrides is None else 1
        hf_from_pretrained_mixin._hf_maybe_resize_token_embeddings()
        assert hf_from_pretrained_mixin.model.base_model.resize_token_embeddings.call_count == expected_calls
