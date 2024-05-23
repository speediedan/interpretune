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

from tests.utils.runif import RunIf
from tests.parity_acceptance.adapters.lightning.cfg_aliases import test_core_gpt2_it_module_base
from tests.utils.warns import CORE_CTX_WARNS, unexpected_warns
from tests.orchestration import run_it, disable_zero_shot
from interpretune.utils.exceptions import MisconfigurationException
from interpretune.base.components.mixins import HFFromPretrainedMixin
from interpretune.base.config.module import ITConfig
from interpretune.base.config.extensions import ITExtensionsConfigMixin
from interpretune.base.config.mixins import HFFromPretrainedConfig, ITExtension

class TestClassMixins:

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

    def test_hf_from_pretrained_hf_cust_config(self):
        test_it_cfg = deepcopy(test_core_gpt2_it_module_base)
        test_it_cfg['defer_model_init'] = True
        test_it_cfg['hf_from_pretrained_cfg'].model_head = ''
        test_it_cfg['hf_from_pretrained_cfg'].pretrained_kwargs['return_unused_kwargs'] = True
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        with pytest.warns(UserWarning, match="`defer_model_init` not currently supported without `model_head`"):
            _ = hf_from_pretrained_mixin._hf_gen_cust_config()

    @pytest.mark.parametrize(
        "head_configured, defer_init",
        [pytest.param(True, True), pytest.param(False, False)],
        ids=["head_config_defer_init", "no_head_config_no_defer_init"],
    )
    def test_hf_from_pretrained_hf_configured_model_init(self, head_configured, defer_init):
        test_it_cfg = deepcopy(test_core_gpt2_it_module_base)
        test_it_cfg['defer_model_init'] = defer_init
        if not head_configured:
            test_it_cfg['hf_from_pretrained_cfg'].model_head = ''
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        cust_config = hf_from_pretrained_mixin._hf_gen_cust_config()
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
        "phase, zero_shot",
        [pytest.param('train', True), pytest.param('test', True), pytest.param('test', False)],
        ids=["train_zero_shot", "test_zero_shot", "test_no_zero_shot"],
    )
    def test_peft(self, recwarn, get_it_session__core_gpt2_peft__initonly, phase, zero_shot):
        expected_warnings = CORE_CTX_WARNS
        test_cfg = get_it_session__core_gpt2_peft__initonly.fixt_test_cfg()
        test_cfg.phase = phase
        if not zero_shot:
            with disable_zero_shot(get_it_session__core_gpt2_peft__initonly):
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
        test_it_cfg = deepcopy(test_core_gpt2_it_module_base)
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
        # we modify our generate function and avoid checking the batch inputs in order to generate our error feedeback
        with mock.patch.object(core_cust_it_m.model, 'generate', generate), \
            mock.patch.object(core_cust_it_m, 'map_gen_inputs', lambda x: x):
            with pytest.warns(UserWarning, match="The following keys were found"), pytest.raises(Exception):
                run_it(it_session=get_it_session__core_cust__initonly, test_cfg=test_cfg)
