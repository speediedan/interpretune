from copy import deepcopy

import pytest
from torch import device

from interpretune.adapters.transformer_lens import ITLensFromPretrainedConfig, ITLensConfig, ITLensCustomConfig
from interpretune.utils.exceptions import MisconfigurationException
from tests.utils.warns import unexpected_warns, TL_CTX_WARNS
from tests.utils.misc import ablate_cls_attrs
from tests.parity_acceptance.cfg_aliases import test_tl_gpt2_it_module_base, test_tl_cust_it_module_base


class TestClassTransformerLens:

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
        test_tl_cfg = deepcopy(test_tl_cust_it_module_base)
        test_tl_cfg.update({'tl_cfg': None})
        with pytest.raises(MisconfigurationException, match="Either a `ITLens"):
            _ = ITLensConfig(**test_tl_cfg)

    @pytest.mark.parametrize(
        "use_hf_pretrained, hf_pretrained_cfg, tl_cust_cfg, tl_pretrained_cfg, base_cfg, expected_warn, expected_error",
        [pytest.param(True, {}, None, None, test_tl_gpt2_it_module_base, "dtype was not provided. Setting", None),
         pytest.param(False, {'x': 2}, {'dtype': 'float32'}, None, test_tl_cust_it_module_base,
                      "attributes will be ignore", None),
         pytest.param(True, {'pretrained_kwargs': {'device_map': {"unsupp": 0, "lm_head": 1}}}, None, None,
                      test_tl_gpt2_it_module_base, "mapping to multiple devices", None),
         pytest.param(True, {'dict unconvertiable': 'to hf_cfg'}, None, None, test_tl_gpt2_it_module_base, None,
                      "or a dict convertible"),
         pytest.param(True, {'pretrained_kwargs': {'torch_dtype': 'bfloat16'}}, None, {'dtype': 'float32'},
                      test_tl_gpt2_it_module_base, "does not match TL dtype", None),
                      ],
        ids=["no_hf_cfg_warn", "hf_cfg_ignore_warn", "multi_device_map_warn", "invalid_hf_cfg_error",
             "TL_HF_dtype_mismatch_warn"],
    )
    def test_tl_pretrained_cfgs(self, recwarn, use_hf_pretrained, hf_pretrained_cfg, tl_cust_cfg, tl_pretrained_cfg,
                                base_cfg, expected_warn, expected_error):
        test_tl_cfg = deepcopy(base_cfg)
        if hf_pretrained_cfg is not None:
            test_tl_cfg['hf_from_pretrained_cfg'] =  hf_pretrained_cfg
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
