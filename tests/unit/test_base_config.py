from copy import deepcopy

import pytest
from unittest.mock import Mock, patch

from interpretune.config import (ITDataModuleConfig, ITLensFromPretrainedConfig, ITLensConfig,
                                 SAELensFromPretrainedConfig, SAELensConfig, ITConfig, AutoCompConfig,
                                 search_candidate_subclass_attrs, HFFromPretrainedConfig)
from it_examples.experiments.rte_boolq import (RTEBoolqEntailmentMapping, GenerativeClassificationConfig,
                                               RTEBoolqSLConfig)
from interpretune.adapters import TLensGenerationConfig
from tests.base_defaults import default_test_task


class TestClassBaseConfigs:

    core_gpt2_shared_config = dict(task_name=default_test_task,
        tokenizer_kwargs={"add_bos_token": True, "local_files_only": False, "padding_side": "left",
                            "model_input_names": ["input_ids", "attention_mask"]},
        model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})
    test_core_gpt2 = {**core_gpt2_shared_config,
                    "hf_from_pretrained_cfg": HFFromPretrainedConfig(pretrained_kwargs={
                        "device_map": "cpu", "torch_dtype": "float32"}, model_head="transformers.GPT2LMHeadModel")}
    tl_gpt2_shared_config = deepcopy(core_gpt2_shared_config)
    tl_gpt2_shared_config["tokenizer_kwargs"]["model_input_names"] = ["input"]
    auto_comp_init_kwargs = { "module_cfg_name": "RTEBoolqConfig", "module_cfg_mixin": RTEBoolqEntailmentMapping}
    sae_cfgs = {"sae_cfgs": [SAELensFromPretrainedConfig(release="gpt2-small-res-jb",
                                                              sae_id="blocks.0.hook_resid_pre", device="cuda")]}
    @staticmethod
    def _gen_base_module_kwargs():
        return {
            **TestClassBaseConfigs.tl_gpt2_shared_config,
            "generative_step_cfg":
            GenerativeClassificationConfig(enabled=True, lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)),
            "hf_from_pretrained_cfg": HFFromPretrainedConfig(
                pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                model_head="transformers.GPT2LMHeadModel"),
            "tl_cfg": ITLensFromPretrainedConfig()}

    @staticmethod
    def gen_autocomp(w_sae_cfg=True, no_subcls=False, target_adapters=None, sae_cfgs=sae_cfgs):
        # note, we use this factory approach instead of subclassing AutoCompConfig to more directly test our desired
        # ctor behavior
        module_cfg_name = "RTEBoolqConfig"
        module_cfg_mixin = RTEBoolqEntailmentMapping if not no_subcls else object
        auto_comp_cfg = {"auto_comp_cfg": AutoCompConfig(
            module_cfg_name=module_cfg_name, module_cfg_mixin=module_cfg_mixin, target_adapters=target_adapters)}
        if w_sae_cfg:
            return {**sae_cfgs, **auto_comp_cfg}
        return auto_comp_cfg

    expected_composition_states = {
        ("RTEBoolqSLConfig",) : "RTEBoolqSLConfig(",
        ("ITConfig", "ITLensConfig", "RTEBoolqEntailmentMapping") :
            "Original module cfg: ITConfig \nNow RTEBoolqConfig, a composition of: \n  - ITLensConfig\n  - "
            "RTEBoolqEntailmentMapping",
        ("ITConfig", "RTEBoolqEntailmentMapping", "SAELensConfig") :
            "Original module cfg: ITConfig \nNow RTEBoolqConfig, a composition of: \n  - SAELensConfig\n  - "
            "RTEBoolqEntailmentMapping",
        ("ITConfig", "SAELensConfig") :
            "Original module cfg: ITConfig \nNow RTEBoolqConfig, a composition of: \n  - SAELensConfig\n",
        ("RTEBoolqEntailmentMapping", "SAELensConfig",  "SAELensConfig") :
            "Original module cfg: SAELensConfig \nNow RTEBoolqConfig, a composition of: \n  - SAELensConfig\n  - "
            "RTEBoolqEntailmentMapping",
        ("ITLensConfig", "ITLensConfig", "RTEBoolqEntailmentMapping") :
            "Original module cfg: ITLensConfig \nNow RTEBoolqConfig, a composition of: \n  - ITLensConfig\n  - "
            "RTEBoolqEntailmentMapping",
    }

    # auto-comp test cfg aliases
    exp_no_comp = (set(), None, True)
    ent_tl_set = {RTEBoolqEntailmentMapping, ITLensConfig}
    ent_sl_set = {RTEBoolqEntailmentMapping, SAELensConfig}
    no_comp_msg = 'Could not find an auto-composition for'

    # NOTE: [Auto-Composition Test Variants]
    #   - `unable_to_resolve_w_subcls`: results in a degen ITLensConfig that is the result of a composition of
    #     ITLensConfig with RTEBoolqEntailmentMapping. This new version of the dataclass will support the needed kwargs
    #     but won't have the proper post_init methods etc so may not work as expected
    #   - `unable_to_resolve_no_subcls`: results in a degen ITLensConfig, but it will result in a TypeError since it
    #     uses the original ITLensConfig rather than composing with RTEBoolqEntailmentMapping

    @pytest.mark.parametrize(
        "cfg_cls, init_kwargs, expected",
        [
            (RTEBoolqSLConfig, {**sae_cfgs}, exp_no_comp),
            (RTEBoolqSLConfig, gen_autocomp(target_adapters="lightning"), (set(), 'No candi', True)),
            (RTEBoolqSLConfig, gen_autocomp(), exp_no_comp),
            (RTEBoolqSLConfig, gen_autocomp(target_adapters="sae_lens"), exp_no_comp),
            (ITLensConfig, gen_autocomp(target_adapters="transformer_lens"), (ent_tl_set, no_comp_msg, True)),
            (ITLensConfig, gen_autocomp(no_subcls=True, target_adapters="lightning"), (ent_tl_set, no_comp_msg, False)),
            (SAELensConfig, gen_autocomp(target_adapters="sae_lens"), (ent_sl_set, None, True)),
            (ITConfig, gen_autocomp(), (ent_sl_set, None, True)),
            (ITConfig, gen_autocomp(no_subcls=True), ({SAELensConfig}, None, True)),
            (ITConfig, gen_autocomp(w_sae_cfg=False, target_adapters=("transformer_lens",)),  (ent_tl_set, None, True)),
        ], ids=["explicit_subclass_sl", "no_comp_req_no_match", "no_comp_required", "already_specified_subcls",
                "unable_to_resolve_w_subcls", "unable_to_resolve_no_subcls", "subclass_subset_match",
                "auto_comp_no_target", "auto_comp_no_subcls_no_target", "auto_comp_with_target"]
    )
    def test_module_cfg_auto_composition(self, cfg_cls, init_kwargs, expected):
        expected_composition_classes, expected_warn, complete_inspect = expected[0], expected[1], expected[2]
        if expected_warn and complete_inspect:
            with pytest.warns(UserWarning, match=expected_warn):
                cfg = cfg_cls(**TestClassBaseConfigs._gen_base_module_kwargs(), **init_kwargs)
        elif expected_warn:
            with pytest.warns(UserWarning, match=expected_warn), pytest.raises(TypeError):
                cfg = cfg_cls(**TestClassBaseConfigs._gen_base_module_kwargs(), **init_kwargs)
        else:
            cfg = cfg_cls(**TestClassBaseConfigs._gen_base_module_kwargs(), **init_kwargs)
        if complete_inspect:
            assert isinstance(cfg, ITConfig)
            full_composition_classes = [cfg_cls.__name__]
            if expected_composition_classes:
                assert set(expected_composition_classes) == set(cfg._composed_classes)
                full_composition_classes += [c.__name__ for c in cfg._composed_classes]
            expected_repr = TestClassBaseConfigs.expected_composition_states[tuple(sorted(full_composition_classes))]
            assert repr(cfg).startswith(expected_repr)

    def test_search_candidate_subclass_equal_candidates(self):
        module1 = Mock(name='Module1')
        module2 = Mock(name='Module2')
        candidate_modules = {Mock(name='Adapter1'): module1, Mock(name='Adapter2'): module2,}
        kwargs_not_in_target_type = {'attr1': 'value1','attr2': 'value2'}
        with patch('interpretune.config.shared.collect_exhaustive_attr_set',
                   return_value={'attr1', 'attr2', 'attr3'}):
            result = search_candidate_subclass_attrs(candidate_modules, kwargs_not_in_target_type)
        assert result == (module1,)

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
