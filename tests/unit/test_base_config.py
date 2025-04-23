from copy import deepcopy

import pytest
from unittest.mock import Mock, patch

from interpretune.config import (ITDataModuleConfig, ITLensFromPretrainedConfig, ITLensConfig,
                                 SAELensFromPretrainedConfig, SAELensConfig, ITConfig, AutoCompConfig,
                                 search_candidate_subclass_attrs, HFFromPretrainedConfig, AnalysisCfg,
                                 AnalysisArtifactCfg)
from interpretune.analysis import SAEAnalysisTargets, OpSchema
from interpretune.analysis.ops.dispatcher import DISPATCHER
from it_examples.experiments.rte_boolq import (RTEBoolqEntailmentMapping, GenerativeClassificationConfig,
                                               RTEBoolqSLConfig)
from interpretune.config.transformer_lens import TLensGenerationConfig
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


class TestAnalysisConfigs:

    def test_analysis_cfg_post_init(self):
        # Test without op
        cfg = AnalysisCfg(name="test_analysis")
        assert cfg.name == "test_analysis"

        # Test with string op that doesn't exist
        with pytest.raises(ValueError, match="Unknown operation"):
            AnalysisCfg(op="nonexistent_op")

        # Test with chained op string that doesn't exist
        with pytest.raises(ValueError):
            AnalysisCfg(op="nonexistent_op1.nonexistent_op2")

        # Test with valid op and name inference
        with patch.object(DISPATCHER, 'get_op', return_value=Mock(name='test_op', alias='test_op_alias')):
            cfg = AnalysisCfg(op="test_op")
            assert cfg.name == "test_op_alias"

        # Test with list of ops
        mock_op1 = Mock(name='op1', alias='op1_alias')
        mock_op2 = Mock(name='op2', alias='op2_alias')
        with patch.object(DISPATCHER, 'create_chain_from_ops', return_value=Mock(alias='chain_alias')) as mock_chain:
            cfg = AnalysisCfg(op=[mock_op1, mock_op2])
            mock_chain.assert_called_once_with([mock_op1, mock_op2])
            assert cfg.name == "chain_alias"

        # Test with single op in list
        with patch.object(DISPATCHER, 'create_chain_from_ops') as mock_chain:
            cfg = AnalysisCfg(op=[mock_op1])
            mock_chain.assert_not_called()
            assert cfg.op == mock_op1

        # Test chained operations with dot notation
        with patch.object(DISPATCHER, 'create_chain', return_value=Mock(alias='dotted_chain_alias')) as mock_chain:
            cfg = AnalysisCfg(op="op1.op2")
            mock_chain.assert_called_once_with("op1.op2")
            assert cfg.name == "dotted_chain_alias"

    def test_analysis_cfg_update(self):
        cfg = AnalysisCfg(name="test_analysis")
        cfg.update(name="updated_name", save_prompts=True)
        assert cfg.name == "updated_name"
        assert cfg.save_prompts is True

    def test_analysis_cfg_materialize_names_filter(self):
        mock_module = Mock()
        mock_sae_targets = Mock(spec=SAEAnalysisTargets)
        mock_sae_targets.target_layers = ["layer1", "layer2"]
        mock_sae_targets.sae_hook_match_fn = lambda x: True

        # Test with already set names_filter
        cfg = AnalysisCfg(names_filter=lambda x: x.endswith("test"))
        with patch('interpretune.config.analysis.resolve_names_filter',
                   return_value=lambda x: x.endswith("resolved")):
            cfg.materialize_names_filter(mock_module)
            assert cfg.names_filter("test_resolved")

        # Test with sae_analysis_targets
        cfg = AnalysisCfg(sae_analysis_targets=mock_sae_targets)
        mock_module = Mock()  # Create a new mock to avoid call count issues
        mock_module.construct_names_filter.return_value = lambda x: x.startswith("constructed")
        with patch('interpretune.config.analysis.resolve_names_filter',
                   return_value=lambda x: x.startswith("constructed_resolved")):
            cfg.materialize_names_filter(mock_module)
            mock_module.construct_names_filter.assert_called_once_with(
                mock_sae_targets.target_layers, mock_sae_targets.sae_hook_match_fn)
            assert cfg.names_filter("constructed_resolved_test")

        # Test with fallback sae_targets
        cfg = AnalysisCfg()
        mock_module = Mock()  # Create a new mock to avoid call count issues
        mock_fallback = Mock(spec=SAEAnalysisTargets)
        mock_fallback.target_layers = ["fallback1", "fallback2"]
        mock_fallback.sae_hook_match_fn = lambda x: True
        mock_module.construct_names_filter.return_value = lambda x: x.startswith("fallback")
        with patch('interpretune.config.analysis.resolve_names_filter',
                   return_value=lambda x: x.startswith("fallback_resolved")):
            cfg.materialize_names_filter(mock_module, fallback_sae_targets=mock_fallback)
            mock_module.construct_names_filter.assert_called_once_with(
                mock_fallback.target_layers, mock_fallback.sae_hook_match_fn)
            assert cfg.names_filter("fallback_resolved_test")

        # Test with no targets
        cfg = AnalysisCfg()
        mock_module = Mock()  # Create a new mock to avoid call count issues
        with pytest.raises(ValueError, match="No SAEAnalysisTargets available"):
            cfg.materialize_names_filter(mock_module)

    def test_analysis_cfg_maybe_set_hooks(self):
        # Test with no hooks
        cfg = AnalysisCfg(op=Mock())
        with patch.object(cfg, 'check_add_default_hooks') as mock_check:
            cfg.maybe_set_hooks()
            mock_check.assert_called_once()

        # Test with existing hooks
        cfg = AnalysisCfg(op=Mock(), fwd_hooks=[(lambda x: True, lambda x: x)])
        with patch.object(cfg, 'check_add_default_hooks') as mock_check:
            cfg.maybe_set_hooks()
            mock_check.assert_not_called()

    def test_analysis_cfg_prepare_model_ctx(self):
        mock_module = Mock()
        mock_op = Mock()

        # Test with op
        cfg = AnalysisCfg(op=mock_op)
        with patch.object(cfg, 'materialize_names_filter') as mock_material, \
             patch.object(cfg, 'maybe_set_hooks') as mock_hooks:
            cfg.prepare_model_ctx(mock_module)
            mock_material.assert_called_once_with(mock_module, None)
            mock_hooks.assert_called_once()

        # Test without op
        cfg = AnalysisCfg()
        with patch.object(cfg, 'materialize_names_filter') as mock_material, \
             patch.object(cfg, 'maybe_set_hooks') as mock_hooks:
            cfg.prepare_model_ctx(mock_module)
            mock_material.assert_called_once_with(mock_module, None)
            mock_hooks.assert_not_called()

    # def test_analysis_cfg_reset_analysis_store(self, request):
    #     """Test reset_analysis_store using a real AnalysisStore from a fixture."""
    #     # Get a real AnalysisStore from a fixture
    #     fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")
    #     real_store = deepcopy(fixture.result)

    #     # Create an AnalysisCfg with the real store
    #     cfg = AnalysisCfg(output_store=real_store)

    #     # Store some values to check they're preserved
    #     original_save_cfg = real_store.save_cfg
    #     original_save_cfg_class = original_save_cfg.__class__
    #     original_dict_items = {k: v for k, v in original_save_cfg.__dict__.items()
    #                            if k != 'output_store'}

    #     # Run the reset method
    #     cfg.reset_analysis_store()

    #     # Verify the store was reset properly
    #     assert cfg.output_store is not None
    #     assert cfg.output_store is real_store  # Should be the same object
    #     assert cfg.output_store.save_cfg.__class__ == original_save_cfg_class

    #     # Check that non-output_store attributes were preserved
    #     for key, value in original_dict_items.items():
    #         assert cfg.output_store.save_cfg.__dict__[key] == value

    #     # Check that the output_store reference is properly set
    #     assert id(cfg.output_store) == id(cfg.output_store.save_cfg.output_store)

    def test_analysis_cfg_save_batch(self):
        mock_batch = Mock()
        mock_analysis_batch = Mock()
        mock_tokenizer = Mock()

        # Test with op
        mock_op = Mock()
        cfg = AnalysisCfg(op=mock_op)
        list(cfg.save_batch(mock_analysis_batch, mock_batch, tokenizer=mock_tokenizer))
        mock_op.save_batch.assert_called_once_with(
            mock_analysis_batch, mock_batch, tokenizer=mock_tokenizer,
            save_prompts=False, save_tokens=False, decode_kwargs=cfg.decode_kwargs
        )

        # Test without op
        cfg = AnalysisCfg(output_schema=OpSchema({}))
        with patch('interpretune.analysis.AnalysisOp.process_batch') as mock_process:
            list(cfg.save_batch(mock_analysis_batch, mock_batch, tokenizer=mock_tokenizer))
            mock_process.assert_called_once_with(
                mock_analysis_batch, mock_batch, output_schema=cfg.output_schema,
                tokenizer=mock_tokenizer, save_prompts=False, save_tokens=False,
                decode_kwargs=cfg.decode_kwargs
            )

    def test_analysis_cfg_add_default_cache_hooks(self):
        mock_filter = lambda x: True

        # Test with include_backward=True
        cfg = AnalysisCfg(names_filter=mock_filter, cache_dict={})
        with patch('interpretune.config.analysis._make_simple_cache_hook') as mock_hook:
            mock_hook.return_value = lambda x: x
            # Method doesn't return hooks, it sets them on the instance
            cfg.add_default_cache_hooks()
            assert len(cfg.fwd_hooks) == 1
            assert len(cfg.bwd_hooks) == 1
            assert mock_hook.call_count == 2

        # Test with include_backward=False
        cfg = AnalysisCfg(names_filter=mock_filter, cache_dict={})
        with patch('interpretune.config.analysis._make_simple_cache_hook') as mock_hook:
            mock_hook.return_value = lambda x: x
            # Method doesn't return hooks, it sets them on the instance
            cfg.add_default_cache_hooks(include_backward=False)
            assert len(cfg.fwd_hooks) == 1
            assert len(cfg.bwd_hooks) == 0
            assert mock_hook.call_count == 1

    def test_analysis_cfg_check_add_default_hooks(self):
        # Test with no op
        cfg = AnalysisCfg()
        cfg.check_add_default_hooks()
        assert cfg.fwd_hooks == []
        assert cfg.bwd_hooks == []

        # Test with logit_diffs_base op
        mock_op = Mock()
        mock_op.name = 'logit_diffs_base'  # Use attribute assignment instead of name parameter
        cfg = AnalysisCfg(op=mock_op)
        # This should return without adding hooks, not raise an exception
        cfg.check_add_default_hooks()
        assert not hasattr(cfg, 'fwd_hooks') or cfg.fwd_hooks == []
        assert not hasattr(cfg, 'bwd_hooks') or cfg.bwd_hooks == []

        # Test with logit_diffs_attr_grad op but no names_filter
        mock_op = Mock()
        mock_op.name = 'logit_diffs_attr_grad'  # Use attribute assignment
        cfg = AnalysisCfg(op=mock_op)
        with pytest.raises(ValueError, match="names_filter required"):
            cfg.check_add_default_hooks()

        # Test with logit_diffs_attr_grad op with names_filter
        mock_op = Mock()
        mock_op.name = 'logit_diffs_attr_grad'  # Use attribute assignment
        cfg = AnalysisCfg(op=mock_op, names_filter=lambda x: True, cache_dict={})
        with patch.object(cfg, 'add_default_cache_hooks') as mock_add:
            cfg.check_add_default_hooks()
            mock_add.assert_called_once()

    def test_analysis_cfg_applied_to(self):
        mock_module = Mock()
        original_id = id(mock_module)

        # Test not applied
        cfg = AnalysisCfg()
        assert cfg.applied_to(mock_module) is False

        # Test applied
        cfg._applied_to[original_id] = mock_module.__class__.__name__
        assert cfg.applied_to(mock_module) is True

    def test_analysis_cfg_reset_applied_state(self):
        mock_module1 = Mock()
        mock_module2 = Mock()
        id1 = id(mock_module1)
        id2 = id(mock_module2)

        # Set up applied state
        cfg = AnalysisCfg()
        cfg._applied_to[id1] = mock_module1.__class__.__name__
        cfg._applied_to[id2] = mock_module2.__class__.__name__

        # Test reset specific module
        cfg.reset_applied_state(mock_module1)
        assert id1 not in cfg._applied_to
        assert id2 in cfg._applied_to

        # Test reset all
        cfg._applied_to[id1] = mock_module1.__class__.__name__  # Re-add module1
        cfg.reset_applied_state()
        assert len(cfg._applied_to) == 0

    # def test_analysis_cfg_apply(self, request):
    #     """Test the apply method with both mocked and real components."""
    #     # Get a real fixture with AnalysisCfg
    #     fixture = request.getfixturevalue("get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis")

    #     # Test with already applied
    #     mock_module = Mock()
    #     mock_module_id = id(mock_module)

    #     cfg = AnalysisCfg()
    #     cfg._applied_to[mock_module_id] = mock_module.__class__.__name__
    #     with patch.object(cfg, 'prepare_model_ctx') as mock_prepare:
    #         cfg.apply(mock_module)
    #         mock_prepare.assert_not_called()

    #     # Reset applied state for next tests
    #     cfg._applied_to.clear()

    #     # Test with custom step and no ignore_manual
    #     mock_module = Mock()
    #     mock_module_id = id(mock_module)
    #     setattr(mock_module, 'analysis_step', lambda x: x)
    #     cfg = AnalysisCfg(op=Mock())

    #     # Directly patch warnings.warn at the module level
    #     with patch('warnings.warn') as mock_warn, \
    #          patch.object(cfg, 'prepare_model_ctx') as mock_prepare:
    #         cfg.apply(mock_module)
    #         mock_warn.assert_called_once()
    #         assert "already has a analysis_step method" in mock_warn.call_args[0][0]
    #         mock_prepare.assert_called_once()
    #         assert mock_module_id in cfg._applied_to

    #     # Test with custom step and ignore_manual=True
    #     mock_module = Mock()
    #     mock_module_id = id(mock_module)
    #     setattr(mock_module, 'analysis_step', lambda x: x)
    #     mock_op = Mock()
    #     cfg = AnalysisCfg(op=mock_op, ignore_manual=True)
    #     with patch.object(cfg, 'prepare_model_ctx') as mock_prepare:
    #         cfg.apply(mock_module)
    #         mock_prepare.assert_called_once()
    #         assert hasattr(mock_module, '_generated_analysis_step')
    #         assert mock_module_id in cfg._applied_to

    #     # Test with no custom step and with op
    #     mock_module = Mock(spec=['__class__'])
    #     mock_module.__class__.__name__ = 'MockModule'
    #     mock_module_id = id(mock_module)
    #     mock_op = Mock()
    #     cfg = AnalysisCfg(op=mock_op)
    #     with patch.object(cfg, 'prepare_model_ctx') as mock_prepare:
    #         cfg.apply(mock_module)
    #         mock_prepare.assert_called_once()
    #         assert hasattr(mock_module, 'analysis_step')
    #         assert mock_module_id in cfg._applied_to

    #     # Test without output_store - use a real AnalysisStore to avoid mocking
    #     mock_module = Mock()
    #     mock_module_id = id(mock_module)
    #     cfg = AnalysisCfg()

    #     # Import from the actual module path to match what's used in the code
    #     with patch('interpretune.analysis.core.AnalysisStore') as mock_store_cls, \
    #          patch.object(cfg, 'prepare_model_ctx') as mock_prepare:
    #         mock_store_cls.return_value = fixture.result
    #         cfg.apply(mock_module)
    #         mock_store_cls.assert_called_once()
    #         mock_prepare.assert_called_once()
    #         assert mock_module_id in cfg._applied_to

    #     # Test with output_store and cache_dir/output_path - use real AnalysisStore
    #     mock_module = Mock()
    #     mock_module_id = id(mock_module)
    #     real_output_store = deepcopy(fixture.result)
    #     real_output_store.cache_dir = "/original/cache"
    #     real_output_store.op_output_dataset_path = "/original/output"
    #     cfg = AnalysisCfg(output_store=real_output_store)
    #     with patch('interpretune.config.analysis.rank_zero_warn') as mock_warn, \
    #          patch.object(cfg, 'prepare_model_ctx') as mock_prepare:
    #         cfg.apply(mock_module, cache_dir="/new/cache", op_output_dataset_path="/new/output")
    #         mock_warn.assert_called_once()
    #         mock_prepare.assert_called_once()
    #         assert mock_module_id in cfg._applied_to

    def test_analysis_artifact_cfg(self):
        # Test default initialization
        cfg = AnalysisArtifactCfg()
        assert cfg.latent_effects_graphs is True
        assert cfg.latent_effects_graphs_per_batch is False
        assert cfg.latents_table_per_sae is True

        # Test post_init forcing latent_effects_graphs to True
        with patch('builtins.print') as mock_print:
            cfg = AnalysisArtifactCfg(latent_effects_graphs=False, latent_effects_graphs_per_batch=True)
            assert cfg.latent_effects_graphs is True
            mock_print.assert_called_once()
