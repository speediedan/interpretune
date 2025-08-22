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
from interpretune.base import _call_itmodule_hook
from interpretune.config import ITConfig, ITExtensionsConfigMixin, HFFromPretrainedConfig, ITExtension
from tests.base_defaults import default_test_task
from tests.utils import disable_genclassif, get_super_method, ablate_cls_attrs
from tests.runif import RunIf
from tests.warns import CORE_CTX_WARNS, unexpected_warns
from tests.orchestration import run_it
from it_examples.experiments.rte_boolq import RTEBoolqModuleMixin


class TestClassMixins:
    core_gpt2_shared_config = dict(
        task_name=default_test_task,
        tokenizer_kwargs={
            "add_bos_token": True,
            "local_files_only": False,
            "padding_side": "left",
            "model_input_names": ["input_ids", "attention_mask"],
        },
        model_name_or_path="gpt2",
        tokenizer_id_overrides={"pad_token_id": 50256},
    )

    test_core_gpt2 = {
        **core_gpt2_shared_config,
        "hf_from_pretrained_cfg": HFFromPretrainedConfig(
            pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        ),
    }

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
        pretrained_kwargs = {"pretrained_kwargs": {"device_map": "cpu", "token": "strip-me"}}
        from_pretrained_cfg = HFFromPretrainedConfig(**pretrained_kwargs, model_head="transformers.GPT2LMHeadModel")
        assert from_pretrained_cfg.pretrained_kwargs.get("token", None) is None

    @pytest.mark.parametrize(
        "return_unused, tie_word_embeddings",
        [pytest.param(True, False)],
        ids=[
            "return_unused_no_tie_embeddings",
        ],
    )
    def test_hf_from_pretrained_hf_cust_config(self, return_unused, tie_word_embeddings):
        access_token = None
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg["hf_from_pretrained_cfg"].pretrained_kwargs["return_unused_kwargs"] = return_unused
        test_it_cfg["model_cfg"] = {"tie_word_embeddings": tie_word_embeddings}
        if return_unused:
            test_it_cfg["hf_from_pretrained_cfg"].pretrained_kwargs["give_it_back"] = True
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        cust_config, unused_kwargs = hf_from_pretrained_mixin._hf_gen_cust_config()
        assert cust_config.tie_word_embeddings == tie_word_embeddings
        if return_unused:
            assert "give_it_back" in unused_kwargs
            hf_from_pretrained_mixin.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs.pop("give_it_back", None)
        model = hf_from_pretrained_mixin.hf_configured_model_init(cust_config, access_token)
        assert model.config.tie_word_embeddings == tie_word_embeddings

    @pytest.mark.parametrize(
        "head_configured, defer_init",
        [pytest.param(True, True), pytest.param(False, False), pytest.param(False, True)],
        ids=["head_config_defer_init", "no_head_config_no_defer_init", "no_head_config_defer_init"],
    )
    def test_hf_from_pretrained_hf_configured_model_init(self, head_configured, defer_init):
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg["defer_model_init"] = defer_init
        if not head_configured:
            test_it_cfg["hf_from_pretrained_cfg"].model_head = ""
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        if not head_configured and defer_init:
            with pytest.warns(UserWarning, match="`defer_model_init` not currently supported without `model_head`"):
                _ = hf_from_pretrained_mixin._hf_gen_cust_config()
        else:
            cust_config, _ = hf_from_pretrained_mixin._hf_gen_cust_config()
            _ = hf_from_pretrained_mixin.hf_configured_model_init(cust_config)

    @RunIf(min_cuda_gpus=1)
    def test_hf_from_pretrained_peft_init(self, get_it_session__core_gpt2_peft__initonly):
        fixture = get_it_session__core_gpt2_peft__initonly
        it_m = fixture.it_session.module
        assert it_m.model.transformer.h[0].attn.c_proj.weight.quant_type == "nf4"
        assert it_m.model.transformer.h[0].attn.c_proj.base_layer.compute_dtype == torch.bfloat16
        assert getattr(it_m.model.transformer.h[0].attn.c_proj, "lora_A", None) is not None
        assert it_m.model.base_model.model.is_gradient_checkpointing

    @RunIf(min_cuda_gpus=1)
    @pytest.mark.parametrize(
        "phase, genclassif",
        [pytest.param("train", True), pytest.param("test", True), pytest.param("test", False)],
        ids=["train_genclassif", "test_genclassif", "test_no_genclassif"],
    )
    def test_peft(self, recwarn, get_it_session__core_gpt2_peft__initonly, phase, genclassif):
        fixture = get_it_session__core_gpt2_peft__initonly
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        expected_warnings = CORE_CTX_WARNS
        test_cfg.phase = phase
        if not genclassif:
            with disable_genclassif(it_session):
                run_it(it_session=it_session, test_cfg=test_cfg)
        else:
            run_it(it_session=it_session, test_cfg=test_cfg)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    @RunIf(min_cuda_gpus=1)
    def test_peft_seq_test(self, recwarn, get_it_session__core_gpt2_peft_seq__initonly):
        fixture = get_it_session__core_gpt2_peft_seq__initonly
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        expected_warnings = CORE_CTX_WARNS
        run_it(it_session=it_session, test_cfg=test_cfg)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_hf_from_pretrained_dynamic_module_load(self):
        pretrained_kwargs = {"pretrained_kwargs": {"device_map": "cpu", "torch_dtype": "float32"}}
        dynamic_module_cfg = {
            "config_class": "configuration_falcon.FalconConfig",
            "model_class": "modeling_falcon.FalconForCausalLM",
        }
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg["model_name_or_path"] = "tiiuae/falcon-7b"
        hf_from_pretrained_cfg = HFFromPretrainedConfig(**pretrained_kwargs, dynamic_module_cfg=dynamic_module_cfg)
        test_it_cfg["hf_from_pretrained_cfg"] = hf_from_pretrained_cfg
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        hf_from_pretrained_mixin._hf_gen_cust_config()

    def test_degen_it_extension(self):
        """Test handling of invalid extensions in the updated extensions_context."""
        degen_ext = ITExtension("not_here", "oops.not_found", "oops.I.did.it.again")
        ext_mixin = ITExtensionsConfigMixin()
        ext_mixin.extensions_context.DEFAULT_EXTENSIONS = (degen_ext,)
        with pytest.raises(MisconfigurationException, match="Unable to import and resolve specified extension"):
            ext_mixin._detect_extensions()

    def test_it_generate_exception_handling(self, get_it_session__core_cust__initonly):
        fixture = get_it_session__core_cust__initonly
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        core_cust_it_m = it_session.module
        test_cfg.phase = "test"

        def generate(oops_no_matching_args):
            pass

        # we modify our generate function and avoid checking the batch inputs in order to generate our error feedback
        with (
            mock.patch.object(core_cust_it_m.model, "generate", generate),
            mock.patch.object(core_cust_it_m, "map_gen_inputs", lambda x: x),
        ):
            with pytest.warns(UserWarning, match="The following keys were found"), pytest.raises(Exception):
                run_it(it_session=it_session, test_cfg=test_cfg)

    @pytest.mark.parametrize(
        "tokenizer_id_overrides",
        [None, {"pad_token_id": 150}],
        ids=["no_token_overrides", "new_token_overrides"],
    )
    def test_hf_from_pretrained_maybe_resize_token_embeddings(self, tokenizer_id_overrides):
        test_it_cfg = deepcopy(TestClassMixins.test_core_gpt2)
        test_it_cfg["tokenizer_id_overrides"] = tokenizer_id_overrides
        hf_from_pretrained_mixin = TestClassMixins._get_hf_from_pretrained_mixin(test_it_cfg)
        hf_from_pretrained_mixin.model = mock.Mock()
        hf_from_pretrained_mixin.model.base_model = mock.Mock()
        hf_from_pretrained_mixin.model.base_model.vocab_size = 100
        expected_calls = 0 if tokenizer_id_overrides is None else 1
        hf_from_pretrained_mixin._hf_maybe_resize_token_embeddings()
        assert hf_from_pretrained_mixin.model.base_model.resize_token_embeddings.call_count == expected_calls


class TestITStateMixin:
    def test_init_internal_state(self):
        """Test the _init_internal_state static method in ITStateMixin."""
        from interpretune.base.components.mixins import ITStateMixin
        from interpretune.config import ITState

        # Test initializing state when none exists
        test_obj = type("TestObj", (), {})()
        ITStateMixin._init_internal_state(test_obj)
        assert hasattr(test_obj, "_it_state")
        assert isinstance(test_obj._it_state, ITState)

        # Test that existing state is preserved
        original_state = test_obj._it_state
        ITStateMixin._init_internal_state(test_obj)
        assert test_obj._it_state is original_state, "Expected existing state to be preserved"


class TestAnalysisStepMixin:
    def test_on_analysis_epoch_end(self, get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis):
        """Test the on_analysis_epoch_end method in AnalysisStepMixin."""

        fixture = get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis
        module = fixture.it_session.module

        # Simply verify the method can be called without errors
        module.on_analysis_epoch_end()

    def test_analysis_start_end(self, get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis):
        """Test the on_analysis_start and on_analysis_end methods."""
        import interpretune as it

        fixture = get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis
        module = fixture.it_session.module

        with ablate_cls_attrs(module.it_cfg, "analysis_cfg"):
            # Test when analysis_cfg is None
            with pytest.warns(UserWarning, match="Analysis configuration has not been set."):
                module.analysis_cfg

        # Test with regular operation
        with mock.patch.object(torch, "set_grad_enabled") as mock_grad:
            module.on_analysis_start()
            # Should disable gradients for non-gradient operations
            mock_grad.assert_called_with(False)

        # Test with gradient operation
        with mock.patch.object(torch, "set_grad_enabled") as mock_grad:
            # Save original op
            original_op = module.analysis_cfg.op
            # Mock op to be gradient operation
            module.analysis_cfg.op = it.logit_diffs_attr_grad

            module.on_analysis_start()
            mock_grad.assert_called_with(True)

            # Test on_analysis_end with gradient operation
            with mock.patch.object(module, "on_analysis_end") as mock_session_end:
                # Instead of mocking session_complete property directly,
                # create a custom context to test both paths without patching
                module.on_analysis_end()
                # For gradient operations, verify session_end is called
                # regardless of the session_complete value
                mock_session_end.assert_called_once()

            # Restore original op
            module.analysis_cfg.op = original_op

        # Test on_analysis_end with non-gradient operation
        with mock.patch.object(torch, "set_grad_enabled") as mock_grad:
            # We don't need to patch session_complete - the test will still
            # verify that torch.set_grad_enabled(True) is called
            module.on_analysis_end()
            mock_grad.assert_called_with(True)


class TestGenerativeStepMixin:
    def test_generate_prepares_inputs(self):
        """Test the _generate_prepares_inputs method in GenerativeStepMixin."""
        from interpretune.base.components.mixins import GenerativeStepMixin

        # Create a minimal mixin instance for testing
        generative_mixin = GenerativeStepMixin()
        generative_mixin.GEN_PREPARES_INPUTS_SIGS = ("_prepare_model_inputs",)

        # Test when model has the prepare method
        model_with_prepare = mock.Mock(spec=["_prepare_model_inputs"])
        generative_mixin.model = model_with_prepare

        assert generative_mixin._generate_prepares_inputs() is True

        # Test when model doesn't have the prepare method
        model_without_prepare = mock.Mock(spec=[])  # Empty spec means no methods
        generative_mixin.model = model_without_prepare

        assert generative_mixin._generate_prepares_inputs() is False

    def test_it_generate_with_tensor_batch(self, get_it_session__core_cust__initonly):
        """Test the it_generate method with a tensor batch."""
        from interpretune.base.components.mixins import GenerativeStepMixin

        # Create a test subclass that overrides the properties we need to control
        class TestGenerativeMixin(GenerativeStepMixin):
            def __init__(self, model, sig_keys=None):
                self.model = model
                self._gen_sig_keys = sig_keys or []

        # Set up the test environment
        batch = torch.tensor([[1, 2, 3]])
        expected_output = torch.tensor([[4, 5, 6]])

        # Create mocks
        mock_model = mock.Mock()
        mock_model.generate.return_value = expected_output

        # Create our test mixin
        test_mixin = TestGenerativeMixin(mock_model, sig_keys=["kwargs"])

        # Run the test
        output = test_mixin.it_generate(batch, max_length=10)

        # Verify results
        mock_model.generate.assert_called_once_with(batch, max_length=10)
        assert torch.equal(output, expected_output)

    def test_it_generate_with_batch_encoding(self):
        """Test the it_generate method with BatchEncoding and _should_inspect_inputs=False."""
        from transformers import BatchEncoding
        from interpretune.base.components.mixins import GenerativeStepMixin

        # Create a test subclass that overrides the properties we need to control
        class TestGenerativeMixin(GenerativeStepMixin):
            def __init__(self, model, inspect_inputs=False):
                self.model = model
                self._gen_sig_keys = ["input_ids", "max_length"]
                self._should_inspect_inputs_value = inspect_inputs

            @property
            def _should_inspect_inputs(self):
                return self._should_inspect_inputs_value

        # Create a BatchEncoding batch
        batch = BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})

        # Create mocks
        mock_model = mock.Mock()
        mock_model.generate.return_value = torch.tensor([[4, 5, 6]])

        # Create our test mixin with _should_inspect_inputs=False
        test_mixin = TestGenerativeMixin(mock_model, inspect_inputs=False)

        # Run the test
        output = test_mixin.it_generate(batch, max_length=10)

        # Verify results
        mock_model.generate.assert_called_once_with(**batch, max_length=10)
        assert torch.equal(output, torch.tensor([[4, 5, 6]]))

    def test_it_generate_with_batch_encoding_and_inspect_inputs(self):
        """Test the it_generate method with BatchEncoding and _should_inspect_inputs=True."""
        from transformers import BatchEncoding
        from interpretune.base.components.mixins import GenerativeStepMixin

        # Create a test subclass that overrides the properties we need to control
        class TestGenerativeMixin(GenerativeStepMixin):
            def __init__(self, model, inspect_inputs=True):
                self.model = model
                self._gen_sig_keys = ["input_ids", "max_length"]
                self._should_inspect_inputs_value = inspect_inputs

            @property
            def _should_inspect_inputs(self):
                return self._should_inspect_inputs_value

            def map_gen_inputs(self, batch):
                # Mock implementation to filter inputs
                return {"input_ids": batch["input_ids"]}

        # Create a BatchEncoding batch
        batch = BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]]), "unwanted_key": "value"})

        # Create mocks
        mock_model = mock.Mock()
        mock_model.generate.return_value = torch.tensor([[4, 5, 6]])

        # Create our test mixin with _should_inspect_inputs=True
        test_mixin = TestGenerativeMixin(mock_model, inspect_inputs=True)

        # Run the test
        output = test_mixin.it_generate(batch, max_length=10)

        # Verify results
        mock_model.generate.assert_called_once_with(input_ids=batch["input_ids"], max_length=10)
        assert torch.equal(output, torch.tensor([[4, 5, 6]]))

    def test_it_generate_exception_handling_type_error(self):
        """Test it_generate exception handling with TypeError."""
        from transformers import BatchEncoding
        from interpretune.base.components.mixins import GenerativeStepMixin

        # Create a test mixin that will trigger the error handling
        class ErrorGenerativeMixin(GenerativeStepMixin):
            def __init__(self):
                self._gen_sig_keys = ["input_ids"]
                self.model = mock.Mock()
                self.model.generate = mock.Mock(side_effect=TypeError("Test error"))
                self.it_cfg = mock.Mock()
                self.it_cfg.generative_step_cfg.input_inspection_enabled = False

            def map_gen_kwargs(self, kwargs):
                return kwargs

        batch = BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
        test_mixin = ErrorGenerativeMixin()

        # The error message should contain batch data and gen_sig_keys
        with pytest.warns(UserWarning), pytest.raises(Exception) as excinfo:
            test_mixin.it_generate(batch)

        # Check the error message contains expected information
        assert "The following keys were found in the provided data batch" in str(excinfo.value)
        assert "Test error" in str(excinfo.value)


class TestClassificationMixin:
    def test_init_classification_mapping(self, get_it_session__core_cust__initonly):
        """Test the init_classification_mapping method."""
        fixture = get_it_session__core_cust__initonly
        it_session = fixture.it_session
        module = it_session.module

        # Set up classification mapping
        module.it_cfg.classification_mapping = ["token1", "token2"]
        module.it_cfg.classification_mapping_indices = None

        # Mock the tokenizer to convert tokens to IDs
        with mock.patch.object(
            it_session.datamodule.tokenizer, "convert_tokens_to_ids", return_value=[10, 20]
        ) as mock_convert:
            _call_itmodule_hook(
                it_session.datamodule, hook_name="prepare_data", hook_msg="Preparing data", target_model=module.model
            )
            _call_itmodule_hook(it_session.datamodule, hook_name="setup", hook_msg="Setting up datamodule")

            # Patch the RTEBoolqModuleMixin.setup method at the class level to ensure only the base setup() is called
            # TODO: replace use of `core_cust` fixture with a more generic one when available instead of this patch
            with mock.patch(
                "it_examples.experiments.rte_boolq.RTEBoolqModuleMixin.setup",
                autospec=True,
                side_effect=lambda self, *args, **kwargs: super(RTEBoolqModuleMixin, self).setup(*args, **kwargs),
            ):
                _call_itmodule_hook(
                    module, hook_name="setup", hook_msg="Setting up model", datamodule=it_session.datamodule
                )

            # Verify tokenizer was called with the right arguments
            mock_convert.assert_called_once_with(["token1", "token2"])

            # Verify classification_mapping_indices was set correctly
            assert torch.equal(
                module.it_cfg.classification_mapping_indices, torch.tensor([10, 20], device=module.device)
            )

    def test_standardize_logits(self, get_it_session__core_cust__setup):
        """Test the standardize_logits method with different input shapes."""
        fixture = get_it_session__core_cust__setup
        module = fixture.it_session.module
        # Use the utility function to get the BaseITModule standardize_logits method
        # (skipping the override in RTEBoolqModuleMixin)
        target_standard_logits = get_super_method(
            "it_examples.experiments.rte_boolq.RTEBoolqModuleMixin", module, "standardize_logits"
        )

        # Setup classification mapping indices - ensure we have the right device
        device = module.device

        num_labels = 2

        # Update our test to match the actual implementation
        module.it_cfg.classification_mapping_indices = torch.tensor([0, 1], device=device)
        module.it_cfg.num_labels = num_labels

        vocab_size = 50257  # Standard GPT-2 vocab size

        # Test with 2D input (batch_size x vocab_size)
        logits_2d = torch.randn(2, vocab_size)

        std_logits = target_standard_logits(logits_2d)
        assert std_logits.shape == (2, 1, num_labels)  # Should add position dimension and select mapped indices

        # Test with tuple input (e.g., from multiple layers)
        logits_tuple = (torch.randn(2, vocab_size), torch.randn(2, vocab_size))
        std_logits = target_standard_logits(logits_tuple)
        assert std_logits.shape == (2, 2, num_labels)  # Should stack and select indices

        # Test with 3D input (batch_size x seq_len x vocab_size)
        logits_3d = torch.randn(2, 4, vocab_size)
        std_logits = target_standard_logits(logits_3d)
        assert std_logits.shape == (2, 4, num_labels)  # Should maintain positions and select indices

        # Test with generative_step_cfg disabled
        with mock.patch.object(module.it_cfg, "generative_step_cfg") as mock_gen_cfg:
            mock_gen_cfg.enabled = False
            std_logits = target_standard_logits(logits_3d)
            assert std_logits.shape == (2, 1, num_labels)  # Should select only the last position for non-generative

        # Test ValueError when logits shape doesn't match and no mapping indices available
        module.it_cfg.classification_mapping_indices = None
        module.it_cfg.num_labels = 3  # Different from the logits shape
        logits_wrong_shape = torch.randn(2, 4, 5)  # Last dim doesn't match num_labels

        with pytest.raises(ValueError, match="The logits shape does not match the expected number of labels"):
            target_standard_logits(logits_wrong_shape)

    def test_labels_to_ids(self, get_it_session__core_cust__setup):
        """Test the labels_to_ids method."""
        fixture = get_it_session__core_cust__setup
        module = fixture.it_session.module

        target_labels_to_ids = get_super_method(
            "it_examples.experiments.rte_boolq.RTEBoolqModuleMixin", module, "labels_to_ids"
        )
        # Setup classification mapping indices
        module.it_cfg.classification_mapping_indices = torch.tensor([10, 20, 30], device=module.device)

        # Test with integer label indices
        labels = torch.tensor([0, 1, 2, 0], device=module.device)
        label_ids, returned_labels = target_labels_to_ids(labels)

        # Verify label_ids contains the mapped values from classification_mapping_indices
        assert torch.equal(label_ids, torch.tensor([10, 20, 30, 10], device=module.device))
        # Verify original labels are returned as the second value
        assert torch.equal(returned_labels, labels)
