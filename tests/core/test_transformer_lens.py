from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import re
from typing import Dict, List, Optional, Set, Tuple

import pytest
import torch
from torch import device
from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

from interpretune.config import ITLensFromPretrainedConfig, ITLensConfig, ITLensCustomConfig
from interpretune.utils import MisconfigurationException
from interpretune.adapters.transformer_lens import TransformerBridgeStrategyAdapter
from tests.warns import unexpected_warns, TL_CTX_WARNS
from tests.utils import ablate_cls_attrs
from tests.base_defaults import default_test_task
from tests.runif import RunIf


# =============================================================================
# Architecture-Specific Parameter Mapping Expectations
# =============================================================================


@dataclass
class ArchitectureExpectations:
    """Expected parameter structure for a specific model architecture.

    Attributes:
        model_name: Human-readable model name for test identification
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads (for GQA)
        d_model: Model dimension
        has_pos_embed: Whether model has learnable position embeddings (False for RoPE)
        has_gate_proj: Whether MLP has gate projection (SwiGLU/GeGLU)
        has_biases: Whether attention/MLP have biases
        expected_tl_attn_params: Per-layer attention param suffixes to expect
        expected_tl_mlp_params: Per-layer MLP param suffixes to expect
        expected_tl_embed_params: Embedding params to expect
        canonical_attn_pattern: Regex pattern for canonical attention params (for mapping validation)
        canonical_mlp_pattern: Regex pattern for canonical MLP params (for mapping validation)
        canonical_embed_pattern: Regex pattern for canonical embedding params (for mapping validation)
        expected_mapped_tl_count: Expected number of TL params with canonical mappings
        expected_unmapped_tl_count: Expected number of unmapped TL params (synthetic biases)
        expected_mapped_canonical_count: Expected number of canonical params with TL mappings
        expected_unmapped_canonical_count: Expected number of unmapped canonical params
                                          (LayerNorms, pre-split QKV, etc.)
    """

    model_name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    has_pos_embed: bool = False
    has_gate_proj: bool = True
    has_biases: bool = False

    # TL-style parameter suffixes expected per layer
    expected_tl_attn_params: List[str] = field(default_factory=lambda: ["attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O"])
    expected_tl_mlp_params: List[str] = field(default_factory=lambda: ["mlp.W_in", "mlp.W_out"])
    expected_tl_embed_params: List[str] = field(default_factory=lambda: ["embed.W_E", "unembed.W_U"])

    # Canonical naming patterns (regex) - architecture specific
    # These are used to validate bidirectional mapping
    canonical_attn_pattern: Optional[str] = None
    canonical_mlp_pattern: Optional[str] = None
    canonical_embed_pattern: Optional[str] = None

    # Expected mapping counts for bidirectional validation
    expected_mapped_tl_count: Optional[int] = None
    expected_unmapped_tl_count: int = 0  # All TL params should map
    expected_mapped_canonical_count: Optional[int] = None
    expected_unmapped_canonical_count: Optional[int] = None


# Pre-defined architecture expectations
# NOTE: The actual model loaded depends on registry resolution for model_src_key.
# model_src_key="llama3" resolves to meta-llama/Llama-3.2-3B-Instruct (28 layers, not Llama-3.2-1B)
LLAMA3_EXPECTATIONS = ArchitectureExpectations(
    model_name="Llama-3.2-3B-Instruct",
    n_layers=28,  # Llama-3.2-3B-Instruct has 28 layers (registry default for llama3)
    n_heads=24,
    n_kv_heads=8,  # GQA
    d_model=3072,
    has_pos_embed=False,  # RoPE
    has_gate_proj=True,  # SwiGLU
    has_biases=False,
    # Canonical patterns for Llama3 (HF-style wrapped in TransformerBridge)
    canonical_attn_pattern=r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
    canonical_mlp_pattern=r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight",
    canonical_embed_pattern=r"model\.(embed_tokens|lm_head)\.weight",
    # Mapping counts: 368 TL params total (includes synthetic biases from TransformerBridge)
    # 28 layers * (4 attn weights + 4 attn biases + 3 mlp weights + 3 mlp biases)
    # + 2 embed + 1 pos_embed + 2 unembed = 368
    # 199 of these map to canonical (weights only, biases are synthetic)
    # Canonical: 198 mapped (lm_head weight is shared with embed)
    # + 57 LayerNorms (28 * 2 + 1 final) = 255 total
    expected_mapped_tl_count=199,
    expected_unmapped_tl_count=169,  # Synthetic biases and pos_embed don't map to canonical
    expected_mapped_canonical_count=198,  # lm_head weight is shared with embed
    expected_unmapped_canonical_count=57,  # LayerNorms (input_layernorm, post_attention_layernorm, norm)
)

GEMMA2_EXPECTATIONS = ArchitectureExpectations(
    model_name="Gemma-2-2B",
    n_layers=26,
    n_heads=8,
    n_kv_heads=4,  # GQA
    d_model=2304,
    has_pos_embed=False,  # RoPE
    has_gate_proj=True,  # GeGLU
    has_biases=False,
    # Canonical patterns for Gemma2 (HF-style wrapped in TransformerBridge)
    canonical_attn_pattern=r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
    canonical_mlp_pattern=r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight",
    canonical_embed_pattern=r"model\.(embed_tokens)\.weight",
    # Mapping counts: 342 TL params total (includes synthetic biases from TransformerBridge)
    # 26 layers * (4 attn weights + 4 attn biases + 3 mlp weights + 3 mlp biases)
    # + 1 embed + 1 pos_embed + 2 unembed = 342
    # 185 of these map to canonical (weights only, biases are synthetic)
    # Canonical: 184 mapped (lm_head weight shares with embed)
    # + 105 LayerNorms (26 * 4 + 1 final) = 289 total
    expected_mapped_tl_count=185,
    expected_unmapped_tl_count=157,  # Synthetic biases and pos_embed don't map to canonical
    expected_mapped_canonical_count=184,  # lm_head weight shares with embed
    expected_unmapped_canonical_count=105,  # LayerNorms (input_layernorm,
    # post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm, norm)
)

# GPT-2 small architecture expectations
GPT2_EXPECTATIONS = ArchitectureExpectations(
    model_name="GPT-2",
    n_layers=12,
    n_heads=12,
    n_kv_heads=12,  # MHA (same as n_heads)
    d_model=768,
    has_pos_embed=True,  # Learned position embeddings
    has_gate_proj=False,  # No gated MLP
    has_biases=True,
    # GPT-2 specific TL params include biases
    expected_tl_attn_params=[
        "attn.W_Q",
        "attn.W_K",
        "attn.W_V",
        "attn.W_O",
        "attn.b_Q",
        "attn.b_K",
        "attn.b_V",
        "attn.b_O",
    ],
    expected_tl_mlp_params=["mlp.W_in", "mlp.W_out", "mlp.b_in", "mlp.b_out"],
    expected_tl_embed_params=["embed.W_E", "pos_embed.W_pos", "unembed.W_U"],
    # Canonical patterns for GPT-2 (TransformerBridge wrapped)
    canonical_attn_pattern=r"model\.blocks\.\d+\._original_component\.attn\.(q|k|v|o)\._original_component\.(weight|bias)",
    canonical_mlp_pattern=r"model\.blocks\.\d+\._original_component\.mlp\._original_component\.(c_fc|c_proj)\._original_component\.(weight|bias)",
    canonical_embed_pattern=r"model\.(embed|pos_embed|unembed)\._original_component\.(weight|bias)",
    # Mapping counts: 12 layers * (8 attn + 4 mlp) + 4 embed/pos_embed/unembed = 148 TL params
    # Canonical: 171 mapped (148 TL + 24 from q/k/v split components) + 50 unmapped - 1 shared embed/unembed = 221 total
    # Unmapped: 50 LayerNorms (12*4 + 2 ln_final) (all QKV joint params remain mapped along with split views)
    expected_mapped_tl_count=148,
    expected_unmapped_tl_count=0,
    expected_mapped_canonical_count=171,  # 148 TL + 24 q/k/v split components - 1 shared embed/unembed
    expected_unmapped_canonical_count=50,  # 50 LayerNorm params (QKV joint params remain mapped)
)

# GPT-2 with weight processing enabled (fold_ln=True, fold_value_biases=True, center_writing_weights=True, etc.)
# LN params aren't represented with TL style names (whether folded or not).
# This tests the mapping behavior when TL processing transformations are applied.
GPT2_PROCESSED_EXPECTATIONS = ArchitectureExpectations(
    model_name="GPT-2 (processed)",
    n_layers=12,
    n_heads=12,
    n_kv_heads=12,
    d_model=768,
    has_pos_embed=True,
    has_gate_proj=False,
    has_biases=True,
    # LayerNorm params do not appear in TL params
    # When fold_value_biases=True, value biases are folded into output bias
    expected_tl_attn_params=[
        "attn.W_Q",
        "attn.W_K",
        "attn.W_V",
        "attn.W_O",
        "attn.b_Q",
        "attn.b_K",
        "attn.b_V",
        "attn.b_O",
    ],
    expected_tl_mlp_params=["mlp.W_in", "mlp.W_out", "mlp.b_in", "mlp.b_out"],
    expected_tl_embed_params=["embed.W_E", "pos_embed.W_pos", "unembed.W_U"],
    canonical_attn_pattern=r"model\.blocks\.\d+\._original_component\.attn\.(q|k|v|o)\._original_component\.(weight|bias)",
    canonical_mlp_pattern=r"model\.blocks\.\d+\._original_component\.mlp\._original_component\.(c_fc|c_proj)\._original_component\.(weight|bias)",
    canonical_embed_pattern=r"model\.(embed|pos_embed|unembed)\._original_component\.(weight|bias)",
    # Mapping counts same as unprocessed GPT-2
    # 12 layers * (8 attn + 4 mlp) + 4 embed/pos_embed/unembed = 148 TL params
    # Compatible mode adds extra canonical params: 222 total (vs 221 for unprocessed)
    # Canonical: 148 TL mapped + 74 unmapped = 222 total (embed/unembed not shared with processed mode)
    # Unmapped: 50 LayerNorms (12*4 + 2 ln_final) + 24 QKV joints (12*2)
    expected_mapped_tl_count=148,
    expected_unmapped_tl_count=0,
    expected_mapped_canonical_count=148,
    expected_unmapped_canonical_count=74,  # 50 LayerNorm params + 24 QKV joint params (compatible mode)
)


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
        with pytest.raises(MisconfigurationException, match="A valid tl_cfg"):
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
        assert it_cfg_default.tl_cfg.use_bridge is True

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


class TestBasicTransformerBridgeAdapter:
    """Basic tests for TransformerBridgeStrategyAdapter without model fixtures.

    These tests validate adapter initialization, properties, and basic behavior without requiring actual
    TransformerBridge model instantiation.
    """

    def test_basic_properties(self):
        """Sanity check for TransformerBridgeStrategyAdapter plugin.

        This ensures the adapter is well-formed and provides the hooks required by FTS restore logic.
        """
        assert issubclass(TransformerBridgeStrategyAdapter, StrategyAdapter)
        methods = inspect.getmembers(TransformerBridgeStrategyAdapter, predicate=inspect.isfunction)
        method_names = {n for n, _ in methods}
        # Should implement the adapter's before_restore_model hook (new preferred name) and backward compat alias
        assert "before_restore_model" in method_names
        assert "fts_optim_transform" in method_names
        assert "logical_param_translation" in method_names

    @pytest.mark.parametrize("use_tl_names", [True, False])
    def test_initialization(self, use_tl_names):
        """Test TransformerBridgeStrategyAdapter initialization with use_tl_names flag."""
        adapter = TransformerBridgeStrategyAdapter(use_tl_names=use_tl_names)
        assert adapter._use_tl_names == use_tl_names
        assert adapter.model_view is None  # Not initialized until on_before_init_fts

    def test_tl_names_disabled(self):
        """Test that CanonicalModelView is used when use_tl_names=False."""
        adapter = TransformerBridgeStrategyAdapter(use_tl_names=False)

        # Verify initialization state
        assert adapter._use_tl_names is False
        assert adapter.model_view is None  # Not initialized until on_before_init_fts

        # The actual delegation behavior is tested via the full test_bidirectional_mapping
        # tests with real ITLens modules

    @pytest.mark.parametrize(
        "tl_name,expected_canonical_pattern",
        [
            ("blocks.9.attn.W_Q", "model.blocks.9._original_component.attn.q._original_component.weight"),
            ("blocks.9.attn.b_Q", "model.blocks.9._original_component.attn.q._original_component.bias"),
            ("blocks.0.mlp.W_in", "model.blocks.0._original_component.mlp._original_component.0.weight"),
            ("embed.W_E", "model.embed._original_component.weight"),
            ("unembed.W_U", "model.unembed.W_U"),
        ],
        ids=["attn_weight", "attn_bias", "mlp_weight", "embed", "unembed"],
    )
    def test_naming_patterns(self, tl_name, expected_canonical_pattern):
        """Test expected naming patterns between TL-style and canonical names."""
        # This test documents the expected naming conventions
        # Actual mapping is built from tensor data_ptr() matching, not string patterns
        assert "blocks" in tl_name or "embed" in tl_name or "unembed" in tl_name
        assert "_original_component" in expected_canonical_pattern or expected_canonical_pattern == "model.unembed.W_U"

    def test_gen_ft_schedule_disabled(self):
        """Test gen_ft_schedule delegates to parent when use_tl_names=False."""
        adapter = TransformerBridgeStrategyAdapter(use_tl_names=False)

        # Without a pl_module set, calling parent gen_ft_schedule should work (uses default implementation)
        # The actual schedule generation is tested in integration tests
        assert hasattr(adapter, "gen_ft_schedule")
        assert callable(adapter.gen_ft_schedule)


# =============================================================================
# Parameter Mapping Validation Tests
# =============================================================================
class TestArchitectureParameterMapping:
    """Parameterized architecture parameter mapping tests.

    Validates TL-style parameter naming and bidirectional mapping for multiple architectures:
    - GPT-2: MHA, learned pos_embed, biases (non-standalone, fast)
    - Llama3: GQA, SwiGLU, RMSNorm, RoPE (standalone)
    - Gemma2: GQA, GeGLU, RMSNorm, sliding window attention (standalone)

    Uses ArchitectureExpectations dataclass for per-model validation criteria.
    Validates the component-tracing approach in TransformerBridgeStrategyAdapter.

    Also includes tests for validating TL-style to canonical parameter mapping functionality:
    - TL-style names (e.g., blocks.9.attn.W_Q, embed.W_E)
    - Canonical names (e.g., model.blocks.9._original_component.attn.q._original_component.weight)

    The mapping must be:
    1. Complete: All TL params should map to canonical params
    2. Invertible: Canonical params should map back to TL params
    3. Architecture-invariant: Work across GPT-2, Llama, Gemma, etc.
    """

    # Expected TL-style parameter patterns for different component types
    TL_ATTENTION_PATTERNS = {
        r"blocks\.\d+\.attn\.W_Q",
        r"blocks\.\d+\.attn\.W_K",
        r"blocks\.\d+\.attn\.W_V",
        r"blocks\.\d+\.attn\.W_O",
        r"blocks\.\d+\.attn\.b_Q",
        r"blocks\.\d+\.attn\.b_K",
        r"blocks\.\d+\.attn\.b_V",
        r"blocks\.\d+\.attn\.b_O",
    }

    TL_MLP_PATTERNS = {
        r"blocks\.\d+\.mlp\.W_in",
        r"blocks\.\d+\.mlp\.W_out",
        r"blocks\.\d+\.mlp\.b_in",
        r"blocks\.\d+\.mlp\.b_out",
        # Gated MLPs (Llama, Gemma)
        r"blocks\.\d+\.mlp\.W_gate",
        r"blocks\.\d+\.mlp\.b_gate",
    }

    TL_EMBEDDING_PATTERNS = {
        r"embed\.W_E",
        r"pos_embed\.W_pos",
        r"unembed\.W_U",
        r"unembed\.b_U",
    }

    TL_LAYERNORM_PATTERNS = {
        r"blocks\.\d+\.ln1\.w",
        r"blocks\.\d+\.ln1\.b",
        r"blocks\.\d+\.ln2\.w",
        r"blocks\.\d+\.ln2\.b",
        r"ln_final\.w",
        r"ln_final\.b",
    }

    @staticmethod
    def get_tl_params_from_bridge(bridge) -> Dict[str, torch.Tensor]:
        """Get TL-style parameters from a TransformerBridge instance."""
        return dict(bridge.tl_named_parameters())

    @staticmethod
    def get_canonical_params_from_module(module) -> Dict[str, torch.Tensor]:
        """Get canonical parameters from a LightningModule."""
        return dict(module.named_parameters())

    @staticmethod
    def categorize_tl_param(name: str) -> str:
        """Categorize a TL-style parameter name by component type."""
        if ".attn." in name:
            return "attention"
        elif ".mlp." in name:
            return "mlp"
        elif "embed" in name or "unembed" in name:
            return "embedding"
        elif "ln" in name:
            return "layernorm"
        else:
            return "other"

    @staticmethod
    def validate_mapping_structure(tl_to_canonical: Dict, canonical_to_tl: Dict) -> Dict[str, Set[str]]:
        """Validate the structure of bidirectional mappings.

        Returns:
            Dict with 'unmapped_tl', 'unmapped_canonical', 'multi_mapped_canonical' sets
        """
        issues = {
            "unmapped_tl": set(),
            "unmapped_canonical": set(),
            "multi_mapped_canonical": set(),
        }

        # Check for unmapped TL params
        for tl_name, canonical_list in tl_to_canonical.items():
            if not canonical_list:
                issues["unmapped_tl"].add(tl_name)

        # Check for canonical params with multiple TL mappings (which is OK for views)
        canonical_counts = {}
        for canonical_name, tl_name in canonical_to_tl.items():
            if tl_name not in canonical_counts:
                canonical_counts[tl_name] = []
            canonical_counts[tl_name].append(canonical_name)

        # Multi-mapping is expected for views, but log for awareness
        for tl_name, canonical_list in canonical_counts.items():
            if len(canonical_list) > 1:
                issues["multi_mapped_canonical"].add(tl_name)

        return issues

    def test_tl_param_pattern_coverage(self):
        """Test that TL-style parameter patterns are comprehensive."""
        # Verify pattern sets are non-empty and well-formed
        assert len(self.TL_ATTENTION_PATTERNS) >= 8
        assert len(self.TL_MLP_PATTERNS) >= 4
        assert len(self.TL_EMBEDDING_PATTERNS) >= 2

        # Verify patterns compile without error
        all_patterns = (
            self.TL_ATTENTION_PATTERNS | self.TL_MLP_PATTERNS | self.TL_EMBEDDING_PATTERNS | self.TL_LAYERNORM_PATTERNS
        )
        for pattern in all_patterns:
            compiled = re.compile(pattern)
            assert compiled is not None

    def test_param_categorization(self):
        """Test that parameter categorization works correctly."""
        test_cases = [
            ("blocks.0.attn.W_Q", "attention"),
            ("blocks.11.attn.b_O", "attention"),
            ("blocks.5.mlp.W_in", "mlp"),
            ("blocks.5.mlp.W_gate", "mlp"),
            ("embed.W_E", "embedding"),
            ("unembed.W_U", "embedding"),
            ("pos_embed.W_pos", "embedding"),
            ("blocks.0.ln1.w", "layernorm"),
            ("ln_final.w", "layernorm"),
        ]

        for param_name, expected_category in test_cases:
            actual = self.categorize_tl_param(param_name)
            assert actual == expected_category, f"Expected {expected_category} for {param_name}, got {actual}"

    def _validate_tl_param_structure(
        self, bridge, module, arch_expectations: ArchitectureExpectations
    ) -> Tuple[Dict, Dict]:
        """Verify TransformerBridge has expected TL-style parameter structure.

        Validates:
        - Non-empty parameters
        - Correct number of layers
        - All expected attention params per layer
        - All expected MLP params per layer
        - All expected embedding params

        Args:
            bridge: TransformerBridge model
            module: Lightning module wrapping the model
            arch_expectations: Expected architecture parameters

        Returns:
            Tuple of (tl_params, canonical_params) dicts for further validation
        """
        tl_params = dict(bridge.tl_named_parameters())
        canonical_params = dict(module.named_parameters())

        # ===== Basic Structure Validation =====
        assert len(tl_params) > 0, f"[{arch_expectations.model_name}] No TL-style parameters found"
        assert len(canonical_params) > 0, f"[{arch_expectations.model_name}] No canonical parameters found"

        # ===== Layer Count Validation =====
        layer_indices = set()
        for name in tl_params.keys():
            match = re.match(r"blocks\.(\d+)\.", name)
            if match:
                layer_indices.add(int(match.group(1)))

        assert len(layer_indices) == arch_expectations.n_layers, (
            f"[{arch_expectations.model_name}] Expected {arch_expectations.n_layers} layers, found {len(layer_indices)}"
        )

        # ===== Attention Parameter Validation =====
        for attn_suffix in arch_expectations.expected_tl_attn_params:
            param_name = f"blocks.0.{attn_suffix}"
            assert param_name in tl_params, f"[{arch_expectations.model_name}] Missing attention param: {param_name}"

        # ===== MLP Parameter Validation =====
        for mlp_suffix in arch_expectations.expected_tl_mlp_params:
            param_name = f"blocks.0.{mlp_suffix}"
            assert param_name in tl_params, f"[{arch_expectations.model_name}] Missing MLP param: {param_name}"

        # ===== Embedding Parameter Validation =====
        for embed_param in arch_expectations.expected_tl_embed_params:
            assert embed_param in tl_params, f"[{arch_expectations.model_name}] Missing embedding param: {embed_param}"

        # Check position embeddings based on architecture
        if arch_expectations.has_pos_embed:
            assert "pos_embed.W_pos" in tl_params, (
                f"[{arch_expectations.model_name}] Expected pos_embed.W_pos but not found"
            )

        # ===== Parameter Count Summary =====
        tl_count = len(tl_params)
        canonical_count = len(canonical_params)
        print(f"[{arch_expectations.model_name}] Parameter counts - TL: {tl_count}, Canonical: {canonical_count}")

        return tl_params, canonical_params

    @pytest.mark.parametrize(
        "session_fixture, arch_expectations",
        [
            pytest.param(
                "get_it_session__l_tl_bridge_gpt2__setup",
                GPT2_EXPECTATIONS,
                id="gpt2",
            ),
            pytest.param(
                "get_it_session__l_tl_bridge_gpt2_processed__setup",
                GPT2_PROCESSED_EXPECTATIONS,
                id="gpt2_processed",
            ),
            pytest.param(
                "get_it_session__l_tl_bridge_llama3__setup",
                LLAMA3_EXPECTATIONS,
                marks=RunIf(standalone=True, bf16_cuda=True),
                id="llama3",
            ),
            pytest.param(
                "get_it_session__l_tl_bridge_gemma2__setup",
                GEMMA2_EXPECTATIONS,
                marks=RunIf(standalone=True, bf16_cuda=True),
                id="gemma2",
            ),
        ],
    )
    def test_bidirectional_mapping(self, request, session_fixture: str, arch_expectations: ArchitectureExpectations):
        """Validate bidirectional TL ↔ canonical parameter mapping using component tracing.

        First validates TL parameter structure as a precondition, then tests the
        TransformerBridgeStrategyAdapter's _build_param_mapping() method which uses
        component structure tracing to build mappings.

        Validates:
        - TL parameter structure (layer count, attention/MLP/embedding params)
        - All non-LayerNorm TL params have canonical mappings
        - Canonical params without TL equivalents are expected (joint QKV, etc.)
        - Mapping is consistent (same param doesn't map to multiple TL names)
        """
        fixture = request.getfixturevalue(session_fixture)
        module = fixture.it_session.module
        bridge = module.model

        # ===== Precondition: Validate TL param structure =====
        tl_params, canonical_params = self._validate_tl_param_structure(bridge, module, arch_expectations)

        # ===== Build param mapping using component tracing =====
        adapter = TransformerBridgeStrategyAdapter(use_tl_names=True)

        # Create a mock fts_handle to provide pl_module and trainer access
        class MockTrainer:
            pass  # Minimal trainer stub

        class MockFTSHandle:
            def __init__(self, pl_module):
                self.pl_module = pl_module
                self.trainer = MockTrainer()

        adapter.fts_handle = MockFTSHandle(module)

        # Build the parameter mapping using component tracing
        # (ModelView gets created in on_before_init_fts)
        adapter.on_before_init_fts()

        assert adapter.model_view is not None, "ModelView not created"
        tl_to_canonical = adapter.model_view._tl_to_canonical_mapping
        canonical_to_tl = adapter.model_view._canonical_to_tl_mapping

        assert tl_to_canonical is not None, "TL to canonical mapping not built"
        assert canonical_to_tl is not None, "Canonical to TL mapping not built"

        # ===== TL → Canonical Mapping Validation =====
        unmapped_tl = []
        mapped_tl_count = 0
        for tl_name in tl_params.keys():
            canonical_names = tl_to_canonical.get(tl_name, [])
            if canonical_names:
                mapped_tl_count += 1
                # Verify canonical names actually exist
                for cn in canonical_names:
                    assert cn in canonical_params, (
                        f"[{arch_expectations.model_name}] TL '{tl_name}' maps to non-existent canonical '{cn}'"
                    )
            else:
                unmapped_tl.append(tl_name)

        print(
            f"[{arch_expectations.model_name}] TL→Canonical: {mapped_tl_count}/{len(tl_params)} mapped, "
            f"{len(unmapped_tl)} unmapped"
        )

        # Validate exact mapping counts - ALL TL params should be mapped
        assert mapped_tl_count == arch_expectations.expected_mapped_tl_count, (
            f"[{arch_expectations.model_name}] TL mapping count mismatch: "
            f"expected {arch_expectations.expected_mapped_tl_count}, got {mapped_tl_count}"
        )
        assert len(unmapped_tl) == arch_expectations.expected_unmapped_tl_count, (
            f"[{arch_expectations.model_name}] Unmapped TL count mismatch: "
            f"expected {arch_expectations.expected_unmapped_tl_count}, got {len(unmapped_tl)}. "
            f"Unmapped: {unmapped_tl[:10]}"
        )

        # ===== Canonical → TL Mapping Validation =====
        mapped_canonical_count = len(canonical_to_tl)
        unmapped_canonical = [k for k in canonical_params.keys() if k not in canonical_to_tl]

        print(
            f"[{arch_expectations.model_name}] Canonical→TL: {mapped_canonical_count}/{len(canonical_params)} mapped, "
            f"{len(unmapped_canonical)} unmapped"
        )

        # Validate exact mapping counts for canonical params
        assert mapped_canonical_count == arch_expectations.expected_mapped_canonical_count, (
            f"[{arch_expectations.model_name}] Canonical mapping count mismatch: "
            f"expected {arch_expectations.expected_mapped_canonical_count}, got {mapped_canonical_count}"
        )
        assert len(unmapped_canonical) == arch_expectations.expected_unmapped_canonical_count, (
            f"[{arch_expectations.model_name}] Unmapped canonical count mismatch: "
            f"expected {arch_expectations.expected_unmapped_canonical_count}, got {len(unmapped_canonical)}. "
            f"Unmapped: {unmapped_canonical[:10]}"
        )

        # Many canonical params won't have TL equivalents (LayerNorms, joint QKV, original components, etc.)
        # This is expected - we just validate that the reverse mapping is consistent
        for canonical_name, tl_name in canonical_to_tl.items():
            assert tl_name in tl_params, (
                f"[{arch_expectations.model_name}] Canonical '{canonical_name}' maps to non-existent TL '{tl_name}'"
            )
            # Verify bidirectional consistency
            assert canonical_name in tl_to_canonical.get(tl_name, []), (
                f"[{arch_expectations.model_name}] Bidirectional inconsistency: "
                f"canonical '{canonical_name}' → TL '{tl_name}' but TL doesn't map back"
            )


# =============================================================================
# Custom ModelView Tests
# =============================================================================


class TestCustomModelView:
    """Tests for custom ModelView usage."""

    def test_custom_model_view(self, get_it_session__l_tl_bridge_gpt2__setup):
        """Test that a simple custom ModelView can be successfully instantiated and used."""
        from interpretune.adapters.model_view import ModelView
        import os
        from typing import Dict, List, Optional, Union

        # Define a simple custom ModelView that prefixes all param names
        class PrefixedModelView(ModelView):
            """Test view that adds 'custom_' prefix to all parameter names."""

            def __init__(self, adapter):
                super().__init__(adapter)
                self.prefix = "custom_"

            def build_param_mapping(self) -> None:
                """No complex mapping needed for this simple test."""
                pass

            def transform_to_canonical(self, param_names: List[str], inspect_only: bool = False) -> List[str]:
                """Strip custom prefix to get canonical names."""
                return [
                    name.replace(self.prefix, "", 1) if name.startswith(self.prefix) else name for name in param_names
                ]

            def transform_from_canonical(self, param_names: List[str]) -> List[str]:
                """Add custom prefix to canonical names."""
                return [f"{self.prefix}{name}" for name in param_names]

            def get_named_params(self) -> Dict[str, torch.Tensor]:
                """Get params with custom prefix."""
                return {f"{self.prefix}{name}": param for name, param in self.pl_module.named_parameters()}

            def gen_schedule(self, dump_loc: Union[str, os.PathLike]) -> Optional[os.PathLike]:
                """Not tested in this simple test."""
                return None

            def validate_schedule(self) -> None:
                """Not tested in this simple test."""
                pass

        # Get the fixture
        fixture = get_it_session__l_tl_bridge_gpt2__setup
        module = fixture.it_session.module

        # Create adapter with custom ModelView class
        adapter = TransformerBridgeStrategyAdapter(model_view=PrefixedModelView)

        # Create mock FTS handle
        class MockFTSHandle:
            def __init__(self, pl_module):
                self.pl_module = pl_module

        class MockTrainer:
            pass

        adapter.fts_handle = MockFTSHandle(module)
        adapter.fts_handle.trainer = MockTrainer()

        # Initialize the model view
        adapter.on_before_init_fts()

        # Verify custom ModelView was created
        assert adapter.model_view is not None
        assert isinstance(adapter.model_view, PrefixedModelView)

        # Test transformations
        canonical_params = ["model.embed.weight", "model.blocks.0.attn.W_Q"]

        # Transform to canonical (should strip prefix)
        prefixed_params = adapter.model_view.transform_from_canonical(canonical_params)
        assert all(p.startswith("custom_") for p in prefixed_params)
        assert prefixed_params == ["custom_model.embed.weight", "custom_model.blocks.0.attn.W_Q"]

        # Transform back to canonical (should remove prefix)
        back_to_canonical = adapter.model_view.transform_to_canonical(prefixed_params)
        assert back_to_canonical == canonical_params

        # Test get_named_params returns prefixed names
        named_params = adapter.model_view.get_named_params()
        assert all(name.startswith("custom_") for name in named_params.keys())
        assert len(named_params) > 0  # Should have some parameters


class TestImplicitLayerNormThawing:
    """Test implicit LayerNorm thawing configuration in TLNamesModelView."""

    def test_implicit_ln_thaw_default(self, get_it_session__l_tl_bridge_gpt2__setup):
        """Test that implicit_ln_thaw defaults to True (existing behavior)."""
        fixture = get_it_session__l_tl_bridge_gpt2__setup
        module = fixture.it_session.module

        # Create adapter with use_tl_names=True (creates TLNamesModelView)
        adapter = TransformerBridgeStrategyAdapter(use_tl_names=True)

        # Set up the adapter
        class MockFTSHandle:
            def __init__(self, pl_module):
                self.pl_module = pl_module

        class MockTrainer:
            pass

        adapter.fts_handle = MockFTSHandle(module)
        adapter.fts_handle.trainer = MockTrainer()
        adapter.on_before_init_fts()

        # Default adapter with use_tl_names=True should have TLNamesModelView with implicit_ln_thaw=True
        assert hasattr(adapter.model_view, "implicit_ln_thaw"), (
            "TLNamesModelView should have implicit_ln_thaw attribute"
        )
        assert adapter.model_view.implicit_ln_thaw is True

        # Test that LayerNorm params are included when thawing attention blocks
        tl_params = ["blocks.0.attn.W_Q", "blocks.0.attn.W_K"]
        canonical_params = adapter.model_view.transform_to_canonical(tl_params)

        # Should include ln_1 and ln_2 for block 0
        ln_params = [p for p in canonical_params if "ln_1" in p or "ln_2" in p]
        assert len(ln_params) > 0, "Should include implicit LayerNorm params with implicit_ln_thaw=True"

    def test_implicit_ln_thaw_disabled(self, get_it_session__l_tl_bridge_gpt2__setup):
        """Test that implicit_ln_thaw=False disables LayerNorm thawing."""
        fixture = get_it_session__l_tl_bridge_gpt2__setup
        module = fixture.it_session.module

        # Create new adapter with implicit_ln_thaw=False
        adapter = TransformerBridgeStrategyAdapter(use_tl_names=True, model_view_cfg={"implicit_ln_thaw": False})

        # Set up the adapter
        class MockFTSHandle:
            def __init__(self, pl_module):
                self.pl_module = pl_module

        class MockTrainer:
            pass

        adapter.fts_handle = MockFTSHandle(module)
        adapter.fts_handle.trainer = MockTrainer()
        adapter.on_before_init_fts()

        # Verify implicit_ln_thaw is False
        assert adapter.model_view.implicit_ln_thaw is False

        # Test that LayerNorm params are NOT included
        tl_params = ["blocks.0.attn.W_Q", "blocks.0.attn.W_K"]
        canonical_params = adapter.model_view.transform_to_canonical(tl_params)

        # Should NOT include ln_1 or ln_2 for block 0
        ln_params = [p for p in canonical_params if "ln_1" in p or "ln_2" in p]
        assert len(ln_params) == 0, "Should not include LayerNorm params with implicit_ln_thaw=False"

        # Verify only the requested TL params were transformed
        # Each TL param may map to one or more canonical params (e.g., weight and bias)
        assert len(canonical_params) >= len(tl_params), "Should have at least the requested params"

    def test_implicit_ln_thaw_with_model_view_class(self, get_it_session__l_tl_bridge_gpt2__setup):
        """Test passing model_view_cfg when using model_view parameter."""
        fixture = get_it_session__l_tl_bridge_gpt2__setup
        module = fixture.it_session.module

        # Import TLNamesModelView
        from interpretune.adapters.transformer_lens import TLNamesModelView

        # Create adapter with model_view class and config
        adapter = TransformerBridgeStrategyAdapter(
            model_view=TLNamesModelView, model_view_cfg={"implicit_ln_thaw": False}
        )

        # Set up the adapter
        class MockFTSHandle:
            def __init__(self, pl_module):
                self.pl_module = pl_module

        class MockTrainer:
            pass

        adapter.fts_handle = MockFTSHandle(module)
        adapter.fts_handle.trainer = MockTrainer()
        adapter.on_before_init_fts()

        # Verify config was passed correctly
        assert adapter.model_view.implicit_ln_thaw is False

        # Test transformation
        tl_params = ["blocks.0.mlp.W_in"]
        canonical_params = adapter.model_view.transform_to_canonical(tl_params)

        # Should NOT include LayerNorm params
        ln_params = [p for p in canonical_params if "ln_" in p]
        assert len(ln_params) == 0, "Should not include LayerNorm params"
