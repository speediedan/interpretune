import warnings
from copy import deepcopy
from dataclasses import dataclass, field
import re

import pytest
import torch
from torch import device

from interpretune.config import ITLensFromPretrainedConfig, ITLensConfig, ITLensCustomConfig
from interpretune.utils import MisconfigurationException
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
        expected_mapped_tl_count: Expected number of TL params the view traces to parameters
        expected_unmapped_tl_count: Expected number of untraced TL params (synthetic zero biases)
        expected_mapped_canonical_count: Expected number of the module's parameters some TL name reaches
        expected_unmapped_canonical_count: Expected number of the module's parameters no TL name reaches
                                          (norms TL does not name, joint QKV, etc.)
        expected_uncoupled_canonical_count: Of those, how many no thaw couples either: parameters the forward pass
                                           never uses
        hf_block_path: Wrapped HuggingFace model's path to a block, with ``{i}`` for the layer index
        hf_block_norms: The block's norm submodules, by the HuggingFace model's own names
        hf_final_norm: Path to the final norm in the wrapped HuggingFace model
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
    expected_tl_attn_params: list[str] = field(default_factory=lambda: ["attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O"])
    expected_tl_mlp_params: list[str] = field(default_factory=lambda: ["mlp.W_in", "mlp.W_out"])
    expected_tl_embed_params: list[str] = field(default_factory=lambda: ["embed.W_E", "unembed.W_U"])

    # Canonical naming patterns (regex) - architecture specific
    # These are used to validate bidirectional mapping
    canonical_attn_pattern: str | None = None
    canonical_mlp_pattern: str | None = None
    canonical_embed_pattern: str | None = None

    # Expected mapping counts for bidirectional validation
    expected_mapped_tl_count: int | None = None
    expected_unmapped_tl_count: int = 0  # All TL params should map
    expected_mapped_canonical_count: int | None = None
    expected_unmapped_canonical_count: int | None = None
    expected_uncoupled_canonical_count: int = 0

    # The norms an implicit thaw must cover, named from the HuggingFace side so the expectation does not share a
    # source with the component index the adapter builds.
    hf_block_path: str = "model.layers.{i}"
    hf_block_norms: tuple[str, ...] = ("input_layernorm", "post_attention_layernorm")
    hf_final_norm: str = "model.norm"


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
    # Mapping counts, measured on TransformerLens 4.0 with the model loaded unquantized (see LightningTLBridgeLlama3).
    # 424 TL params: 28 layers * (4 attn weights + 4 attn biases + 3 mlp weights + 2 mlp biases + ln1.w + ln2.w)
    # + embed.W_E + ln_final.w + unembed.W_U + unembed.b_U; a rotary model gets no pos_embed.W_pos under 4.0.
    # Every TL param maps except the 168 synthetic biases (28 * 6) the bridge zero-fills for a bias-free model, and
    # all 255 canonical params map (lm_head's weight is tied to the embedding).
    expected_mapped_tl_count=256,
    expected_unmapped_tl_count=168,  # synthetic attn/mlp biases only
    expected_mapped_canonical_count=255,
    expected_unmapped_canonical_count=0,
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
    # Mapping counts, measured on TransformerLens 4.0. 394 TL params: 26 layers * (4 attn weights + 4 attn biases
    # + 3 mlp weights + 2 mlp biases + ln1.w + ln2.w) + embed.W_E + ln_final.w + unembed.W_U + unembed.b_U.
    # Every TL param maps except the 156 synthetic biases (26 * 6). Of the 289 canonical params, the 52 unmapped
    # are each block's post_attention_layernorm and post_feedforward_layernorm: the bridge names them
    # ln1_post/ln2_post, but tl_named_parameters() does not emit them.
    expected_mapped_tl_count=238,
    expected_unmapped_tl_count=156,  # synthetic attn/mlp biases only
    expected_mapped_canonical_count=237,
    expected_unmapped_canonical_count=52,
    hf_block_norms=(
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ),
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
    # Mapping counts: 12 layers * (8 attn + 4 mlp + 4 LayerNorm) + 4 embed/pos_embed/unembed + 2 ln_final = 198 TL
    # params. TransformerLens 4.0 gives LayerNorms TL-style names (`blocks.N.ln1.w`, `ln_final.b`, ...); before it
    # they had none, the TL count was 148 and the 50 LayerNorm params were the unmapped canonical remainder.
    # Canonical: 221 = 197 reached (198 TL names, embed and unembed sharing one tensor) + the 24 joint QKV weights and
    # biases (12 * 2), which no TL name reaches: their q/k/v splits are strided views of them, so each is thawed with
    # its splits as a storage alias.
    expected_mapped_tl_count=198,
    expected_unmapped_tl_count=0,
    expected_mapped_canonical_count=197,
    expected_unmapped_canonical_count=24,
    hf_block_path="transformer.h.{i}",
    hf_block_norms=("ln_1", "ln_2"),
    hf_final_norm="transformer.ln_f",
)

# GPT-2 with weight processing enabled (fold_ln=True, fold_value_biases=True, center_writing_weights=True, etc.)
# LN params carry TL-style names under TransformerLens 4.0, folded or not.
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
    # LayerNorm params appear in TL params too (see the counts below); the lists here name the attn/MLP subset
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
    # TL counts match unprocessed GPT-2: 198, LayerNorms included (see GPT2_EXPECTATIONS).
    # Compatible mode adds extra canonical params: 222 total (vs 221 for unprocessed)
    # Canonical: 198 reached + 24 unreached = 222 total (embed/unembed not shared with processed mode)
    # Unreached: the 24 joint QKV params (12*2). Compatible mode materializes the q/k/v splits as tensors of their own,
    # so the joints share no storage with them and the forward pass never uses them: nothing couples them either.
    expected_mapped_tl_count=198,
    expected_unmapped_tl_count=0,
    expected_mapped_canonical_count=198,
    expected_unmapped_canonical_count=24,  # QKV joint params (compatible mode)
    expected_uncoupled_canonical_count=24,
    hf_block_path="transformer.h.{i}",
    hf_block_norms=("ln_1", "ln_2"),
    hf_final_norm="transformer.ln_f",
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
        # A TransformerBridge forwards unknown attributes to the model it wraps, and that model carries its
        # own `cfg`, so ablating only the bridge's leaves `cfg` resolvable and this warn path unreachable.
        # (Under the weight-converting HookedTransformer there was one object and one `cfg`.) Ablate both, so
        # the assertion still exercises "config genuinely unreachable" rather than passing for free.
        with (
            ablate_cls_attrs(tl_test_module.model.original_model, "cfg"),
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

    def test_tl_retired_use_bridge_is_refused_by_name(self):
        """``use_bridge`` is gone, and passing it fails by name rather than being accepted and ignored.

        It selected the weight-converting ``HookedTransformer`` against ``TransformerBridge``.
        TransformerLens 4.0 removes that class, so the flag has one reachable value and no reason to exist;
        ``enable_compatibility_mode`` on the bridge config is what still expresses the old numerics.

        The assertion is that construction RAISES. A retired field that is quietly swallowed leaves a config
        whose author believes they selected something they did not, and nothing downstream can tell.
        """
        for cfg_cls in (ITLensFromPretrainedConfig, ITLensCustomConfig):
            assert "use_bridge" not in cfg_cls.__dataclass_fields__, (
                f"{cfg_cls.__name__} still declares the retired use_bridge field"
            )

        test_tl_cfg = deepcopy(TestClassTransformerLens.test_tlens_gpt2)
        test_tl_cfg["tl_cfg"]["use_bridge"] = False
        with pytest.raises(TypeError, match="use_bridge"):
            ITLensFromPretrainedConfig(**test_tl_cfg["tl_cfg"])

    def test_tl_custom_config_is_no_longer_special_cased(self):
        """A custom (config-only) tl_cfg builds like any other, with no warn-and-force branch.

        It was forced onto the legacy path because ``TransformerBridge`` could not be built without an HF
        model. ``TransformerBridge.boot_native`` removes that premise, so the special case is gone and
        constructing a custom config must not warn about bridge support.
        """
        test_tl_cfg_custom = deepcopy(TestClassTransformerLens.test_tlens_cust)
        test_tl_cfg_custom["tl_cfg"] = ITLensCustomConfig(**test_tl_cfg_custom["tl_cfg"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            it_cfg_custom = ITLensConfig(**test_tl_cfg_custom)
        # Assert on the specific warning that is gone, not on silence: this path legitimately emits
        # unrelated instantiation feedback (tokenizer_name resolution), and asserting silence would
        # make the test fail for a reason that has nothing to do with the special case being removed.
        offending = [w for w in caught if "does not support TransformerBridge" in str(w.message)]
        assert not offending, f"the retired warn-and-force branch still fires: {offending}"
        assert it_cfg_custom.tl_cfg is not None

    def test_tl_custom_config_hook_flags_take_effect_or_refuse(self):
        """Hook-exposing flags on a config-only model either reach the model or are refused by name.

        ``boot_native`` accepts them on the config without propagating them, so ``use_attn_result=True`` built a
        model with no ``hook_result`` and no error. The TL-native attention cannot provide per-head results at all.
        """
        from transformer_lens.config import TransformerBridgeConfig
        from transformer_lens.model_bridge.bridge import TransformerBridge

        from interpretune.adapters.transformer_lens.adapter import BaseITLensModule

        small = dict(d_model=16, d_head=4, n_heads=4, n_layers=1, n_ctx=8, d_vocab=32, act_fn="relu", seed=0)
        bridge = TransformerBridge.boot_native(dict(small))
        BaseITLensModule._apply_bridge_hook_flags(bridge, TransformerBridgeConfig(**small, use_hook_mlp_in=True))
        assert bridge.cfg.use_hook_mlp_in is True
        with pytest.raises(NotImplementedError, match="use_attn_result"):
            BaseITLensModule._apply_bridge_hook_flags(bridge, TransformerBridgeConfig(**small, use_attn_result=True))


class TestTrailingPadTensorRefusal:
    """The bridge loop starts from the last position, so trailing-pad tensors must be refused.

    Measured on gpt2/CPU (#123): a right-padded row's first step reads post-pad logits, emits end-of-text at once, and
    freezes — only the unpadded (longest) row responds. These pin the refusal without instantiating a model.
    """

    def test_right_padded_batch_raises_by_name(self):
        """A row ending in pads names the pad id and the accepted alternatives."""
        from interpretune.adapters.transformer_lens.adapter import _reject_trailing_pad_tensor_batch

        batch = torch.tensor([[10, 11, 12], [20, 50256, 50256]])
        with pytest.raises(ValueError, match="trailing pad token id 50256"):
            _reject_trailing_pad_tensor_batch(batch, pad_token_id=50256, padding_side="right")

    def test_clean_and_left_padded_batches_pass(self):
        """Batches whose last position is real reach the loop untouched."""
        from interpretune.adapters.transformer_lens.adapter import _reject_trailing_pad_tensor_batch

        _reject_trailing_pad_tensor_batch(
            torch.tensor([[50256, 50256, 20, 21], [10, 11, 12, 13]]),
            pad_token_id=50256,
            padding_side="left",
        )
        _reject_trailing_pad_tensor_batch(
            torch.tensor([[10, 11, 12], [20, 21, 22]]), pad_token_id=50256, padding_side="right"
        )
        _reject_trailing_pad_tensor_batch(torch.tensor([[10, 50256]]), pad_token_id=None, padding_side="right")

    def test_every_tl_composition_inherits_the_refusal(self):
        """Every TL-backed module composition resolves ``it_generate`` to the guarded override.

        All of them run the same bridge generation loop, so a composition that bypassed the override (one
        registered without ``BaseITLensModule``, or with a class ahead of it that redefines ``it_generate``)
        would generate from trailing pads silently. C3 is resolved by hand because some registered tuples
        carry bases whose metaclasses only reconcile inside the real composed class.
        """
        import interpretune as it
        from interpretune.adapters.transformer_lens.adapter import BaseITLensModule

        def c3(bases):
            seqs = [list(b.__mro__) for b in bases] + [list(bases)]
            out = []
            while any(seqs):
                seqs = [seq for seq in seqs if seq]
                head = next(seq[0] for seq in seqs if not any(seq[0] in other[1:] for other in seqs))
                out.append(head)
                for seq in seqs:
                    if seq[0] is head:
                        seq.pop(0)
            return out

        tl_keys = [
            key
            for key in it.ADAPTER_REGISTRY.registry
            if key[0] == "module" and any(getattr(a, "value", a) == "transformer_lens" for a in key[1:])
        ]
        assert tl_keys, "no TransformerLens module compositions registered"
        owners = {
            key: next(c for c in c3(it.ADAPTER_REGISTRY.registry[key][key]) if "it_generate" in c.__dict__)
            for key in tl_keys
        }
        assert {key: owner for key, owner in owners.items() if owner is not BaseITLensModule} == {}


# Every architecture the mapping tests cover; the gated models need bf16 CUDA and the gated-repo token.
ARCH_MAPPING_CASES = [
    pytest.param("get_it_session__l_tl_bridge_gpt2__setup", GPT2_EXPECTATIONS, id="gpt2"),
    pytest.param("get_it_session__l_tl_bridge_gpt2_processed__setup", GPT2_PROCESSED_EXPECTATIONS, id="gpt2_processed"),
    pytest.param(
        "get_it_session__l_tl_bridge_llama3__setup",
        LLAMA3_EXPECTATIONS,
        # measured up to 10.0 GiB in its own process (24 GiB card), across both tests that use this case
        marks=RunIf(bf16_cuda=True, min_gpu_mem_gb=11.5, requires_env="HF_GATED_PUBLIC_REPO_AUTH_KEY"),
        id="llama3",
    ),
    pytest.param(
        "get_it_session__l_tl_bridge_gemma2__setup",
        GEMMA2_EXPECTATIONS,
        # measured 5.6 GiB in its own process (24 GiB card), across both tests that use this case
        marks=RunIf(bf16_cuda=True, min_gpu_mem_gb=6.5, requires_env="HF_GATED_PUBLIC_REPO_AUTH_KEY"),
        id="gemma2",
    ),
]


# =============================================================================
# TransformerLens View Mapping, per Architecture
# =============================================================================
class TestBridgeViewArchitectures:
    """The TransformerLens view (``TransformerBridgeView``) on each architecture the bridge wraps.

    - GPT-2: MHA, learned pos_embed, biases (unprocessed and processed)
    - Llama3: GQA, SwiGLU, RMSNorm, RoPE (bf16 CUDA)
    - Gemma2: GQA, GeGLU, RMSNorm with pre and post norms, sliding window attention (bf16 CUDA)

    finetuning-scheduler's mapping cases run here per architecture (interpretune#760); the rest checks what those cases
    cannot know: the TransformerLens names each architecture should have, how many trace to parameters, and that the
    implicit norm thaw covers exactly the norms the wrapped HuggingFace model has.
    """

    @staticmethod
    def _view(request, session_fixture: str):
        pytest.importorskip("finetuning_scheduler.tunable_view")
        from interpretune.adapters.transformer_lens.tunable_view import TransformerBridgeView
        from interpretune.tuning.views import VIEW_ATTR

        module = request.getfixturevalue(session_fixture).it_session.module
        view = getattr(module, VIEW_ATTR)
        assert isinstance(view, TransformerBridgeView) and view.root is module
        return module, view

    @staticmethod
    def _validate_tl_param_structure(bridge, arch: ArchitectureExpectations) -> dict[str, torch.Tensor]:
        """The bridge's TransformerLens names: layer count and the attention, MLP and embedding names expected."""
        tl_params = dict(bridge.tl_named_parameters())
        assert tl_params, f"[{arch.model_name}] No TL-style parameters found"
        layers = {int(m.group(1)) for name in tl_params if (m := re.match(r"blocks\.(\d+)\.", name))}
        assert len(layers) == arch.n_layers, f"[{arch.model_name}] Expected {arch.n_layers} layers, found {len(layers)}"
        for suffix in (*arch.expected_tl_attn_params, *arch.expected_tl_mlp_params):
            assert f"blocks.0.{suffix}" in tl_params, f"[{arch.model_name}] Missing block param: blocks.0.{suffix}"
        for name in arch.expected_tl_embed_params:
            assert name in tl_params, f"[{arch.model_name}] Missing embedding param: {name}"
        assert ("pos_embed.W_pos" in tl_params) == arch.has_pos_embed, f"[{arch.model_name}] pos_embed.W_pos"
        return tl_params

    @pytest.mark.parametrize("session_fixture, arch_expectations", ARCH_MAPPING_CASES)
    def test_view_conformance(self, request, session_fixture: str, arch_expectations: ArchitectureExpectations):
        """The read-only finetuning-scheduler cases: a complete, round-tripping mapping and a default schedule that
        compiles."""
        _, view = self._view(request, session_fixture)
        from finetuning_scheduler.testing import TunableViewConformance

        cases = ["mapping_completeness", "mapping_round_trip", "default_schedule_compiles"]
        report = TunableViewConformance(lambda: view).run(select=cases)
        assert report.ok, report.failures
        assert set(report.ran) == set(cases), report.ran

    @pytest.mark.parametrize("session_fixture, arch_expectations", ARCH_MAPPING_CASES)
    def test_trace_covers_the_model(self, request, session_fixture: str, arch_expectations: ArchitectureExpectations):
        """Every TransformerLens name traces to parameters except the synthesized zero biases, and every parameter
        the model uses is reached by a name or thawed with one (a fused projection's storage alias, a norm
        TransformerLens does not name)."""
        module, view = self._view(request, session_fixture)
        arch = arch_expectations
        tl_params = self._validate_tl_param_structure(module.model, arch)
        physical = dict(module.named_parameters())
        traced = dict(view.to_physical(list(n for n in tl_params if n not in view.untraced)))
        assert len(traced) == arch.expected_mapped_tl_count, f"[{arch.model_name}] traced {len(traced)}"
        assert len(view.untraced) == arch.expected_unmapped_tl_count, f"[{arch.model_name}] {view.untraced[:10]}"
        assert all(not tl_params[n].any() for n in view.untraced), f"[{arch.model_name}] an untraced name is not zero"
        reached = {p for ps in traced.values() for p in ps}
        assert reached <= set(physical)
        unreached = set(physical) - reached
        assert len(reached) == arch.expected_mapped_canonical_count, f"[{arch.model_name}] reached {len(reached)}"
        assert len(unreached) == arch.expected_unmapped_canonical_count, (
            f"[{arch.model_name}] unreached {sorted(unreached)[:10]}"
        )
        uncoupled = unreached - set(view.coupled(sorted(reached)))
        assert len(uncoupled) == arch.expected_uncoupled_canonical_count, (
            f"[{arch.model_name}] neither reached nor coupled: {sorted(uncoupled)[:10]}"
        )

    @pytest.mark.parametrize("session_fixture, arch_expectations", ARCH_MAPPING_CASES)
    def test_implicit_norm_thaw_covers_every_norm(
        self, request, session_fixture: str, arch_expectations: ArchitectureExpectations
    ):
        """Thawing part of a block thaws every norm of that block, and thawing an embedding thaws the final norm.

        The expected norms come from the wrapped HuggingFace model's own module paths, not from the bridge's components,
        so a component index that missed a norm (or matched one family's names) fails here. Norms are what the implicit
        thaw adds over the storage aliases a view without it couples.
        """
        from interpretune.adapters.transformer_lens.tunable_view import TransformerBridgeView

        module, view = self._view(request, session_fixture)
        arch = arch_expectations
        aliases_only = TransformerBridgeView(module, implicit_norm_thaw=False)
        physical = dict(module.named_parameters())
        hf_model = module.model.original_model

        def norms_thawed_with(logical: str) -> set[int]:
            ps = view.to_physical([logical])[logical]
            added = set(view.coupled(ps)) - set(aliases_only.coupled(ps))
            return {physical[name].data_ptr() for name in added}

        def hf_ptrs(*paths):
            return {p.data_ptr() for path in paths for p in hf_model.get_submodule(path).parameters()}

        for layer in (0, arch.n_layers - 1):
            block = arch.hf_block_path.format(i=layer)
            expected = hf_ptrs(*(f"{block}.{norm}" for norm in arch.hf_block_norms))
            assert expected, f"[{arch.model_name}] no HF norm params found under {block}"
            assert norms_thawed_with(f"blocks.{layer}.attn.W_Q") == expected, (
                f"[{arch.model_name}] layer {layer}: the implicit thaw does not cover exactly {arch.hf_block_norms}"
            )
        assert norms_thawed_with("embed.W_E") == hf_ptrs(arch.hf_final_norm), (
            f"[{arch.model_name}] the implicit thaw of embed.W_E is not {arch.hf_final_norm}"
        )
