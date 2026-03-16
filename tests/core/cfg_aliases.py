from collections.abc import Sequence
from copy import deepcopy
from enum import auto
from dataclasses import dataclass, field
from typing import Iterable
from pathlib import Path
import tempfile


import interpretune as it
from interpretune.protocol import Adapter, AutoStrEnum
from interpretune.config import (
    HFFromPretrainedConfig,
    HFGenerationConfig,
    GenerativeClassificationConfig,
    CoreGenerationConfig,
    ITLensCustomConfig,
    TLensGenerationConfig,
    AutoCompConfig,
    ITLensBridgeConfig,
    ITLensFromPretrainedNoProcessingConfig,
    SAELensFromPretrainedConfig,
    AnalysisCfg,
    CircuitTracerConfig,
    ITLensCfg,
    NNsightConfig,
)
from interpretune.extensions import DebugLMConfig, MemProfilerCfg
from interpretune.analysis import LatentAnalysisTargets, AnalysisOp
from it_examples.experiments.rte_boolq import RTEBoolqEntailmentMapping
from tests.base_defaults import BaseAugTest, BaseCfg, AnalysisBaseCfg
from tests.parity_acceptance.cfg_aliases import parity_cli_cfgs, mod_initargs, CLI_TESTS
from tests.parity_acceptance.test_it_tl import TLParityCfg
from tests.parity_acceptance.test_it_cli import CLICfg
from tests.utils import get_nested


nf4_bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
}
base_lora_cfg = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM"}
gpt2_lora_cfg = {"target_modules": ["c_attn", "c_proj"], **base_lora_cfg}
gpt2_hf_bnb_lora_cfg = {"bitsandbytesconfig": nf4_bnb_config, "lora_cfg": gpt2_lora_cfg}
gpt2_hf_bnb_lora_cfg_seq = {"bitsandbytesconfig": nf4_bnb_config, "lora_cfg": {**gpt2_lora_cfg, "task_type": "SEQ_CLS"}}
gpt2_seq_hf_from_pretrained_kwargs = {
    "pretrained_kwargs": {"device_map": "cpu", "dtype": "float32"},
    "model_head": "transformers.AutoModelForSequenceClassification",
}
tl_cust_mi_cfg = {
    "cfg": dict(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=200,
        act_fn="relu",
        attention_dir="causal",
        tokenizer_name="gpt2",
        seed=1,
        use_attn_result=True,
    )
}

test_tl_cust_2L_config = {
    "n_layers": 2,
    "d_mlp": 10,
    "d_model": 10,
    "d_head": 5,
    "n_heads": 2,
    "n_ctx": 200,
    "act_fn": "relu",
    "tokenizer_name": "gpt2",
}


@dataclass(kw_only=True)
class CoreCfgForcePrepare(BaseCfg):
    model_src_key: str | None = "cust"
    force_prepare_data: bool | None = True
    dm_override_cfg: dict | None = field(
        default_factory=lambda: {"enable_datasets_cache": False, "dataset_path": "/tmp/force_prepare_tests_ds"}
    )


@dataclass(kw_only=True)
class TLMechInterpCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.transformer_lens)
    model_src_key: str | None = "cust"
    force_prepare_data: bool | None = True
    tl_cfg: dict | None = field(default_factory=lambda: ITLensCustomConfig(**tl_cust_mi_cfg))
    dm_override_cfg: dict | None = field(
        default_factory=lambda: {
            "enable_datasets_cache": True,
            "tokenizer_kwargs": {"padding_side": "right", "model_input_names": ["input"]},
            "dataset_path": str(Path(tempfile.gettempdir()) / "tl_cust_mi_force_prepare_ds"),
        }
    )


@dataclass(kw_only=True)
class TLDebugCfg(TLParityCfg):
    debug_lm_cfg: DebugLMConfig | None = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
    generative_step_cfg: GenerativeClassificationConfig | None = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=TLensGenerationConfig(
                max_new_tokens=4,
                generate_kwargs={"output_logits": True, "return_dict_in_generate": True},
            ),
        )
    )


@dataclass(kw_only=True)
class NSDebugCfg(BaseCfg):
    """NNsight debug configuration for GPT2-based sanity tests."""

    debug_lm_cfg: DebugLMConfig | None = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight)
    generative_step_cfg: GenerativeClassificationConfig | None = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=CoreGenerationConfig(
                max_new_tokens=4, generate_kwargs={"output_logits": True, "return_dict_in_generate": True}
            ),
        )
    )


@dataclass(kw_only=True)
class CoreGPT2PEFTCfg(BaseCfg):
    device_type: str = "cuda"
    model_src_key: str | None = "gpt2"
    dm_override_cfg: dict | None = field(default_factory=lambda: {"train_batch_size": 1, "eval_batch_size": 1})
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = field(
        default_factory=lambda: HFFromPretrainedConfig(
            **gpt2_hf_bnb_lora_cfg,
            pretrained_kwargs={"device_map": "cpu", "dtype": "float32"},
            model_head="transformers.GPT2LMHeadModel",
            activation_checkpointing=True,
        )
    )
    limit_train_batches: int | None = 3
    limit_val_batches: int | None = 3
    limit_test_batches: int | None = 2


@dataclass(kw_only=True)
class CoreGPT2PEFTSeqCfg(CoreGPT2PEFTCfg):
    phase: str | None = "test"
    generative_step_cfg: GenerativeClassificationConfig | None = field(
        default_factory=lambda: GenerativeClassificationConfig(enabled=False)
    )
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = field(
        default_factory=lambda: HFFromPretrainedConfig(
            **gpt2_hf_bnb_lora_cfg_seq, **gpt2_seq_hf_from_pretrained_kwargs, activation_checkpointing=True
        )
    )


@dataclass(kw_only=True)
class CoreMemProfCfg(BaseCfg):
    memprofiler_cfg: MemProfilerCfg | None = field(
        default_factory=lambda: MemProfilerCfg(enabled=True, cuda_allocator_history=True)
    )
    dm_override_cfg: dict | None = field(default_factory=lambda: {"train_batch_size": 1, "eval_batch_size": 1})
    limit_train_batches: int | None = 5
    limit_val_batches: int | None = 3
    limit_test_batches: int | None = 5
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"


@dataclass(kw_only=True)
class LightningLlama3DebugCfg(BaseCfg):
    debug_lm_cfg: DebugLMConfig | None = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: str | None = "test"
    device_type: str | None = "cuda"
    model_src_key: str | None = "llama3"
    precision: str | int | None = "bf16-true"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)


@dataclass(kw_only=True)
class LightningGemma2DebugCfg(BaseCfg):
    debug_lm_cfg: DebugLMConfig | None = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: str | None = "test"
    device_type: str | None = "cuda"
    model_src_key: str | None = "gemma2"
    precision: str | int | None = "bf16-true"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)


@dataclass(kw_only=True)
class LightningGemma3DebugCfg(BaseCfg):
    debug_lm_cfg: DebugLMConfig | None = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: str | None = "test"
    device_type: str | None = "cuda"
    model_src_key: str | None = "gemma3"
    precision: str | int | None = "bf16-true"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)


################################################################################
# Transformer Lens Test Configs
################################################################################


@dataclass(kw_only=True)
class LightningTLBridgeLlama3(BaseCfg):
    """Llama3 with TransformerBridge for parameter mapping validation tests.

    Uses meta-llama/Llama-3.2-3B-Instruct (registry default for model_src_key=llama3).
    Exercises:
    - GQA (grouped-query attention) with n_key_value_heads != n_heads
    - SwiGLU MLP with gate projection (W_gate)
    - RMSNorm instead of LayerNorm
    """

    model_src_key: str | None = "llama3"
    device_type: str | None = "cuda"
    precision: str | int | None = "bf16-true"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    tl_cfg: ITLensBridgeConfig = field(
        default_factory=lambda: ITLensBridgeConfig(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            default_padding_side="left",
        )
    )


@dataclass(kw_only=True)
class LightningTLBridgeGemma2(BaseCfg):
    """Gemma2 with TransformerBridge for parameter mapping validation tests.

    Uses google/gemma-2-2b for reasonable test size while exercising:
    - Sliding window attention with interleaved global attention
    - GeGLU activation with gate projection
    - RMSNorm instead of LayerNorm
    """

    model_src_key: str | None = "gemma2"
    device_type: str | None = "cuda"
    precision: str | int | None = "bf16-true"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    tl_cfg: ITLensBridgeConfig = field(
        default_factory=lambda: ITLensBridgeConfig(
            model_name="google/gemma-2-2b",
            default_padding_side="left",
        )
    )
    datamodule_cls: str | None = "tests.modules.FingerprintTestITDataModule"
    module_cls: str | None = "tests.modules.TestITModule"


@dataclass(kw_only=True)
class LightningGPT2(BaseCfg):
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)
    model_cfg: dict | None = field(default_factory=lambda: {"tie_word_embeddings": False})


@dataclass(kw_only=True)
class LightningTLGPT2(BaseCfg):
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    tl_cfg: ITLensFromPretrainedNoProcessingConfig = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gpt2-small", default_padding_side="left", use_bridge=False
        )
    )


@dataclass(kw_only=True)
class LightningTLBridgeGPT2(BaseCfg):
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    logging_level: str | int = "DEBUG"
    tl_cfg: ITLensBridgeConfig = field(
        default_factory=lambda: ITLensBridgeConfig(
            model_name="gpt2-small",
            default_padding_side="left",
        )
    )


@dataclass(kw_only=True)
class LightningTLBridgeGPT2Processed(BaseCfg):
    """GPT-2 with TransformerBridge using default weight processing (fold_ln, fold_value_biases, etc.).

    This config uses ITLensBridgeConfig with enable_compatibility_mode=True to apply fold_ln=True,
    fold_value_biases=True, etc. Used to test bidirectional mapping behavior when LayerNorms are folded etc.
    """

    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    tl_cfg: ITLensCfg | None = field(
        default_factory=lambda: ITLensBridgeConfig(
            model_name="gpt2-small",
            default_padding_side="left",
            enable_compatibility_mode=True,
        )
    )


################################################################################
# NNsight Test Configs
################################################################################


@dataclass(kw_only=True)
class CoreNNsightGPT2(BaseCfg):
    """Core NNsight adapter with GPT-2 for unit testing.

    Registered in example_module_registry.yaml as gpt2.rte.nnsight. Uses adapter combination (core, nnsight).
    """

    phase: str = "test"
    model_src_key: str | None = "gpt2"  # Must match registry
    model_cfg_key: str = "rte"  # Must match registry
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight)  # Must match registry
    nnsight_cfg: "NNsightConfig | None" = field(
        default_factory=lambda: NNsightConfig(
            model_name="openai-community/gpt2",
            device_map="cpu",
            torch_dtype="float32",
            dispatch=True,
        )
    )


@dataclass(kw_only=True)
class LightningNNsightGPT2(BaseCfg):
    """Lightning NNsight adapter with GPT-2 for unit testing.

    Registered in example_module_registry.yaml as gpt2.rte.nnsight. Uses adapter combination (lightning, nnsight).
    """

    phase: str = "test"
    model_src_key: str | None = "gpt2"  # Must match registry
    model_cfg_key: str = "rte"  # Must match registry
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.nnsight)  # Must match registry
    nnsight_cfg: "NNsightConfig | None" = field(
        default_factory=lambda: NNsightConfig(
            model_name="openai-community/gpt2",
            device_map="cpu",
            torch_dtype="float32",
            dispatch=True,
        )
    )


@dataclass(kw_only=True)
class CoreNNsightRemoteGPT2(BaseCfg):
    """Core NNsight adapter with GPT-2 for unit testing.

    Registered in example_module_registry.yaml as gpt2.rte.nnsight. Uses adapter combination (core, nnsight).
    """

    phase: str = "test"
    model_src_key: str | None = "gpt2"  # Must match registry
    model_cfg_key: str = "rte"  # Must match registry
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight)  # Must match registry
    nnsight_cfg: "NNsightConfig | None" = field(
        default_factory=lambda: NNsightConfig(
            model_name="openai-community/gpt2", device_map="cpu", torch_dtype="float32", dispatch=True, remote=True
        )
    )


################################################################################
# Circuit Tracer Test Configs
################################################################################


@dataclass(kw_only=True)
class CircuitTracerTLGemma2(BaseCfg):
    """Circuit Tracer with TransformerLens backend on Gemma2.

    Uses adapter combination (core, transformer_lens, circuit_tracer) which composes BaseCircuitTracerModule with
    BaseITLensModule for TL-specific functionality.
    """

    phase: str = "test"
    model_src_key: str | None = "gemma2"
    model_cfg_key: str = "rte_base_test"
    device_type: str = "cuda"
    # Use transformer_lens adapter for TL backend
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.transformer_lens, Adapter.circuit_tracer)
    tl_cfg: ITLensCfg | None = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gemma-2-2b", default_padding_side="left", use_bridge=False
        )
    )
    generative_step_cfg: GenerativeClassificationConfig | None = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1, output_logits=True, return_dict_in_generate=True),
        )
    )
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="transformerlens",  # though TL is default we explicitly set backend here for clarity
            transcoder_set="gemma",
            analysis_target_tokens=["▁Dallas", "▁Austin"],
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
        )
    )


@dataclass(kw_only=True)
class LightningCircuitTracerTLGemma2(BaseCfg):
    """Lightning Circuit Tracer with TransformerLens backend on Gemma2.

    Uses adapter combination (lightning, transformer_lens, circuit_tracer) which composes BaseCircuitTracerModule with
    BaseITLensModule and Lightning adapters.
    """

    phase: str = "test"
    model_src_key: str | None = "gemma2"
    model_cfg_key: str = "rte_base_test"  # Required to lookup registry entry
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens, Adapter.circuit_tracer)
    tl_cfg: ITLensCfg | None = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gemma-2-2b", default_padding_side="left", use_bridge=False
        )
    )
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="transformerlens",  # though TL is default we explicitly set backend here for clarity
            transcoder_set="gemma",
            analysis_target_tokens=["▁Dallas", "▁Austin"],
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
        )
    )


# Circuit Tracer with NNsight backend configs
##############################################################################################################
@dataclass(kw_only=True)
class CircuitTracerNNsightGemma2(BaseCfg):
    """Circuit Tracer with NNsight backend on Gemma2.

    Registered in example_module_registry.yaml as gemma2.rte.circuit_tracer_nnsight. Uses adapter combination (core,
    nnsight, circuit_tracer).
    """

    phase: str = "test"
    model_src_key: str | None = "gemma2"
    model_cfg_key: str = "rte_base_test"  # Must match registry
    device_type: str = "cuda"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight, Adapter.circuit_tracer)
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(
            model_name="google/gemma-2-2b",
            device_map="cuda",
            torch_dtype="float32",
            dispatch=True,
            attn_implementation="eager",
            tokenizer_kwargs={"padding_side": "left", "add_bos_token": True},
        )
    )
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="nnsight",
            transcoder_set="gemma",
            analysis_target_tokens=["▁Dallas", "▁Austin"],
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
        )
    )


@dataclass(kw_only=True)
class LightningCircuitTracerNNsightGemma2(BaseCfg):
    """Lightning Circuit Tracer with NNsight backend on Gemma2.

    Registered in example_module_registry.yaml as gemma2.rte.circuit_tracer_nnsight. Uses adapter combination
    (lightning, nnsight, circuit_tracer).
    """

    phase: str = "test"
    model_src_key: str | None = "gemma2"
    model_cfg_key: str = "rte_base_test"  # Must match registry
    device_type: str = "cuda"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.nnsight, Adapter.circuit_tracer)
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(
            model_name="google/gemma-2-2b",
            device_map="cuda",
            torch_dtype="float32",
            dispatch=True,
            attn_implementation="eager",
            tokenizer_kwargs={"padding_side": "left", "add_bos_token": True},
        )
    )
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="nnsight",
            transcoder_set="gemma",
            analysis_target_tokens=["▁Dallas", "▁Austin"],
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
        )
    )


@dataclass(kw_only=True)
class CircuitTracerNNsightRemoteGemma2(BaseCfg):
    """Circuit Tracer with NNsight backend on Gemma2 for remote mode testing.

    Registered in example_module_registry.yaml as gemma2.rte.circuit_tracer_nnsight_remote. Uses adapter combination
    (core, nnsight, circuit_tracer) with nnsight_remote=True.
    """

    phase: str = "test"
    model_src_key: str | None = "gemma2"
    model_cfg_key: str = "rte_base_test"  # Must match registry
    device_type: str = "cuda"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight, Adapter.circuit_tracer)
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(
            model_name="google/gemma-2-2b",
            torch_dtype="float32",
            dispatch=False,
            remote=True,
            attn_implementation="eager",
            tokenizer_kwargs={"padding_side": "left", "add_bos_token": True},
        )
    )
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="nnsight",
            transcoder_set="gemma",
            analysis_target_tokens=["▁Dallas", "▁Austin"],
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
            nnsight_remote=True,
        )
    )


##############################################################################################################


# Circuit Tracer with NNsight backend configs (Gemma3)
##############################################################################################################
@dataclass(kw_only=True)
class CircuitTracerNNsightGemma3(BaseCfg):
    """Circuit Tracer with NNsight backend on Gemma3-1B-PT.

    Registered in example_module_registry.yaml as gemma3.rte.circuit_tracer_nnsight. Uses adapter combination (core,
    nnsight, circuit_tracer). Gemma3 only supports NNsight backend in circuit-tracer (no TL/HookedTransformer path).
    """

    phase: str = "test"
    model_src_key: str | None = "gemma3"
    model_cfg_key: str = "rte_base_test"  # Must match registry
    device_type: str = "cuda"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight, Adapter.circuit_tracer)
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="nnsight",
            transcoder_set="mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine",
            analysis_target_tokens=["▁Dallas", "▁Austin"],
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
        )
    )


@dataclass(kw_only=True)
class LightningCircuitTracerNNsightGemma3(BaseCfg):
    """Lightning Circuit Tracer with NNsight backend on Gemma3-1B-PT.

    Registered in example_module_registry.yaml as gemma3.rte.circuit_tracer_nnsight. Uses adapter combination
    (lightning, nnsight, circuit_tracer).
    """

    phase: str = "test"
    model_src_key: str | None = "gemma3"
    model_cfg_key: str = "rte_base_test"  # Must match registry
    device_type: str = "cuda"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.nnsight, Adapter.circuit_tracer)
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="nnsight",
            transcoder_set="mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine",
            analysis_target_tokens=["▁Dallas", "▁Austin"],
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
        )
    )


##############################################################################################################

################################################################################
# SAE Test Configs
################################################################################


@dataclass(kw_only=True)
class CoreSLHTGPT2(BaseCfg):
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    use_bridge: bool = False  # Explicitly use legacy HookedSAETransformer path
    tl_cfg: ITLensFromPretrainedNoProcessingConfig = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gpt2-small", default_padding_side="left", use_bridge=False
        )
    )
    # force_prepare_data: bool | None = True  # sometimes useful to enable for test debugging


@dataclass(kw_only=True)
class CoreSLNNsightGPT2(BaseCfg):
    """NNsight backend variant of CoreSLHTGPT2 for basic SAE adapter tests."""

    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight, Adapter.sae_lens)
    generative_step_cfg: GenerativeClassificationConfig = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=HFGenerationConfig(
                model_config={"max_new_tokens": 1, "output_logits": True, "return_dict_in_generate": True}
            ),
        )
    )
    hf_from_pretrained_cfg: HFFromPretrainedConfig = field(
        default_factory=lambda: HFFromPretrainedConfig(
            pretrained_kwargs={"dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        )
    )
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(
            model_name="openai-community/gpt2",
            device_map="cpu",
            torch_dtype="float32",
            dispatch=True,
        )
    )
    auto_comp_cfg: AutoCompConfig = field(
        default_factory=lambda: AutoCompConfig(
            module_cfg_name="RTEBoolqConfig", module_cfg_mixin=RTEBoolqEntailmentMapping, target_adapters="nnsight"
        )
    )


@dataclass(kw_only=True)
class CoreSLBridgeGPT2(BaseCfg):
    """TransformerBridge variant of CoreSLHTGPT2 for basic SAE adapter tests."""

    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    tl_cfg: ITLensBridgeConfig = field(
        default_factory=lambda: ITLensBridgeConfig(model_name="gpt2-small", default_padding_side="left")
    )
    hf_from_pretrained_cfg: HFFromPretrainedConfig = field(
        default_factory=lambda: HFFromPretrainedConfig(
            pretrained_kwargs={"device_map": "cpu", "dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        )
    )


@dataclass(kw_only=True)
class CoreSLHTGPT2Analysis(AnalysisBaseCfg):
    phase: str | None = "analysis"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    use_bridge: bool = False  # Explicitly use legacy HookedSAETransformer path
    generative_step_cfg: GenerativeClassificationConfig = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1, output_logits=True, return_dict_in_generate=True),
        )
    )
    latent_analysis_targets: LatentAnalysisTargets = field(
        default_factory=lambda: LatentAnalysisTargets(sae_release="gpt2-small-hook-z-kk", target_layers=[9, 10])
    )
    hf_from_pretrained_cfg: HFFromPretrainedConfig = field(
        default_factory=lambda: HFFromPretrainedConfig(
            pretrained_kwargs={"dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        )
    )
    tl_cfg: ITLensFromPretrainedNoProcessingConfig = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gpt2-small", default_padding_side="left", use_bridge=False
        )
    )
    sae_cfgs: list = field(default_factory=lambda: [])
    auto_comp_cfg: AutoCompConfig = field(
        default_factory=lambda: AutoCompConfig(
            module_cfg_name="RTEBoolqConfig", module_cfg_mixin=RTEBoolqEntailmentMapping
        )
    )
    # TODO: customize these cache paths for testing efficiency
    # cache_dir: str | None = None
    # op_output_dataset_path: str | None = None
    # important for ephemeral CI runner alignment
    force_prepare_data: bool | None = True
    dm_override_cfg: dict | None = field(
        default_factory=lambda: {
            "enable_datasets_cache": True,
            "dataset_path": str(Path(tempfile.gettempdir()) / "force_prepare_analysis_ds"),
        }
    )

    def __post_init__(self):
        super().__post_init__()
        # Dynamically generate sae_cfgs from sae_targets.latent_model_fqns
        if self.latent_analysis_targets and hasattr(self.latent_analysis_targets, "latent_model_fqns"):
            self.sae_cfgs = [
                SAELensFromPretrainedConfig(release=sae_fqn.release, sae_id=sae_fqn.sae_id)
                for sae_fqn in self.latent_analysis_targets.latent_model_fqns
            ]


@dataclass(kw_only=True)
class CoreSLHTGPT2LogitDiffsBase(CoreSLHTGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_base, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLHTGPT2LogitDiffsSAE(CoreSLHTGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_sae, save_prompts=True, save_tokens=True, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLHTGPT2LogitDiffsAttrGrad(CoreSLHTGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_attr_grad, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLHTGPT2LogitDiffsAttrAblation(CoreSLHTGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_attr_ablation, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


################################################################################
# NNsight SAE Analysis Test Configs (Backend Parity)
################################################################################


@dataclass(kw_only=True)
class CoreSLNNsightGPT2Analysis(AnalysisBaseCfg):
    """NNsight backend variant of CoreSLHTGPT2Analysis for SAE backend parity testing.

    Uses (core, nnsight, sae_lens) adapter composition with SAELensConfig(backend='nnsight').
    """

    phase: str | None = "analysis"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight, Adapter.sae_lens)
    generative_step_cfg: GenerativeClassificationConfig = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=HFGenerationConfig(
                model_config={"max_new_tokens": 1, "output_logits": True, "return_dict_in_generate": True}
            ),
        )
    )
    latent_analysis_targets: LatentAnalysisTargets = field(
        default_factory=lambda: LatentAnalysisTargets(sae_release="gpt2-small-hook-z-kk", target_layers=[9, 10])
    )
    hf_from_pretrained_cfg: HFFromPretrainedConfig = field(
        default_factory=lambda: HFFromPretrainedConfig(
            pretrained_kwargs={"dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        )
    )
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(
            model_name="openai-community/gpt2",
            device_map="cpu",
            torch_dtype="float32",
            dispatch=True,
        )
    )
    sae_cfgs: list = field(default_factory=lambda: [])
    auto_comp_cfg: AutoCompConfig = field(
        default_factory=lambda: AutoCompConfig(
            module_cfg_name="RTEBoolqConfig", module_cfg_mixin=RTEBoolqEntailmentMapping, target_adapters="nnsight"
        )
    )
    force_prepare_data: bool | None = True
    dm_override_cfg: dict | None = field(
        default_factory=lambda: {
            "enable_datasets_cache": True,
            "dataset_path": str(Path(tempfile.gettempdir()) / "force_prepare_analysis_ns_ds"),
            # NOTE: attention_mask MUST be in signature_columns.  Without it,
            # _remove_unused_columns strips the proper mask created during
            # tokenization.  DataCollatorWithPadding then synthesizes an
            # all-ones mask (sequences are already pre-padded to max_length),
            # so GPT-2 via NNsight never masks the left-padding tokens.
            # TL computes its own correct left-padding mask internally
            # (default_padding_side="left"), so both backends agree when
            # NNsight receives the original tokenization mask.
            "signature_columns": ["input_ids", "attention_mask", "labels"],
        }
    )

    def __post_init__(self):
        super().__post_init__()
        # Dynamically generate sae_cfgs from sae_targets.latent_model_fqns
        if self.latent_analysis_targets and hasattr(self.latent_analysis_targets, "latent_model_fqns"):
            self.sae_cfgs = [
                SAELensFromPretrainedConfig(release=sae_fqn.release, sae_id=sae_fqn.sae_id)
                for sae_fqn in self.latent_analysis_targets.latent_model_fqns
            ]


@dataclass(kw_only=True)
class CoreSLNNsightGPT2LogitDiffsBase(CoreSLNNsightGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_base, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLNNsightGPT2LogitDiffsSAE(CoreSLNNsightGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_sae, save_prompts=True, save_tokens=True, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLNNsightGPT2LogitDiffsAttrGrad(CoreSLNNsightGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_attr_grad, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLNNsightGPT2LogitDiffsAttrAblation(CoreSLNNsightGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_attr_ablation, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


################################################################################
# TransformerBridge SAE Analysis Test Configs (Bridge vs Hooked Parity)
################################################################################


@dataclass(kw_only=True)
class CoreSLBridgeGPT2Analysis(AnalysisBaseCfg):
    """TransformerBridge variant of CoreSLHTGPT2Analysis for Bridge vs Hooked parity testing.

    Uses (core, sae_lens) adapter composition with use_bridge=True and ITLensBridgeConfig instead of
    ITLensFromPretrainedNoProcessingConfig.
    """

    phase: str | None = "analysis"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    use_bridge: bool = True
    generative_step_cfg: GenerativeClassificationConfig = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1, output_logits=True, return_dict_in_generate=True),
        )
    )
    latent_analysis_targets: LatentAnalysisTargets = field(
        default_factory=lambda: LatentAnalysisTargets(sae_release="gpt2-small-hook-z-kk", target_layers=[9, 10])
    )
    hf_from_pretrained_cfg: HFFromPretrainedConfig = field(
        default_factory=lambda: HFFromPretrainedConfig(
            pretrained_kwargs={"dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        )
    )
    tl_cfg: ITLensBridgeConfig = field(
        default_factory=lambda: ITLensBridgeConfig(model_name="gpt2-small", default_padding_side="left")
    )
    sae_cfgs: list = field(default_factory=lambda: [])
    auto_comp_cfg: AutoCompConfig = field(
        default_factory=lambda: AutoCompConfig(
            module_cfg_name="RTEBoolqConfig", module_cfg_mixin=RTEBoolqEntailmentMapping
        )
    )
    force_prepare_data: bool | None = True
    dm_override_cfg: dict | None = field(
        default_factory=lambda: {
            "enable_datasets_cache": True,
            "dataset_path": str(Path(tempfile.gettempdir()) / "force_prepare_analysis_br_ds"),
            # NOTE: TransformerBridge requires explicit attention_mask in the data pipeline.
            # Unlike HookedTransformer, which auto-creates an attention mask from
            # tokenizer.padding_side in ``input_to_embed()``, TransformerBridge passes
            # tokens directly to the underlying HF model.  Without an explicit mask the
            # HF model treats left-padding tokens as real content, introducing a
            # systematic logit shift (~5 logit-units for GPT-2 on RTE).
            # Two changes are needed:
            # 1) tokenizer_kwargs must include 'attention_mask' in model_input_names so
            #    the tokenizer actually PRODUCES the mask (HF tokenizers check
            #    model_input_names to decide whether to generate attention_mask).
            # 2) signature_columns must include 'attention_mask' so
            #    _remove_unused_columns keeps it in the saved dataset.
            "tokenizer_kwargs": {
                "model_input_names": ["input", "attention_mask"],
                "padding_side": "left",
                "add_bos_token": True,
            },
            "signature_columns": ["input", "attention_mask", "labels"],
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.latent_analysis_targets and hasattr(self.latent_analysis_targets, "latent_model_fqns"):
            self.sae_cfgs = [
                SAELensFromPretrainedConfig(release=sae_fqn.release, sae_id=sae_fqn.sae_id)
                for sae_fqn in self.latent_analysis_targets.latent_model_fqns
            ]


@dataclass(kw_only=True)
class CoreSLBridgeGPT2LogitDiffsBase(CoreSLBridgeGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_base, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLBridgeGPT2LogitDiffsSAE(CoreSLBridgeGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_sae, save_prompts=True, save_tokens=True, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLBridgeGPT2LogitDiffsAttrGrad(CoreSLBridgeGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_attr_grad, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLBridgeGPT2LogitDiffsAttrAblation(CoreSLBridgeGPT2Analysis):
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = (
        AnalysisCfg(target_op=it.logit_diffs_attr_ablation, save_prompts=False, save_tokens=False, ignore_manual=True),
    )


@dataclass(kw_only=True)
class CoreSLCust(BaseCfg):
    phase: str | None = "test"
    model_src_key: str | None = "cust"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    tl_cfg: dict | None = field(default_factory=lambda: ITLensCustomConfig(cfg=test_tl_cust_2L_config))


@dataclass(kw_only=True)
class LightningSLHTGPT2(BaseCfg):
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
    use_bridge: bool = False
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.sae_lens)
    tl_cfg: ITLensFromPretrainedNoProcessingConfig = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gpt2-small", default_padding_side="left", use_bridge=False
        )
    )


class CLI_UNIT_TESTS(AutoStrEnum):
    seed_null = auto()
    env_seed = auto()
    invalid_env_seed = auto()
    invalid_cfg_seed = auto()
    nonint_cfg_seed = auto()
    seed_false = auto()
    seed_true = auto()
    excess_args = auto()


TEST_CONFIGS_CLI_UNIT = (
    BaseAugTest(
        alias=CLI_UNIT_TESTS.seed_null.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True),
        expected={"seed_test": lambda x: int(x) >= 0},
    ),
    BaseAugTest(
        alias=CLI_UNIT_TESTS.env_seed.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True, env_seed=13),
        expected={"seed_test": lambda x: int(x) == 13},
    ),
    BaseAugTest(
        alias=CLI_UNIT_TESTS.invalid_env_seed.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True, env_seed="oops"),
        expected={"seed_test": lambda x: int(x) >= 0},
    ),
    BaseAugTest(
        alias=CLI_UNIT_TESTS.invalid_cfg_seed.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True),
        expected={"seed_test": lambda x: int(x) >= 0},
    ),
    BaseAugTest(
        alias=CLI_UNIT_TESTS.nonint_cfg_seed.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True),
        expected={"seed_test": lambda x: int(x) == 14},
    ),
    BaseAugTest(
        alias=CLI_UNIT_TESTS.seed_false.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True),
        expected={"seed_test": lambda x: x is None},
    ),
    BaseAugTest(
        alias=CLI_UNIT_TESTS.seed_true.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True),
        expected={"seed_test": lambda x: int(x) >= 0},
    ),
    BaseAugTest(
        alias=CLI_UNIT_TESTS.excess_args.value,
        cfg=CLICfg(compose_cfg=True, debug_mode=True, extra_args=["--foo"]),
        expected={"seed_test": lambda x: int(x) >= 0},
    ),
)

EXPECTED_RESULTS_CLI_UNIT = {cfg.alias: cfg.expected for cfg in TEST_CONFIGS_CLI_UNIT}

################################################################################
# core adapter training with no transformer_lens adapter context
################################################################################
unit_exp_cli_cfgs = {}
base_unit_exp_cli_cfg = deepcopy(parity_cli_cfgs["exp_cfgs"][CLI_TESTS.core_optim_train])
get_nested(base_unit_exp_cli_cfg, mod_initargs)["experiment_tag"] = CLI_UNIT_TESTS.seed_null.value

unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null] = base_unit_exp_cli_cfg
unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null]["seed_everything"] = None

unit_exp_cli_cfgs[CLI_UNIT_TESTS.env_seed] = deepcopy(unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null])

unit_exp_cli_cfgs[CLI_UNIT_TESTS.invalid_env_seed] = deepcopy(unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null])

unit_exp_cli_cfgs[CLI_UNIT_TESTS.invalid_cfg_seed] = deepcopy(unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null])
unit_exp_cli_cfgs[CLI_UNIT_TESTS.invalid_cfg_seed]["seed_everything"] = -1

unit_exp_cli_cfgs[CLI_UNIT_TESTS.nonint_cfg_seed] = deepcopy(unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null])
unit_exp_cli_cfgs[CLI_UNIT_TESTS.nonint_cfg_seed]["seed_everything"] = 14.0

unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_false] = deepcopy(unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null])
unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_false]["seed_everything"] = False

unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_true] = deepcopy(unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null])
unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_true]["seed_everything"] = True

unit_exp_cli_cfgs[CLI_UNIT_TESTS.excess_args] = deepcopy(unit_exp_cli_cfgs[CLI_UNIT_TESTS.seed_null])
