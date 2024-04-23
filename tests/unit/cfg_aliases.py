from typing import Optional, Dict
from copy import deepcopy
from dataclasses import dataclass, field

from tests.parity_acceptance.base.cfg_aliases import gpt2_hf_from_pretrained_kwargs, enable_activation_checkpointing
from tests.parity_acceptance.plugins.transformer_lens.test_interpretune_tl import TLParityCfg
from tests.configuration import BaseCfg
from interpretune.base.config.mixins import HFFromPretrainedConfig, ZeroShotClassificationConfig
from interpretune.analysis.debug_generation import DebugLMConfig
from interpretune.analysis.memprofiler import MemProfilerCfg
from interpretune.base.contract.session import Framework

nf4_bnb_config = {"load_in_4bit": True, "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4",
                   "bnb_4bit_compute_dtype": "bfloat16"}
base_lora_cfg = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM"}
gpt2_lora_cfg = {"target_modules": ["c_attn", "c_proj"], **base_lora_cfg}
gpt2_hf_bnb_lora_cfg = {"bitsandbytesconfig": nf4_bnb_config, "lora_cfg": gpt2_lora_cfg}
gpt2_seq_hf_from_pretrained_kwargs = deepcopy(gpt2_hf_from_pretrained_kwargs)
gpt2_seq_hf_from_pretrained_kwargs.update({"model_head": "transformers.AutoModelForSequenceClassification"})


@dataclass(kw_only=True)
class TLDebugCfg(TLParityCfg):
    debug_lm_cfg: Optional[DebugLMConfig] = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: Optional[str] = "test"
    model_src_key: Optional[str] = "gpt2"

@dataclass(kw_only=True)
class CoreGPT2PEFTCfg(BaseCfg):
    device_type: str = "cuda"
    model_src_key: Optional[str] = "gpt2"
    dm_override_cfg: Optional[Dict] = field(default_factory=lambda: {'train_batch_size': 1, 'eval_batch_size': 1})
    hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = field(default_factory=lambda: HFFromPretrainedConfig(
        **gpt2_hf_bnb_lora_cfg, **gpt2_hf_from_pretrained_kwargs, **enable_activation_checkpointing))
    limit_train_batches: Optional[int] = 3
    limit_val_batches: Optional[int] = 3
    limit_test_batches: Optional[int] = 2

@dataclass(kw_only=True)
class CoreGPT2PEFTSeqCfg(CoreGPT2PEFTCfg):
    phase: Optional[str] = "test"
    zero_shot_cfg: Optional[ZeroShotClassificationConfig] = \
        field(default_factory=lambda: ZeroShotClassificationConfig(enabled=False))
    hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = field(default_factory=lambda: HFFromPretrainedConfig(
        **gpt2_hf_bnb_lora_cfg, **gpt2_seq_hf_from_pretrained_kwargs, **enable_activation_checkpointing))


@dataclass(kw_only=True)
class CoreMemProfCfg(BaseCfg):
    memprofiler_cfg: Optional[MemProfilerCfg] = field(default_factory=lambda: MemProfilerCfg(enabled=True,
                                                                                         cuda_allocator_history=True))
    dm_override_cfg: Optional[Dict] = field(default_factory=lambda: {'train_batch_size': 1, 'eval_batch_size': 1})
    limit_train_batches: Optional[int] = 5
    limit_val_batches: Optional[int] = 3
    limit_test_batches: Optional[int] = 5
    phase: Optional[str] = "test"
    model_src_key: Optional[str] = "gpt2"

@dataclass(kw_only=True)
class LightningLlama2DebugCfg(BaseCfg):
    debug_lm_cfg: Optional[DebugLMConfig] = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: Optional[str] = "test"
    device_type: Optional[str] = "cuda"
    model_src_key: Optional[str] = "llama2"
    precision: Optional[str | int] = "bf16-true"
    framework_ctx: Optional[Framework] = Framework.lightning
