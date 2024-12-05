from typing import Optional, Dict, Sequence
from copy import deepcopy
from enum import auto
from dataclasses import dataclass, field

from interpretune.adapters.registration import Adapter
from interpretune.base.config.shared import AutoStrEnum
from interpretune.base.config.mixins import HFFromPretrainedConfig, ZeroShotClassificationConfig
from interpretune.adapters.transformer_lens import ITLensCustomConfig
from interpretune.extensions.debug_generation import DebugLMConfig
from interpretune.extensions.memprofiler import MemProfilerCfg
from tests.base_defaults import BaseAugTest, BaseCfg
from tests.parity_acceptance.cfg_aliases import parity_cli_cfgs, mod_initargs, CLI_TESTS
from tests.parity_acceptance.test_it_tl import TLParityCfg
from tests.parity_acceptance.test_it_cli import CLICfg
from tests.utils import get_nested


nf4_bnb_config = {"load_in_4bit": True, "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4",
                   "bnb_4bit_compute_dtype": "bfloat16"}
base_lora_cfg = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM"}
gpt2_lora_cfg = {"target_modules": ["c_attn", "c_proj"], **base_lora_cfg}
gpt2_hf_bnb_lora_cfg = {"bitsandbytesconfig": nf4_bnb_config, "lora_cfg": gpt2_lora_cfg}
gpt2_seq_hf_from_pretrained_kwargs = {"pretrained_kwargs": {"device_map": "cpu", "torch_dtype": "float32"},
                                      "model_head": "transformers.AutoModelForSequenceClassification"}
tl_cust_mi_cfg = {"cfg":
    dict(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=200,
    #d_vocab=50278,
    act_fn="relu",
    attention_dir="causal",
    #attn_only=True, # defaults to False
    tokenizer_name='gpt2',
    seed=1,
    use_attn_result=True,
    #normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    #positional_embedding_type="shortformer")
)}

test_tl_cust_2L_config = {"n_layers":2, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
                          "act_fn":'relu', "tokenizer_name": 'gpt2'}

@dataclass(kw_only=True)
class CoreCfgForcePrepare(BaseCfg):
    model_src_key: Optional[str] = "cust"
    force_prepare_data: Optional[bool] = True
    dm_override_cfg: Optional[Dict] = field(default_factory=lambda: {'enable_datasets_cache': False,
                                                                     'dataset_path': '/tmp/force_prepare_tests_ds'})

@dataclass(kw_only=True)
class TLMechInterpCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.transformer_lens)
    model_src_key: Optional[str] = "cust"
    tl_cfg: Optional[Dict] = field(default_factory=lambda: ITLensCustomConfig(**tl_cust_mi_cfg))

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
        **gpt2_hf_bnb_lora_cfg, pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
        model_head="transformers.GPT2LMHeadModel", activation_checkpointing=True))
    limit_train_batches: Optional[int] = 3
    limit_val_batches: Optional[int] = 3
    limit_test_batches: Optional[int] = 2

@dataclass(kw_only=True)
class CoreGPT2PEFTSeqCfg(CoreGPT2PEFTCfg):
    phase: Optional[str] = "test"
    zero_shot_cfg: Optional[ZeroShotClassificationConfig] = \
        field(default_factory=lambda: ZeroShotClassificationConfig(enabled=False))
    hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = field(default_factory=lambda: HFFromPretrainedConfig(
        **gpt2_hf_bnb_lora_cfg, **gpt2_seq_hf_from_pretrained_kwargs, activation_checkpointing=True))


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
class LightningLlama3DebugCfg(BaseCfg):
    debug_lm_cfg: Optional[DebugLMConfig] = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: Optional[str] = "test"
    device_type: Optional[str] = "cuda"
    model_src_key: Optional[str] = "llama3"
    precision: Optional[str | int] = "bf16-true"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)

@dataclass(kw_only=True)
class LightningGemma2DebugCfg(BaseCfg):
    debug_lm_cfg: Optional[DebugLMConfig] = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: Optional[str] = "test"
    device_type: Optional[str] = "cuda"
    model_src_key: Optional[str] = "gemma2"
    precision: Optional[str | int] = "bf16-true"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)

@dataclass(kw_only=True)
class LightningGPT2(BaseCfg):
    model_src_key: Optional[str] = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)
    model_cfg: Optional[Dict] = field(default_factory=lambda: {"tie_word_embeddings": False})

@dataclass(kw_only=True)
class LightningTLGPT2(BaseCfg):
    model_src_key: Optional[str] = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)

@dataclass(kw_only=True)
class CoreSLGPT2(BaseCfg):
    phase: Optional[str] = "test"
    model_src_key: Optional[str] = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)

@dataclass(kw_only=True)
class CoreSLCust(BaseCfg):
    phase: Optional[str] = "test"
    model_src_key: Optional[str] = "cust"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    tl_cfg: Optional[Dict] = field(default_factory=lambda: ITLensCustomConfig(cfg=test_tl_cust_2L_config))

@dataclass(kw_only=True)
class LightningSLGPT2(BaseCfg):
    phase: Optional[str] = "test"
    model_src_key: Optional[str] = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.sae_lens)

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
    BaseAugTest(alias=CLI_UNIT_TESTS.seed_null.value, cfg=CLICfg(compose_cfg=True, debug_mode=True),
                expected={'seed_test': lambda x: int(x) >= 0}),
    BaseAugTest(alias=CLI_UNIT_TESTS.env_seed.value, cfg=CLICfg(compose_cfg=True, debug_mode=True, env_seed=13),
                expected={'seed_test': lambda x: int(x) == 13}),
    BaseAugTest(alias=CLI_UNIT_TESTS.invalid_env_seed.value, cfg=CLICfg(compose_cfg=True, debug_mode=True,
                                                                        env_seed="oops"),
                expected={'seed_test': lambda x: int(x) >= 0}),
    BaseAugTest(alias=CLI_UNIT_TESTS.invalid_cfg_seed.value, cfg=CLICfg(compose_cfg=True, debug_mode=True),
                expected={'seed_test': lambda x: int(x) >= 0}),
    BaseAugTest(alias=CLI_UNIT_TESTS.nonint_cfg_seed.value, cfg=CLICfg(compose_cfg=True, debug_mode=True),
                expected={'seed_test': lambda x: int(x) == 14}),
    BaseAugTest(alias=CLI_UNIT_TESTS.seed_false.value, cfg=CLICfg(compose_cfg=True, debug_mode=True),
                expected={'seed_test': lambda x: x is None}),
    BaseAugTest(alias=CLI_UNIT_TESTS.seed_true.value, cfg=CLICfg(compose_cfg=True, debug_mode=True),
                expected={'seed_test': lambda x: int(x) >= 0}),
    BaseAugTest(alias=CLI_UNIT_TESTS.excess_args.value, cfg=CLICfg(compose_cfg=True, debug_mode=True,
                                                                   extra_args=["--foo"]),
                expected={'seed_test': lambda x: int(x) >= 0}),
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
