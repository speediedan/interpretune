from collections.abc import Sequence
from copy import deepcopy
from enum import auto
from dataclasses import dataclass, field
from typing import Iterable, Union


import interpretune as it
from interpretune.protocol import Adapter, AutoStrEnum
from interpretune.config import (HFFromPretrainedConfig, GenerativeClassificationConfig, ITLensCustomConfig,
                                TLensGenerationConfig, AutoCompConfig, ITLensFromPretrainedNoProcessingConfig,
                                SAELensFromPretrainedConfig, AnalysisCfg)
from interpretune.extensions import DebugLMConfig, MemProfilerCfg
from interpretune.analysis import SAEAnalysisTargets, AnalysisOp
from it_examples.experiments.rte_boolq import RTEBoolqEntailmentMapping
from tests.base_defaults import BaseAugTest, BaseCfg, AnalysisBaseCfg
from tests.parity_acceptance.cfg_aliases import parity_cli_cfgs, mod_initargs, CLI_TESTS
from tests.parity_acceptance.test_it_tl import TLParityCfg
from tests.parity_acceptance.test_it_cli import CLICfg
from tests.utils import get_nested


nf4_bnb_config = {"load_in_4bit": True, "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4",
                   "bnb_4bit_compute_dtype": "bfloat16"}
base_lora_cfg = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM"}
gpt2_lora_cfg = {"target_modules": ["c_attn", "c_proj"], **base_lora_cfg}
gpt2_hf_bnb_lora_cfg = {"bitsandbytesconfig": nf4_bnb_config, "lora_cfg": gpt2_lora_cfg}
gpt2_hf_bnb_lora_cfg_seq = {"bitsandbytesconfig": nf4_bnb_config, "lora_cfg": {**gpt2_lora_cfg, "task_type": "SEQ_CLS"}}
gpt2_seq_hf_from_pretrained_kwargs = {"pretrained_kwargs": {"device_map": "cpu", "torch_dtype": "float32"},
                                      "model_head": "transformers.AutoModelForSequenceClassification"}
tl_cust_mi_cfg = {
    "default_padding_side":"left",
    "cfg":
    dict(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=200,
    act_fn="relu",
    attention_dir="causal",
    tokenizer_name='gpt2',
    seed=1,
    use_attn_result=True,
)}

test_tl_cust_2L_config = {"n_layers":2, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
                          "act_fn":'relu', "tokenizer_name": 'gpt2'}

@dataclass(kw_only=True)
class CoreCfgForcePrepare(BaseCfg):
    model_src_key: str | None = "cust"
    force_prepare_data: bool | None = True
    dm_override_cfg: dict | None = field(default_factory=lambda: {'enable_datasets_cache': False,
                                                                  'dataset_path': '/tmp/force_prepare_tests_ds'})

@dataclass(kw_only=True)
class TLMechInterpCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.transformer_lens)
    model_src_key: str | None = "cust"
    force_prepare_data: bool | None = True
    tl_cfg: dict | None = field(default_factory=lambda: ITLensCustomConfig(**tl_cust_mi_cfg))
    dm_override_cfg: dict | None = field(default_factory=lambda: {'enable_datasets_cache': False, 'tokenizer_kwargs': {
        'padding_side': 'right', 'model_input_names': ['input']}})

@dataclass(kw_only=True)
class TLDebugCfg(TLParityCfg):
    debug_lm_cfg: DebugLMConfig | None = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"

@dataclass(kw_only=True)
class CoreGPT2PEFTCfg(BaseCfg):
    device_type: str = "cuda"
    model_src_key: str | None = "gpt2"
    dm_override_cfg: dict | None = field(default_factory=lambda: {'train_batch_size': 1, 'eval_batch_size': 1})
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = field(default_factory=lambda: HFFromPretrainedConfig(
        **gpt2_hf_bnb_lora_cfg, pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
        model_head="transformers.GPT2LMHeadModel", activation_checkpointing=True))
    limit_train_batches: int | None = 3
    limit_val_batches: int | None = 3
    limit_test_batches: int | None = 2

@dataclass(kw_only=True)
class CoreGPT2PEFTSeqCfg(CoreGPT2PEFTCfg):
    phase: str | None = "test"
    generative_step_cfg: GenerativeClassificationConfig | None = \
        field(default_factory=lambda: GenerativeClassificationConfig(enabled=False))
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = field(default_factory=lambda: HFFromPretrainedConfig(
        **gpt2_hf_bnb_lora_cfg_seq, **gpt2_seq_hf_from_pretrained_kwargs, activation_checkpointing=True))

@dataclass(kw_only=True)
class CoreMemProfCfg(BaseCfg):
    memprofiler_cfg: MemProfilerCfg | None = field(default_factory=lambda: MemProfilerCfg(enabled=True,
                                                                                         cuda_allocator_history=True))
    dm_override_cfg: dict | None = field(default_factory=lambda: {'train_batch_size': 1, 'eval_batch_size': 1})
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
class LightningGPT2(BaseCfg):
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning,)
    model_cfg: dict | None = field(default_factory=lambda: {"tie_word_embeddings": False})

@dataclass(kw_only=True)
class LightningTLGPT2(BaseCfg):
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)

@dataclass(kw_only=True)
class CoreSLGPT2(BaseCfg):
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    # force_prepare_data: Optional[bool] = True  # sometimes useful to enable for test debugging

@dataclass(kw_only=True)
class CoreSLGPT2Analysis(AnalysisBaseCfg):
    phase: str | None = "analysis"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    generative_step_cfg: GenerativeClassificationConfig = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)
        )
    )
    sae_analysis_targets: SAEAnalysisTargets = field(
        default_factory=lambda: SAEAnalysisTargets(
            sae_release="gpt2-small-hook-z-kk",
            target_layers=[9, 10]
        )
    )
    hf_from_pretrained_cfg: HFFromPretrainedConfig = field(
        default_factory=lambda: HFFromPretrainedConfig(
            pretrained_kwargs={'torch_dtype': 'float32'},
            model_head='transformers.GPT2LMHeadModel'
        )
    )
    tl_cfg: ITLensFromPretrainedNoProcessingConfig = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gpt2-small",
            default_padding_side='left'
        )
    )
    sae_cfgs: list = field(default_factory=lambda: [])
    auto_comp_cfg: AutoCompConfig = field(default_factory=lambda: AutoCompConfig(
        module_cfg_name='RTEBoolqConfig', module_cfg_mixin=RTEBoolqEntailmentMapping))
    # TODO: customize these cache paths for testing efficiency
    # cache_dir: Optional[str] = None
    # op_output_dataset_path: Optional[str] = None
    # force_prepare_data: Optional[bool] = True  # sometimes useful to enable for test debugging

    def __post_init__(self):
        super().__post_init__()
        # Dynamically generate sae_cfgs from sae_targets.sae_fqns
        if self.sae_analysis_targets and hasattr(self.sae_analysis_targets, 'sae_fqns'):
            self.sae_cfgs = [SAELensFromPretrainedConfig(release=sae_fqn.release, sae_id=sae_fqn.sae_id)
                            for sae_fqn in self.sae_analysis_targets.sae_fqns]

@dataclass(kw_only=True)
class CoreSLGPT2LogitDiffsBase(CoreSLGPT2Analysis):
    analysis_cfgs: Union[AnalysisCfg, AnalysisOp, Iterable[Union[AnalysisCfg, AnalysisOp]]] = \
          (AnalysisCfg(target_op=it.logit_diffs_base, save_prompts=False, save_tokens=False, ignore_manual=True),)


@dataclass(kw_only=True)
class CoreSLGPT2LogitDiffsSAE(CoreSLGPT2Analysis):
    analysis_cfgs: Union[AnalysisCfg, AnalysisOp, Iterable[Union[AnalysisCfg, AnalysisOp]]] = \
          (AnalysisCfg(target_op=it.logit_diffs_sae, save_prompts=True, save_tokens=True, ignore_manual=True),)

@dataclass(kw_only=True)
class CoreSLGPT2LogitDiffsAttrGrad(CoreSLGPT2Analysis):
    analysis_cfgs: Union[AnalysisCfg, AnalysisOp, Iterable[Union[AnalysisCfg, AnalysisOp]]] = \
          (AnalysisCfg(target_op=it.logit_diffs_attr_grad, save_prompts=False, save_tokens=False, ignore_manual=True),)

@dataclass(kw_only=True)
class CoreSLGPT2LogitDiffsAttrAblation(CoreSLGPT2Analysis):
    analysis_cfgs: Union[AnalysisCfg, AnalysisOp, Iterable[Union[AnalysisCfg, AnalysisOp]]] = \
          (AnalysisCfg(target_op=it.logit_diffs_attr_ablation, save_prompts=False, save_tokens=False,
                       ignore_manual=True),)

@dataclass(kw_only=True)
class CoreSLCust(BaseCfg):
    phase: str | None = "test"
    model_src_key: str | None = "cust"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    tl_cfg: dict | None = field(default_factory=lambda: ITLensCustomConfig(cfg=test_tl_cust_2L_config))

@dataclass(kw_only=True)
class LightningSLGPT2(BaseCfg):
    phase: str | None = "test"
    model_src_key: str | None = "gpt2"
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
