from typing import Optional
from dataclasses import dataclass, field

from tests.parity_acceptance.plugins.transformer_lens.test_interpretune_tl import TLParityCfg
from tests.configuration import BaseCfg
from interpretune.analysis.debug_generation import DebugLMConfig


@dataclass(kw_only=True)
class TLDebugCfg(TLParityCfg):
    debug_lm_cfg: Optional[DebugLMConfig] = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: Optional[str] = "test"
    model_src_key: Optional[str] = "pretrained"

@dataclass(kw_only=True)
class CoreLlama2DebugCfg(BaseCfg):
    debug_lm_cfg: Optional[DebugLMConfig] = field(default_factory=lambda: DebugLMConfig(enabled=True))
    phase: Optional[str] = "test"
    device_type: Optional[str] = "cuda"
    model_src_key: Optional[str] = "pretrained"
