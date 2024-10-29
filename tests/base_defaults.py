from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Any, Dict
from collections.abc import Iterable
import pytest

from interpretune.adapters import ADAPTER_REGISTRY
from interpretune.adapters.registration import Adapter
from interpretune.base.config.mixins import HFFromPretrainedConfig, ZeroShotClassificationConfig
from interpretune.extensions.memprofiler import MemProfilerCfg
from interpretune.extensions.debug_generation import DebugLMConfig
from tests.runif import RunIf, RUNIF_ALIASES


### global test defaults
default_test_bs = 2
default_prof_bs = 1

################################################################################
# Core test generation and encapsulation
################################################################################

@dataclass(kw_only=True)
class BaseAugTest:
    alias: str
    cfg: Optional[Tuple] = None
    marks: Optional[Dict] = None  # test instance-specific marks
    expected: Optional[Dict] = None
    result_gen: Optional[Callable] = None
    function_marks: Dict[str, Any] = field(default_factory=dict)  # marks applied at test function level

    def __post_init__(self):
        if self.expected is None and self.result_gen is not None:
            assert callable(self.result_gen), "result_gen must be callable"
            self.expected = self.result_gen(self.alias)
        if self.cfg is None and self.cfg_gen is not None:
            assert callable(self.cfg_gen), "cfg_gen must be callable"
            self.cfg = self.cfg_gen(self.alias)
        elif isinstance(self.cfg, Dict):
            self.cfg = self.cfg[self.alias]
        if self.marks or self.function_marks:
            self.marks = self._get_marks(self.marks, self.function_marks)

    def _get_marks(self, marks: Optional[Dict | str], function_marks: Dict) -> Optional[RunIf]:
        # support RunIf aliases applied to function level
        if marks:
            if isinstance(marks, Dict):
                function_marks.update(marks)
            elif isinstance(marks, str):
                function_marks.update(RUNIF_ALIASES[marks])
            else:
                raise ValueError(f"Unexpected marks input type (should be Dict, str or None): {type(marks)}")
        if function_marks:
            return RunIf(**function_marks)

def pytest_param_factory(test_configs: List[BaseAugTest], unpack: bool = True) -> List:
    return [pytest.param(
            config.alias,
            *config.cfg if unpack else (config.cfg,),
            id=config.alias,
            marks=config.marks or tuple(),
        )
        for config in test_configs
    ]

@dataclass(kw_only=True)
class BaseCfg:
    phase: str = "train"
    device_type: str = "cpu"
    model_key: str = "rte"  # "real-model"-based acceptance/parity testing/profiling
    precision: str | int = 32
    adapter_ctx: Iterable[Adapter | str] = (Adapter.core,)
    model_src_key: Optional[str] = None
    limit_train_batches: Optional[int] = 1
    limit_val_batches: Optional[int] = 1
    limit_test_batches: Optional[int] = 1
    dm_override_cfg: Optional[Dict] = None
    zero_shot_cfg: Optional[ZeroShotClassificationConfig] = None
    hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = None
    memprofiler_cfg: Optional[MemProfilerCfg] = None
    debug_lm_cfg: Optional[DebugLMConfig] = None
    model_cfg: Optional[Dict] = None
    tl_cfg: Optional[Dict] = None
    sl_cfg: Optional[Dict] = None
    max_epochs: Optional[int] = 1
    cust_fwd_kwargs: Optional[Dict] = None
    # used when adding a new test dataset or changing a test model to force re-caching of test datasets
    force_prepare_data: bool = False  # TODO: make this settable via an env variable as well
    max_steps: Optional[int] = None
    save_checkpoints: bool = False

    def __post_init__(self):
        self.adapter_ctx = ADAPTER_REGISTRY.canonicalize_composition(self.adapter_ctx)
        self.max_steps = self.max_steps or self.limit_train_batches
