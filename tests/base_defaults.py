from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Any, Dict, Sequence, TYPE_CHECKING, Iterable, Union
import pytest

from interpretune.adapters import ADAPTER_REGISTRY
from interpretune.config import HFFromPretrainedConfig, GenerativeClassificationConfig, AutoCompConfig
from interpretune.extensions import MemProfilerCfg, DebugLMConfig
from interpretune.protocol import Adapter
from interpretune.analysis import SAEAnalysisTargets
from tests.runif import RunIf, RUNIF_ALIASES

if TYPE_CHECKING:
    from interpretune.analysis import AnalysisOp, AnalysisCfg

default_test_task = "rte"

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

def pytest_factory(test_configs: List[BaseAugTest], unpack: bool = True, fq_alias: bool = False) -> List:
    return [pytest.param(
            config.alias,
            *config.cfg if unpack else (config.cfg,),
            id=config.alias if not fq_alias else config.alias.split(".")[-1],
            marks=config.marks or tuple(),
        )
        for config in test_configs
    ]

@dataclass(kw_only=True)
class BaseCfg:
    phase: str = "train"
    device_type: str = "cpu"
    model_key: str = default_test_task  # "real-model"-based acceptance/parity testing/profiling
    precision: str | int = "torch.float32"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core,)
    model_src_key: Optional[str] = None
    limit_train_batches: Optional[int] = 1
    limit_val_batches: Optional[int] = 1
    limit_test_batches: Optional[int] = 1
    dm_override_cfg: Optional[Dict] = None
    generative_step_cfg: Optional[GenerativeClassificationConfig] = None
    hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = None
    memprofiler_cfg: Optional[MemProfilerCfg] = None
    debug_lm_cfg: Optional[DebugLMConfig] = None
    model_cfg: Optional[Dict] = None
    tl_cfg: Optional[Dict] = None
    sae_cfgs: Optional[Dict] = None
    auto_comp_cfg: Optional[AutoCompConfig] = None
    add_saes_on_init: bool = False
    req_grad_mask: Optional[Tuple] = None  # used to toggle requires grad for non-fts contexts
    max_epochs: Optional[int] = 1
    cust_fwd_kwargs: Optional[Dict] = None
    # used when adding a new test dataset or changing a test model to force re-caching of test datasets
    force_prepare_data: bool = False  # TODO: make this settable via an env variable as well
    max_steps: Optional[int] = None
    save_checkpoints: bool = False

    def __post_init__(self):
        self.adapter_ctx = ADAPTER_REGISTRY.canonicalize_composition(self.adapter_ctx)
        self.max_steps = self.max_steps or self.limit_train_batches

@dataclass(kw_only=True)
class AnalysisBaseCfg(BaseCfg):
    # TODO: we may want to narrow Iterable to Sequence here
    analysis_cfgs: Union['AnalysisCfg', 'AnalysisOp', Iterable[Union['AnalysisCfg', 'AnalysisOp']]] = None
    limit_analysis_batches: int = 2
    cache_dir: Optional[str] = None
    op_output_dataset_path: Optional[str] = None
    # Add optional sae_analysis_targets as a fallback
    sae_analysis_targets: Optional[SAEAnalysisTargets] = None
    # Add artifact configuration
    artifact_cfg: Optional[Dict] = None
    # Global override for ignore_manual setting in analysis configs
    ignore_manual: bool = False

    def __post_init__(self):
        super().__post_init__()

@dataclass(kw_only=True)
class OpTestConfig:
    """Configuration for operation testing."""
    target_op: Any  # The operation to test
    resolved_op: Optional['AnalysisOp'] = None
    session_fixt: str = "get_it_session__sl_gpt2_analysis__setup"
    batch_size: int = 1
    generate_required_only: bool = True
    override_req_cols: Optional[tuple] = None
    deepcopy_session_fixt: bool = False
