from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Any, Dict, Sequence, TYPE_CHECKING, Iterable
from interpretune.config.transformer_lens import ITLensCfg
import pytest

from interpretune.adapters import ADAPTER_REGISTRY
from interpretune.config import (
    HFFromPretrainedConfig,
    GenerativeClassificationConfig,
    AutoCompConfig,
    AnalysisCfg,
    CircuitTracerConfig,
)
from interpretune.extensions import MemProfilerCfg, DebugLMConfig
from interpretune.protocol import Adapter
from interpretune.analysis import SAEAnalysisTargets
from tests.runif import RunIf, RUNIF_ALIASES

if TYPE_CHECKING:
    from interpretune.analysis import AnalysisOp

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
    cfg: Tuple | None = None
    marks: Dict | None = None  # test instance-specific marks
    expected: Dict | None = None
    result_gen: Callable | None = None
    function_marks: dict[str, Any] = field(default_factory=dict)  # marks applied at test function level

    def __post_init__(self):
        if self.expected is None and self.result_gen is not None:
            assert callable(self.result_gen), "result_gen must be callable"
            self.expected = self.result_gen(self.alias)
        elif isinstance(self.cfg, Dict):
            self.cfg = self.cfg[self.alias]
        if self.marks or self.function_marks:
            self.marks = self._get_marks(self.marks, self.function_marks)

    def _get_marks(self, marks: Dict | str | None, function_marks: Dict) -> RunIf | None:
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


def pytest_factory(test_configs: list[BaseAugTest], unpack: bool = True, fq_alias: bool = False) -> List:
    return [
        pytest.param(
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
    model_src_key: str | None = None
    model_cfg_key: str = default_test_task  # default model cfg, "real-model"-based acceptance/parity testing/profiling
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core,)
    datamodule_cls: str | None = None  # Fully qualified class name (e.g., "tests.modules.DivergeTestITModule")
    module_cls: str | None = None  # Fully qualified class name (e.g., "tests.modules.DivergeTestITModule")
    precision: str | int = "torch.float32"
    limit_train_batches: int | None = 1
    limit_val_batches: int | None = 1
    limit_test_batches: int | None = 1
    dm_override_cfg: Dict | None = None
    generative_step_cfg: GenerativeClassificationConfig | None = None
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = None
    memprofiler_cfg: MemProfilerCfg | None = None
    debug_lm_cfg: DebugLMConfig | None = None
    model_cfg: Dict | None = None
    tl_cfg: ITLensCfg | None = None
    circuit_tracer_cfg: CircuitTracerConfig | None = None
    sae_cfgs: Dict | None = None
    auto_comp_cfg: AutoCompConfig | None = None
    add_saes_on_init: bool = False
    req_grad_mask: Tuple | None = None  # used to toggle requires grad for non-fts contexts
    max_epochs: int | None = 1
    cust_fwd_kwargs: Dict | None = None
    # used when adding a new test dataset or changing a test model to force re-caching of test datasets
    force_prepare_data: bool = False  # TODO: make this settable via an env variable as well
    max_steps: int | None = None
    save_checkpoints: bool = False
    req_deterministic: bool = False
    logging_level: str | int = "INFO"  # Logging level for test runs

    def __post_init__(self):
        self.adapter_ctx = ADAPTER_REGISTRY.canonicalize_composition(self.adapter_ctx)
        self.max_steps = self.max_steps or self.limit_train_batches


@dataclass(kw_only=True)
class AnalysisBaseCfg(BaseCfg):
    # TODO: we may want to narrow Iterable to Sequence here
    analysis_cfgs: AnalysisCfg | AnalysisOp | Iterable[AnalysisCfg | AnalysisOp] = None
    limit_analysis_batches: int = 2
    cache_dir: str | None = None
    op_output_dataset_path: str | None = None
    # Add optional sae_analysis_targets as a fallback
    sae_analysis_targets: SAEAnalysisTargets | None = None
    # Add artifact configuration
    artifact_cfg: Dict | None = None
    # Global override for ignore_manual setting in analysis configs
    ignore_manual: bool = False

    def __post_init__(self):
        super().__post_init__()


@dataclass(kw_only=True)
class OpTestConfig:
    """Configuration for operation testing."""

    target_op: Any  # The operation to test
    resolved_op: AnalysisOp | None = None
    session_fixt: str = "get_it_session__sl_gpt2_analysis__setup"
    batch_size: int = 1
    generate_required_only: bool = True
    override_req_cols: tuple | None = None
    deepcopy_session_fixt: bool = False
