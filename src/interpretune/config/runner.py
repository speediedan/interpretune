from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import TYPE_CHECKING, Optional, Union, List, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from interpretune.utils import rank_zero_warn, rank_zero_debug, MisconfigurationException
from interpretune.analysis import AnalysisOp
from interpretune.analysis.cache import get_analysis_cache_dir
from interpretune.config.analysis import AnalysisCfg, AnalysisArtifactCfg, SAEAnalysisTargets
from interpretune.protocol import StrOrPath

if TYPE_CHECKING:
    from interpretune.session import ITSession
    from interpretune.protocol import ITModuleProtocol, ITDataModuleProtocol, SAEAnalysisModuleProtocol


# Standalone functions for analysis initialization
def to_analysis_cfgs(
    analysis_cfgs: Optional[Union[AnalysisCfg, "AnalysisOp", Iterable[Union[AnalysisCfg, "AnalysisOp"]]]],
) -> List[AnalysisCfg]:
    """Convert various input formats to a list of AnalysisCfg objects.

    Args:
        analysis_cfgs: Input that can be:
            - An AnalysisCfg instance
            - An AnalysisOp instance
            - An iterable of AnalysisCfg or AnalysisOp instances
            - None (returns empty list)

    Returns:
        List of standardized AnalysisCfg objects

    Raises:
        ValueError: If the input type is not supported
    """
    processed_cfgs = []

    # Handle None case
    if analysis_cfgs is None:
        return processed_cfgs

    # Handle single AnalysisCfg
    if isinstance(analysis_cfgs, AnalysisCfg):
        processed_cfgs.append(analysis_cfgs)
        return processed_cfgs

    # Handle single AnalysisOp
    if hasattr(analysis_cfgs, "name") and hasattr(analysis_cfgs, "alias"):
        processed_cfgs.append(AnalysisCfg(target_op=analysis_cfgs))  # type: ignore[arg-type]
        return processed_cfgs

    # Handle iterable of AnalysisCfg or AnalysisOp
    try:
        # Check if it's iterable first
        iter(analysis_cfgs)  # type: ignore[call-overload]
        for cfg in analysis_cfgs:  # type: ignore[union-attr]
            if isinstance(cfg, AnalysisCfg):
                processed_cfgs.append(cfg)
            elif hasattr(cfg, "name") and hasattr(cfg, "alias"):  # Check if it's an AnalysisOp
                processed_cfgs.append(AnalysisCfg(target_op=cfg))
            else:
                raise ValueError(f"Unsupported analysis configuration type: {type(cfg)}")
    except TypeError:
        # If analysis_cfgs is not iterable
        raise ValueError(
            f"analysis_cfgs must be an AnalysisCfg, AnalysisOp, or an iterable of these types, "
            f"but got {type(analysis_cfgs)}"
        )

    return processed_cfgs


def init_analysis_dirs(
    module: "SAEAnalysisModuleProtocol",
    cache_dir: Optional[Union[str, Path]] = None,
    op_output_dataset_path: Optional[Union[str, Path]] = None,
    analysis_cfgs: Optional[List[AnalysisCfg]] = None,
) -> tuple[Path, Path]:
    """Initialize the analysis directories for the given module and analysis configurations.

    Args:
        module: The module to set up analysis directories for
        cache_dir: Optional path to cache directory, will be created if it doesn't exist
        op_output_dataset_path: Optional path for analysis outputs, will be created if it doesn't exist
        analysis_cfgs: Optional list of analysis configurations to check for op directories

    Returns:
        Tuple of (cache_dir, op_output_dataset_path) as Path objects
    """
    # Setup cache directory using analysis/cache helper which selects a temporary
    # directory by default and a permanent path under IT_ANALYSIS_CACHE only
    # when persistent analysisstore caching is enabled.
    cache_dir = get_analysis_cache_dir(module, explicit_cache_dir=cache_dir)
    assert isinstance(cache_dir, StrOrPath), "cache_dir must be a str or Path"
    cache_dir = Path(cache_dir)
    # Creation is handled by get_analysis_cache_dir for permanent dirs; ensure
    # temp dirs exist as well.
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Setup output dataset path
    if op_output_dataset_path is None:
        op_output_dataset_path = module.core_log_dir / "analysis_datasets"  # type: ignore[attr-defined]  # protocol provides core_log_dir
    assert isinstance(op_output_dataset_path, StrOrPath), "op_output_dataset_path must be a str or Path"
    op_output_dataset_path = Path(op_output_dataset_path)
    op_output_dataset_path.mkdir(exist_ok=True, parents=True)

    # Check for op in analysis configurations and verify directory is empty
    if analysis_cfgs:
        for cfg in analysis_cfgs:
            if cfg.op is not None:
                # Accept either AnalysisOp instances or OpWrapper-like objects / simple objects providing a `name`
                if not (isinstance(cfg.op, AnalysisOp) or hasattr(cfg.op, "name")):
                    raise AssertionError("cfg.op must be an AnalysisOp instance or provide a `name` attribute")
                op_dir = op_output_dataset_path / cfg.op.name  # type: ignore[attr-defined]  # validated above to have name
                if op_dir.exists() and any(op_dir.iterdir()):
                    raise Exception(
                        f"Analysis dataset directory for op '{cfg.op.name}' ({op_dir}) is not empty. "  # type: ignore[attr-defined]  # validated above to have name
                        "Please delete it or specify a different path."
                    )

    return cache_dir, op_output_dataset_path


def init_analysis_cfgs(
    module: "SAEAnalysisModuleProtocol",
    analysis_cfgs: List[AnalysisCfg],
    cache_dir: Optional[Union[str, Path]] = None,
    op_output_dataset_path: Optional[Union[str, Path]] = None,
    sae_analysis_targets: Optional["SAEAnalysisTargets"] = None,
    ignore_manual: bool = False,
) -> None:
    """Initialize analysis configurations for the given module.

    Args:
        module: The module to initialize configurations for
        analysis_cfgs: List of analysis configurations to initialize
        cache_dir: Optional path to cache directory
        op_output_dataset_path: Optional path for analysis outputs
        sae_analysis_targets: Optional analysis targets to use
        ignore_manual: Whether to ignore existing manual analysis steps
    """
    # Initialize directories
    cache_dir, op_output_dataset_path = init_analysis_dirs(module, cache_dir, op_output_dataset_path, analysis_cfgs)

    # Apply ignore_manual setting if specified
    if ignore_manual:
        for cfg in analysis_cfgs:
            cfg.ignore_manual = True

    # Apply each analysis configuration to the module, respecting already-applied configs
    for cfg in analysis_cfgs:
        if not cfg.applied_to(module):
            cfg.apply(module, str(cache_dir), str(op_output_dataset_path), sae_analysis_targets)


@dataclass(kw_only=True)
class SessionRunnerCfg:
    it_session: ITSession | None = None
    module: ITModuleProtocol | None = None
    datamodule: ITDataModuleProtocol | None = None
    limit_train_batches: int = -1
    limit_val_batches: int = -1
    limit_test_batches: int = -1
    max_steps: int = -1
    max_epochs: int = -1

    def __post_init__(self):
        if self.it_session is not None:
            self._session_validation()
        else:
            if not all((self.module, self.datamodule)):
                raise MisconfigurationException(
                    "If not providing `it_session`, must provide both a `datamodule` and `module`"
                )

    def _session_validation(self):
        if any((self.module, self.datamodule)):
            rank_zero_warn(
                "`module`/`datamodule` should only be specified if not providing `it_session`. Attempting to"
                " use the `module`/`datamodule` handles from `it_session`."
            )
        assert self.it_session is not None
        self.module = self.it_session.module
        self.datamodule = self.it_session.datamodule


@dataclass(kw_only=True)
class AnalysisRunnerCfg(SessionRunnerCfg):
    # Change the field to a private attribute that will store the raw value
    analysis_cfgs: Optional[Union[AnalysisCfg, AnalysisOp, Iterable[Union[AnalysisCfg, AnalysisOp]]]] = None
    limit_analysis_batches: int = -1
    cache_dir: Optional[str | Path] = None
    op_output_dataset_path: Optional[str | Path] = None
    # Add optional sae_analysis_targets as a fallback
    sae_analysis_targets: Optional[SAEAnalysisTargets] = None
    # Add artifact configuration
    artifact_cfg: AnalysisArtifactCfg = field(default_factory=AnalysisArtifactCfg)
    # Global override for ignore_manual setting in analysis configs
    ignore_manual: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.it_session is not None and self.it_session.module is not None
        self.it_session.module.analysis_run_cfg = self  # type: ignore[attr-defined]

        # No need to call _process_analysis_cfgs() as it's now a property
        if self.analysis_cfgs is None:
            rank_zero_debug(
                "No analysis_cfgs provided on runner initialization, expecting one to be passed with"
                " run_analysis invocation"
            )

        # Convert Path objects to strings if provided
        if isinstance(self.cache_dir, Path):
            self.cache_dir = str(self.cache_dir)
        if isinstance(self.op_output_dataset_path, Path):
            self.op_output_dataset_path = str(self.op_output_dataset_path)

    @property
    def _processed_analysis_cfgs(self) -> List[AnalysisCfg]:
        """Process and return the analysis_cfgs as a standardized list of AnalysisCfg objects."""
        return to_analysis_cfgs(self.analysis_cfgs)
