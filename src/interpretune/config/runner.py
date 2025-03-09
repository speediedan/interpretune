from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import TYPE_CHECKING, Optional, Union, List, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from interpretune.utils import rank_zero_warn, rank_zero_debug, MisconfigurationException
from interpretune.config.analysis import IT_ANALYSIS_CACHE, AnalysisCfg, AnalysisArtifactCfg

if TYPE_CHECKING:
    from interpretune.session import ITSession
    from interpretune.protocol import ITModuleProtocol, ITDataModuleProtocol, SAEAnalysisProtocol
    from interpretune.analysis import SAEAnalysisTargets, AnalysisOp


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
                raise MisconfigurationException("If not providing `it_session`, must provide both a `datamodule` and"
                                                " `module`")

    def _session_validation(self):
        if any((self.module, self.datamodule)):
            rank_zero_warn("`module`/`datamodule` should only be specified if not providing `it_session`. Attempting to"
                           " use the `module`/`datamodule` handles from `it_session`.")
        self.module = self.it_session.module
        self.datamodule = self.it_session.datamodule


@dataclass(kw_only=True)
class AnalysisRunnerCfg(SessionRunnerCfg):
    # New single attribute that can handle different types of inputs
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

    # Internal storage for processed configs
    _processed_analysis_cfgs: List[AnalysisCfg] = field(default_factory=list, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.it_session.module.analysis_run_cfg = self

        # Process analysis_cfgs if provided
        if self.analysis_cfgs is not None:
            self._process_analysis_cfgs()
        else:
            rank_zero_debug("No analysis_cfgs provided on runner initialization, expecting one to be passed with"
                            " run_analysis invocation")

        # Convert Path objects to strings if provided
        if isinstance(self.cache_dir, Path):
            self.cache_dir = str(self.cache_dir)
        if isinstance(self.op_output_dataset_path, Path):
            self.op_output_dataset_path = str(self.op_output_dataset_path)

    def _process_analysis_cfgs(self):
        """Process the analysis_cfgs input into a standardized list of AnalysisCfg objects."""
        self._processed_analysis_cfgs = []

        # Handle single AnalysisCfg
        if isinstance(self.analysis_cfgs, AnalysisCfg):
            self._processed_analysis_cfgs.append(self.analysis_cfgs)
            return

        # Handle single AnalysisOp
        if hasattr(self.analysis_cfgs, 'name') and hasattr(self.analysis_cfgs, 'alias'):
            self._processed_analysis_cfgs.append(AnalysisCfg(op=self.analysis_cfgs))
            return

        # Handle iterable of AnalysisCfg or AnalysisOp
        try:
            for cfg in self.analysis_cfgs:
                if isinstance(cfg, AnalysisCfg):
                    self._processed_analysis_cfgs.append(cfg)
                elif hasattr(cfg, 'name') and hasattr(cfg, 'alias'):  # Check if it's an AnalysisOp
                    self._processed_analysis_cfgs.append(AnalysisCfg(op=cfg))
                else:
                    raise ValueError(f"Unsupported analysis configuration type: {type(cfg)}")
        except TypeError:
            # If analysis_cfgs is not iterable
            raise ValueError(f"analysis_cfgs must be an AnalysisCfg, AnalysisOp, or an iterable of these types, "
                             f"but got {type(self.analysis_cfgs)}")

    def init_analysis_cfgs(self, module: SAEAnalysisProtocol) -> None:
        """Initialize or reinitialize analysis configurations based on current settings."""
        # Initialize directories - this will create any necessary directories and handle path conversions
        self.init_analysis_dirs(module)

        # Apply runner-level ignore_manual setting to all configs if set to True
        if self.ignore_manual:
            for cfg in self._processed_analysis_cfgs:
                cfg.ignore_manual = True

        # Apply each analysis configuration to the module
        for cfg in self._processed_analysis_cfgs:
            cfg.apply(module, self.cache_dir, self.op_output_dataset_path, self.sae_analysis_targets)

    def init_analysis_dirs(self, module: SAEAnalysisProtocol):
        """Initialize the analysis directories once a module handle is available."""
        if self.cache_dir is None:
            self.cache_dir = (Path(IT_ANALYSIS_CACHE) /
                         module.datamodule.dataset['validation'].config_name /
                         module.datamodule.dataset['validation']._fingerprint /
                         module.__class__._orig_module_name)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        if self.op_output_dataset_path is None:
            self.op_output_dataset_path = module.core_log_dir / "analysis_datasets"
        self.op_output_dataset_path = Path(self.op_output_dataset_path)
        self.op_output_dataset_path.mkdir(exist_ok=True, parents=True)

        # Check for op in analysis configurations and verify directory is empty
        for cfg in self._processed_analysis_cfgs:
            if cfg.op is not None:
                op_dir = self.op_output_dataset_path / cfg.op.name
                if op_dir.exists() and any(op_dir.iterdir()):
                    raise Exception(
                        f"Analysis dataset directory for op '{cfg.op.name}' ({op_dir}) is not empty. "
                        "Please delete it or specify a different path."
                    )
