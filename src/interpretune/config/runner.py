from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field
from pathlib import Path

import interpretune as it
from interpretune.utils import rank_zero_warn, rank_zero_debug, MisconfigurationException
from interpretune.config.analysis import IT_ANALYSIS_CACHE, AnalysisSetCfg, AnalysisCfg, AnalysisArtifactCfg


if TYPE_CHECKING:
    from interpretune.session import ITSession
    from interpretune.protocol import ITModuleProtocol, ITDataModuleProtocol, SAEAnalysisProtocol
    from interpretune.analysis import AnalysisOp, SAEAnalysisTargets


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
    # Allow one of analysis_set_cfg, analysis_cfg, or analysis_op
    analysis_set_cfg: AnalysisSetCfg | None = None
    analysis_cfg: AnalysisCfg | None = None
    analysis_op: AnalysisOp | None = None
    limit_analysis_batches: int = -1
    cache_dir: Optional[str | Path] = None
    op_output_dataset_path: Optional[str | Path] = None
    # Add optional sae_analysis_targets as a fallback
    sae_analysis_targets: Optional[SAEAnalysisTargets] = None
    # Add artifact configuration
    artifact_cfg: AnalysisArtifactCfg = field(default_factory=AnalysisArtifactCfg)

    def __post_init__(self):
        super().__post_init__()
        self.it_session.module.analysis_run_cfg = self

        # Validate that at most one of analysis_set_cfg, analysis_cfg, or analysis_op is provided
        provided_options = sum(1 for option in [self.analysis_set_cfg, self.analysis_cfg,
                                                self.analysis_op] if option is not None)
        if provided_options > 1:
            raise MisconfigurationException("Only one of analysis_set_cfg, analysis_cfg, or analysis_op should be"
                                            " provided")
        if provided_options == 0:
            rank_zero_debug("No analysis_cfg provided on runner initialization, expecting one to be passed with"
                            " run_analysis invocation")

        # Convert analysis_op to analysis_cfg if provided
        if self.analysis_op is not None:
            self.analysis_cfg = AnalysisCfg(op=self.analysis_op)

        # Convert Path objects to strings if provided
        if isinstance(self.cache_dir, Path):
            self.cache_dir = str(self.cache_dir)
        if isinstance(self.op_output_dataset_path, Path):
            self.op_output_dataset_path = str(self.op_output_dataset_path)

    def init_analysis_cfgs(self, module: SAEAnalysisProtocol) -> None:
        """Initialize or reinitialize analysis configurations based on current settings."""
        # Initialize directories - this will create any necessary directories and handle path conversions
        self.init_analysis_dirs(module)

        # Configure analysis based on which configuration type is provided
        if self.analysis_set_cfg is not None:
            self.init_set_analysis_cfgs(module)
        elif self.analysis_cfg is not None:
            #self.init_single_analysis_cfg(module)
            self.analysis_cfg.apply(module, self.cache_dir, self.op_output_dataset_path, self.sae_analysis_targets)
        elif self.analysis_op is not None:
            # Convert analysis_op to analysis_cfg if it hasn't been done yet
            if self.analysis_cfg is None:
                self.analysis_cfg = AnalysisCfg(op=self.analysis_op)
            self.analysis_cfg.apply(module, self.cache_dir, self.op_output_dataset_path, self.sae_analysis_targets)

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
        if (self.analysis_set_cfg is not None and
            self.analysis_set_cfg.analysis_ops is not None):
            for op in self.analysis_set_cfg.analysis_ops:
                op_dir = self.op_output_dataset_path / op.name
                if op_dir.exists() and any(op_dir.iterdir()):
                    raise Exception(
                        f"Analysis dataset directory for op '{op.name}' ({op_dir}) is not empty. "
                        "Please delete it or specify a different path."
                    )
        elif (self.analysis_cfg is not None and
              self.analysis_cfg.op is not None):
            op_dir = self.op_output_dataset_path / self.analysis_cfg.op.name
            if op_dir.exists() and any(op_dir.iterdir()):
                raise Exception(
                    f"Analysis dataset directory for op '{self.analysis_cfg.op.name}' ({op_dir}) is not empty. "
                    "Please delete it or specify a different path."
                )

    def init_set_analysis_cfgs(self, module: SAEAnalysisProtocol):
        """Initialize analysis configurations."""
        if self.analysis_set_cfg is None:
            return
        # Check if logit_diffs_sae is enabled
        clean_w_sae_enabled = (self.analysis_set_cfg.analysis_ops is not None and
                               it.logit_diffs_sae in self.analysis_set_cfg.analysis_ops)

        prompts_tokens_cfg = dict(save_prompts=not clean_w_sae_enabled, save_tokens=not clean_w_sae_enabled)

        # Handle existing analysis_cfgs
        if self.analysis_set_cfg.analysis_cfgs:
            for op, cfg in self.analysis_set_cfg.analysis_cfgs.items():
                # Apply analysis cfg to module to prepare it for op execution, passing runner's sae_analysis_targets
                cfg.apply(module, self.cache_dir, self.op_output_dataset_path,
                          self.sae_analysis_targets)

        # Generate analysis_cfgs from analysis_ops
        elif self.analysis_set_cfg.analysis_ops:
            for op in self.analysis_set_cfg.analysis_ops:
                # Configure AnalysisCfg for the user based on the provided op with default settings
                cfg = AnalysisCfg(
                    op=op,
                    sae_analysis_targets=self.sae_analysis_targets,
                    save_prompts=True if op == it.logit_diffs_sae else prompts_tokens_cfg["save_prompts"],
                    save_tokens=True if op == it.logit_diffs_sae else prompts_tokens_cfg["save_tokens"],
                )

                # Apply analysis cfg to module to prepare it for op execution, passing runner's sae_analysis_targets
                cfg.apply(module, self.cache_dir, self.op_output_dataset_path, self.sae_analysis_targets)

                # Add to analysis_cfgs
                self.analysis_set_cfg.analysis_cfgs[op] = cfg
