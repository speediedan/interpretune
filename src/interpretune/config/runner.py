from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import TYPE_CHECKING
from dataclasses import dataclass

from interpretune.utils import rank_zero_warn, MisconfigurationException

if TYPE_CHECKING:
    from interpretune.session import ITSession
    from interpretune.config.analysis import AnalysisSetCfg
    from interpretune.protocol import ITModuleProtocol, ITDataModuleProtocol


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
    # for now, we require that the user provide an AnalysisSetCfg to run analysis
    analysis_set_cfg: AnalysisSetCfg
    limit_analysis_batches: int = -1

    def __post_init__(self):
        super().__post_init__()
        self.it_session.module.analysis_run_cfg = self
        if not self.analysis_set_cfg:
            raise MisconfigurationException("AnalysisSetCfg must be provided to run analysis")
        if (hasattr(self.analysis_set_cfg, "limit_analysis_batches") and
            self.analysis_set_cfg.limit_analysis_batches is not None):
            self.limit_analysis_batches = self.analysis_set_cfg.limit_analysis_batches
