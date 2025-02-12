from typing import Any
import logging
from functools import partialmethod
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from interpretune.base.config.shared import AllPhases, AnalysisMode, CorePhases
from interpretune.base.datamodules import ITDataModule
from interpretune.adapters.core import ITModule
from interpretune.base.contract.protocol import ITModuleProtocol, ITDataModuleProtocol
from interpretune.base.contract.session import ITSession
from interpretune.base.contract.analysis import NamesFilter
from interpretune.base.analysis import AnalysisSetCfg, AnalysisCache
from interpretune.base.config.mixins import AnalysisCfg
from interpretune.base.call import _call_itmodule_hook, it_init, it_session_end
from interpretune.utils.types import Optimizable
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.exceptions import MisconfigurationException


log = logging.getLogger(__name__)


def core_train_loop(module: ITModule, datamodule: ITDataModule, limit_train_batches: int, limit_val_batches: int,
    max_epochs: int, *args, **kwargs):
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # TODO: add optimizers property setter to corehelperattributes
    optim = module.optimizers[0]
    train_ctx = {"module": module, "optimizer": optim}
    for epoch_idx in range(max_epochs):
        module.model.train()
        module.current_epoch = epoch_idx
        _call_itmodule_hook(module, hook_name="on_train_epoch_start", hook_msg="Running train epoch start hooks")
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= limit_train_batches >= 0:
                break
            run_step(step_fn="training_step", batch=batch, batch_idx=batch_idx, **train_ctx)
        if val_dataloader is not None:
            module.model.eval()
            for batch_idx, batch in enumerate(val_dataloader):
                with torch.inference_mode():
                    if batch_idx >= limit_val_batches >= 0:
                        break
                    run_step(step_fn="validation_step", batch=batch, batch_idx=batch_idx, **train_ctx)
        module.model.train()
        _call_itmodule_hook(module, hook_name="on_train_epoch_end", hook_msg="Running train epoch end hooks")

def core_test_loop(module: ITModule, datamodule: ITDataModule, limit_test_batches: int, *args, **kwargs):
    dataloader = datamodule.test_dataloader()
    test_ctx = {"module": module}
    module._it_state._current_epoch = 0
    module.model.eval()
    for batch_idx, batch in enumerate(dataloader):
        with torch.inference_mode():
            if batch_idx >= limit_test_batches >= 0:
                break
            run_step(step_fn="test_step", batch=batch, batch_idx=batch_idx, **test_ctx)
    _call_itmodule_hook(module, hook_name="on_test_epoch_end", hook_msg="Running test epoch end hooks")

def core_analysis_loop(module: ITModule, datamodule: ITDataModule, limit_analysis_batches: int = -1,
                       step_fn: str = "analysis_step", max_epochs: int = 1, *args, **kwargs):
    _call_itmodule_hook(module, hook_name="on_analysis_start", hook_msg="Running analysis start hooks")
    # TODO: probably better to make this a method on the AnalysisStepMixin
    # TODO: allow for custom dataloader associations
    dataloader = datamodule.test_dataloader()
    test_ctx = {"module": module}
    for epoch_idx in range(max_epochs):
        module.current_epoch = epoch_idx
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            if batch_idx >= limit_analysis_batches >= 0:
                break
            if module.analysis_cfg.mode != AnalysisMode.attr_patching:
                with torch.inference_mode():
                    run_step(step_fn=step_fn, batch=batch, batch_idx=batch_idx, **test_ctx)
            else:
                run_step(step_fn=step_fn, batch=batch, batch_idx=batch_idx, **test_ctx)
        _call_itmodule_hook(module, hook_name="on_analysis_epoch_end", hook_msg="Running analysis epoch end hooks")
    analysis_caches = module.analysis_caches
    _call_itmodule_hook(module, hook_name="on_analysis_end", hook_msg="Running analysis end hooks")
    return analysis_caches

# TODO: currently, analysis_step will only be supported IT SessionRunner and not Lightning framework's Trainer
#       If sufficient interest is shown, we can consider a PR adding support for Lightning's Trainer

def run_step(step_fn, module, batch, batch_idx, optimizer: Optimizable | None = None):
    batch = module.batch_to_device(batch)
    step_func = getattr(module, step_fn)
    if module.global_step == 0 and step_fn in ("training_step", "test_step"):
        _call_itmodule_hook(module, hook_name="_on_test_or_train_batch_start",
                            hook_msg="Running custom test or train batch start hook", batch=batch, batch_idx=batch_idx)
    if step_fn == "training_step":
        optimizer.zero_grad()
    if module.torch_dtype == torch.bfloat16:
        with torch.autocast(device_type=module.device.type, dtype=module.torch_dtype):
            loss = step_func(batch, batch_idx)
    else:
        loss = step_func(batch, batch_idx)
    if step_fn == "training_step":
        _call_itmodule_hook(module, hook_name="on_train_batch_end",
                            hook_msg="Running custom on_train_batch end hook", outputs=loss,
                            batch=batch, batch_idx=batch_idx)
        loss.backward()
        optimizer.step()
    module.global_step += 1


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
        if not self.analysis_set_cfg:
            raise MisconfigurationException("AnalysisSetCfg must be provided to run analysis")
        if (hasattr(self.analysis_set_cfg, "limit_analysis_batches") and
            self.analysis_set_cfg.limit_analysis_batches is not None):
            self.limit_analysis_batches = self.analysis_set_cfg.limit_analysis_batches

class SessionRunner:
    """A barebones trainer that can be used to orchestrate training when no adapter is specified during ITSession
    composition."""
    def __init__(self, run_cfg: SessionRunnerCfg | dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        super().__init__(*args, **kwargs)
        self.run_cfg = run_cfg if isinstance(run_cfg, SessionRunnerCfg) else SessionRunnerCfg(**run_cfg)
        # Only training and testing commands are supported in SessionRunner
        self.supported_commands = (None, "train", "test")
        self.it_init()

    @property
    def phase(self) -> CorePhases | None:
        return self._current_phase

    @phase.setter
    def phase(self, phase: CorePhases):
        self._current_phase = phase

    def it_init(self):
        # Unless overridden we dispatch the trainer-independent `it_init`
        it_init(**self.run_cfg.it_session)

    def it_session_end(self):
        """Dispatch any phase-specific session end hooks."""
        it_session_end(session_type=self.phase, **self.run_cfg.it_session)

    def _run(self, phase, loop_fn, *args: Any, **kwargs: Any) -> Any | None:
        self.phase = CorePhases[phase]
        phase_artifacts = loop_fn(**self.run_cfg.__dict__)
        self.it_session_end()
        return phase_artifacts

    test = partialmethod(_run, phase="test", loop_fn=core_test_loop)
    train = partialmethod(_run, phase="train", loop_fn=core_train_loop)

# TODO: if only a single analysis mode and analysis cache is generated by a analysis set return the single
#       analysiscache/store via  analysis_set_results in analysismgr so it can be pipelined to other analysis ops
#       (Make configurable and document)'
# TODO: need to decide whether to build collection of  multiple analysis step artifacts into the module or keep
#       collecting each run artifact in an external dict as is currently done)
# TODO: decide how best to handle running a single analysis mode, still require AnalysisRunnerCfg?

class AnalysisRunner(SessionRunner):
    """Trainer subclass with analysis orchestration logic."""
    def __init__(self, run_cfg: AnalysisRunnerCfg | dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        super().__init__(run_cfg, *args, **kwargs)
        # Extend supported commands to include analysis
        self.supported_commands = (*self.supported_commands, "analysis")
        self.analysis_set_results = {}

    # TODO: revert to allowing for a single mode to be run at a time (rather than an internal self.analysis_set_results)
    #       if that pattern makes sense
    def run_analysis_mode(self, mode: AnalysisMode, cache: AnalysisCache, names_filter: NamesFilter, **kwargs):
        self.run_cfg.module.analysis_cfg = AnalysisCfg(analysis_cache=cache, mode=mode, names_filter=names_filter,
                                                       **kwargs)
        self.analysis_set_results[mode] = self.analysis()

    def run_analysis_set(self) -> dict[str, Any]:
        # for now we are requiring that the user provide an AnalysisSetCfg to run analysis
        # assert self.trainer_cfg.analysis_set_cfg, "AnalysisSetCfg must be provided to run analysis set"
        cfg = self.run_cfg.analysis_set_cfg.sae_analysis_targets
        names_filter = self.run_cfg.module.construct_names_filter(cfg.target_layers, cfg.sae_hook_match_fn)
        for mode, analysis_cache in self.run_cfg.analysis_set_cfg.analysis_caches.items():
            if mode == AnalysisMode.ablation:
                # Ensure that 'clean_w_sae' has already run (maybe upstream logic altered or previous runs were skipped)
                assert AnalysisMode.clean_w_sae in self.analysis_set_results, "clean_w_sae must be run before ablation"
                ref_results = self.analysis_set_results[AnalysisMode.clean_w_sae]
                ref_artifacts = {
                    "base_logit_diffs": ref_results.logit_diffs,
                    "answer_indices": ref_results.answer_indices,
                    "alive_latents": ref_results.alive_latents,
                }
                self.run_analysis_mode(mode, analysis_cache, names_filter, **ref_artifacts)
            else:
                self.run_analysis_mode(mode, analysis_cache, names_filter)
        # TODO: maybe return results in case we want to chain multiple analysis modes or user doesn't provide a dict
        return self.analysis_set_results

    def _run(self, phase, loop_fn, *args: Any, **kwargs: Any) -> Any | None:
        self.phase = AllPhases[phase]
        phase_artifacts = loop_fn(**self.run_cfg.__dict__)
        self.it_session_end()
        return phase_artifacts

    analysis = partialmethod(_run, phase="analysis", loop_fn=core_analysis_loop)
