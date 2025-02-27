from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Any, Optional, Sequence, Dict
import logging
from functools import partialmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm.auto import tqdm
from datasets import Dataset

import interpretune as it
from interpretune.adapters.core import ITModule
from interpretune.analysis.core import AnalysisStore, SAEAnalysisTargets, schema_to_features
from interpretune.base.call import _call_itmodule_hook, it_init, it_session_end
from interpretune.base.config.shared import AllPhases, CorePhases, ITSerializableCfg
from interpretune.base.config.analysis import AnalysisCfg
from interpretune.analysis.protocol import SAEAnalysisProtocol
from interpretune.base.contract.protocol import ITModuleProtocol, ITDataModuleProtocol
from interpretune.base.contract.session import ITSession
from interpretune.base.datamodules import ITDataModule, IT_ANALYSIS_CACHE
from interpretune.analysis.ops import AnalysisOp
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

def analysis_store_generator(module: ITModule, datamodule: ITDataModule,
                             limit_analysis_batches: int = -1,
                                step_fn: str = "analysis_step", max_epochs: int = 1, *args, **kwargs):
    # TODO: should we create separate dataset phase subsplits (per epoch)?
    dataloader = datamodule.test_dataloader()
    test_ctx = {"module": module, "as_generator": True}
    for epoch_idx in range(max_epochs):
        module.current_epoch = epoch_idx
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            if batch_idx >= limit_analysis_batches >= 0:
                break
            if module.analysis_cfg.op != it.logit_diffs_attr_grad:
                with torch.inference_mode():
                    yield from run_step(step_fn=step_fn, batch=batch, batch_idx=batch_idx, **test_ctx)
            else:
                yield from run_step(step_fn=step_fn, batch=batch, batch_idx=batch_idx, **test_ctx)
        _call_itmodule_hook(module, hook_name="on_analysis_epoch_end", hook_msg="Running analysis epoch end hooks")

def core_analysis_loop(module: ITModule, datamodule: ITDataModule,
                      limit_analysis_batches: int = -1,
                      step_fn: str = "analysis_step",
                      max_epochs: int = 1, *args, **kwargs):
    """Create dataset using the ITAnalysisFormatter for optimal handling of analysis data."""

    # Run analysis start hooks
    _call_itmodule_hook(module, hook_name="on_analysis_start", hook_msg="Running analysis start hooks")
    # TODO: probably better to make this a method on the AnalysisStepMixin
    # TODO: allow for custom dataloader associations
    # Generate appropriate features based on module configuration and current op context
    features = schema_to_features(module=module, op=module.analysis_cfg.op)
    gen_kwargs = dict(module=module, datamodule=datamodule, limit_analysis_batches=limit_analysis_batches,
                      step_fn=step_fn, max_epochs=max_epochs)

    # Convert ColCfg objects to dicts for JSON serialization
    serializable_col_cfg = {k: v.to_dict() for k, v in module.analysis_cfg.op.output_schema.items()}
    it_format_kwargs = dict(col_cfg=serializable_col_cfg)
    from_gen_kwargs = dict(generator=analysis_store_generator, gen_kwargs=gen_kwargs, features=features, split="test",
                           cache_dir=module.analysis_cfg.output_store.cache_dir)

    # Create dataset with ITAnalysisFormatter
    dataset = Dataset.from_generator(**from_gen_kwargs).with_format("interpretune", **it_format_kwargs)
    dataset.save_to_disk(str(module.analysis_cfg.output_store.save_dir))
    # Assign dataset to analysis store
    module.analysis_cfg.output_store.dataset = dataset

    # Run analysis end hooks
    _call_itmodule_hook(module, hook_name="on_analysis_end", hook_msg="Running analysis end hooks")

    return module.analysis_cfg.output_store

# TODO: currently, analysis_step will only be supported IT SessionRunner and not Lightning framework's Trainer
#       If sufficient interest is shown, we can consider a PR adding support for Lightning's Trainer

def run_step(step_fn, module, batch, batch_idx, optimizer: Optimizable | None = None, as_generator: bool = False):
    batch = module.batch_to_device(batch)
    step_func = getattr(module, step_fn)
    if module.global_step == 0 and step_fn in ("training_step", "test_step"):
        _call_itmodule_hook(
            module,
            hook_name="_on_test_or_train_batch_start",
            hook_msg="Running custom test or train batch start hook",
            batch=batch,
            batch_idx=batch_idx,
        )
    if step_fn == "training_step" and optimizer is not None:
        optimizer.zero_grad()
    if module.torch_dtype == torch.bfloat16:
        with torch.autocast(device_type=module.device.type, dtype=module.torch_dtype):
            output = step_func(batch, batch_idx)
    else:
        output = step_func(batch, batch_idx)
    if step_fn == "training_step" and optimizer is not None:
        _call_itmodule_hook(module, hook_name="on_train_batch_end", hook_msg="Running custom on_train_batch end hook",
                            outputs=output, batch=batch, batch_idx=batch_idx)
        output.backward()
        optimizer.step()
    module.global_step += 1

    if as_generator:
        yield from output
    else:
        return output


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
class AnalysisSetCfg(ITSerializableCfg):
    # Accept any sequence as input and convert to tuple in __post_init__
    # TODO: ultimately we want to support an arbitrary DAG of ops
    analysis_ops: Sequence[AnalysisOp] | None = None
    # currently allow executing a sequence of Ops (generating cfgs) or a sequence of AnalysisCfg objects if passed
    analysis_cfgs: Dict[AnalysisOp, AnalysisCfg] = field(default_factory=dict)
    # TODO: next two attributes available in AnalysisCfg as well, prob should decouple from AnalysisSetCfg
    cache_dir: Optional[str] = None
    op_output_dataset_path: Optional[str] = None
    # we allow for limiting the number of analysis batches both here and in the runner for convenience
    limit_analysis_batches: int = -1
    sae_analysis_targets: SAEAnalysisTargets = field(default_factory=SAEAnalysisTargets)
    latent_effects_graphs: bool = True
    latent_effects_graphs_per_batch: bool = False  # can be overwhelming with many batches
    latents_table_per_sae: bool = True
    top_k_latents_table: int = 2
    top_k_latent_dashboards: int = 1  # (don't set too high, num dashboards = top_k_latent_dashboards * num_hooks * 2)
    top_k_clean_logit_diffs: int = 10

    def __post_init__(self):
        if not self.analysis_ops and not self.analysis_cfgs:
            raise MisconfigurationException("Either analysis_ops or analysis_cfgs must be provided")
        # Ensure analysis_ops is stored as a tuple if provided
        if self.analysis_ops is not None and not isinstance(self.analysis_ops, tuple):
            self.analysis_ops = tuple(self.analysis_ops)
        if self.latent_effects_graphs_per_batch and not self.latent_effects_graphs:
            print("Note: Setting latent_effects_graphs to True since latent_effects_graphs_per_batch is True")
            self.latent_effects_graphs = True
        if self.analysis_ops:
            self.validate_analysis_order()

    def init_analysis_dirs(self, module: SAEAnalysisProtocol):
        # TODO: after debugging update the default path to be more configurable instead of default to validation
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
        for op in self.analysis_ops:
            op_dir = self.op_output_dataset_path / op.name
            if op_dir.exists() and any(op_dir.iterdir()):
                raise Exception(
                    f"Analysis dataset directory for op '{op.name}' ({op_dir}) is not empty. "
                    "Please delete it or specify a different path."
                )

    # TODO: revisit appropriate type approach/hints here
    def init_analysis_cfgs(self, module: SAEAnalysisProtocol):
        target_layers, match_fn = self.sae_analysis_targets.target_layers, self.sae_analysis_targets.sae_hook_match_fn
        names_filter = module.construct_names_filter(target_layers, match_fn)
        clean_w_sae_enabled = it.logit_diffs_sae in self.analysis_ops
        # TODO: decouple prompts and tokens logic from AnalysisSetCfg, unnest SaveCfg to AnalysisStore (configurable
        #       from AnalysisCfg)
        # TODO: We currently generate AnalysisCfg objects for each analysis op in the set but we want to allow
        #       for the user to explicitly provide a sequence of AnalysisCfg objects instead (should be mutually
        #       exclusive with specifying ops)
        prompts_tokens_cfg = dict(save_prompts=not clean_w_sae_enabled, save_tokens=not clean_w_sae_enabled)
        # TODO: make this a DatasetDict for each epoch for each analysis op? Should just need to flush cache per epoch
        #       now.
        self.init_analysis_dirs(module)
        # TODO: Decouple analysis_ops from AnalysisSetCfg, instead create an AnalysisDispatcher
        # TODO: For now, allowing both an analysis_ops path and a separate analysis_cfg path, but should probably unify
        if self.analysis_cfgs:
            for op, cfg in self.analysis_cfgs.items():
                # TODO: disentangle AnalysisCfg path from AnalysisSetCfg so AnalysisSetCfg is optional
                cfg.setup()
                cfg.names_filter = names_filter
        else:
            for op in self.analysis_ops:
                # Use shared cache_dir for all ops; create op-specific output directory only
                op_output_path = self.op_output_dataset_path / op.name
                op_output_path.mkdir(exist_ok=True, parents=True)
                analysis_dirs = {"cache_dir": self.cache_dir, "op_output_dataset_path": op_output_path}
                # Create analysis store
                analysis_store = AnalysisStore(**analysis_dirs)
                # Configure AnalysisCfg for the user based on the provided op with default settings
                self.analysis_cfgs[op] = AnalysisCfg(
                    output_store=analysis_store,
                    op=op,
                    names_filter=names_filter,
                    save_prompts=True if op == it.logit_diffs_sae else prompts_tokens_cfg["save_prompts"],
                    save_tokens=True if op == it.logit_diffs_sae else prompts_tokens_cfg["save_tokens"],
                )

    def validate_analysis_order(self):
        """Validates and potentially reorders analysis operations to ensure dependencies are met.

        Currently ensures that logit_diffs.sae comes before ablation if ablation is enabled.
        """
        # TODO: currently AnalysisSetCfg supports a simple sequence of AnalysisOps, we ultimately want to support an
        #       arbitrary DAG of ops.
        # Ensure that clean_w_sae comes before ablation if ablation is enabled
        # TODO: update for analysis_cfg path
        # Check if ablation analysis is requested
        if it.logit_diffs_attr_ablation in self.analysis_ops:
            # Ensure logit_diffs.sae is included and comes before ablation
            if it.logit_diffs_sae not in self.analysis_ops:
                print("Note: Adding logit_diffs.sae op since it is required for ablation")
                self.analysis_ops = tuple([it.logit_diffs_sae] + list(self.analysis_ops))
            # Sort ops to ensure logit_diffs.sae comes before ablation
            sorted_ops = sorted(self.analysis_ops,
                            key=lambda x: (x != it.logit_diffs_sae,
                                        x != it.logit_diffs_attr_ablation))
            if sorted_ops != list(self.analysis_ops):
                print("Note: Re-ordering analysis ops to ensure logit_diffs.sae runs before ablation")
                self.analysis_ops = tuple(sorted_ops)

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

# TODO: if only a single analysis op and analysis cache is generated by a analysis set return the single
#       AnalysisStore/store via  analysis_set_results in analysismgr so it can be pipelined to other analysis ops
#       (Make configurable and document)'
# TODO: need to decide whether to build collection of  multiple analysis step artifacts into the module or keep
#       collecting each run artifact in an external dict as is currently done)
# TODO: decide how best to handle running a single analysis op, still require AnalysisRunnerCfg?

class AnalysisRunner(SessionRunner):
    """Trainer subclass with analysis orchestration logic."""
    def __init__(self, run_cfg: AnalysisRunnerCfg | dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        super().__init__(run_cfg, *args, **kwargs)
        # Extend supported commands to include analysis
        self.supported_commands = (*self.supported_commands, "analysis")
        self.analysis_set_results = {}

    # TODO: Likely re-enable running a single mode/op to be run at a time rather than requiring AnalysisSetCfg and an
    #  internal self.analysis_set_results if that pattern makes sense
    # def run_analysis_mode(self, mode: AnalysisMode, analysis_store: AnalysisStore, names_filter: NamesFilter,
    # **kwargs):
    #     self.run_cfg.module.analysis_cfg = AnalysisCfg(analysis_store=analysis_store, mode=mode,
    # names_filter=names_filter, **kwargs)
    #     self.analysis_set_results[mode] = self.analysis()

    def run_analysis_set(self) -> dict[str, Any]:
        """Run sequence of analysis operations, handling dependencies between ops."""
        for op, analysis_cfg in self.run_cfg.analysis_set_cfg.analysis_cfgs.items():
            if op == it.logit_diffs_attr_ablation:
                # Ensure that 'logit_diffs.sae' has already run and produced results
                assert it.logit_diffs_sae in self.analysis_set_results, \
                    "logit_diffs.sae must be run before ablation"
                # Set the input store to the results from logit_diffs.sae op
                analysis_cfg.input_store = self.analysis_set_results[it.logit_diffs_sae]

                # Validate required fields are present in input store
                required_fields = ['logit_diffs', 'answer_indices', 'alive_latents']
                for attr in required_fields:
                    assert hasattr(analysis_cfg.input_store, attr), f"Input store missing required field: {attr}"

                # Validate input data format - check batch size matches dataloader config
                expected_batch_size = self.run_cfg.module.datamodule.itdm_cfg.eval_batch_size
                assert (isinstance(analysis_cfg.input_store.logit_diffs[0], torch.Tensor) and
                       analysis_cfg.input_store.logit_diffs[0].size(0) == expected_batch_size), \
                    f"Expected first batch of logit_diffs to match dataloader batch size ({expected_batch_size})"
                assert all([not isinstance(v, torch.Tensor)
                           for hook_dict in analysis_cfg.input_store.alive_latents
                           for v in hook_dict.values()])
            self.run_cfg.module.analysis_cfg = analysis_cfg
            self.analysis_set_results[op] = self.analysis()  # TODO: make this a DatasetDict instead?
        # TODO: maybe return results in case we want to chain multiple analysis modes or user doesn't provide a dict
        return self.analysis_set_results

    def _run(self, phase, loop_fn, *args: Any, **kwargs: Any) -> Any | None:
        self.phase = AllPhases[phase]
        phase_artifacts = loop_fn(**self.run_cfg.__dict__)
        self.it_session_end()
        return phase_artifacts

    analysis = partialmethod(_run, phase="analysis", loop_fn=core_analysis_loop)
