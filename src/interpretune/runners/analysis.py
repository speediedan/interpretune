from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Any, TYPE_CHECKING
import logging
from functools import partialmethod

import torch
from tqdm.auto import tqdm
from datasets import Dataset

import interpretune as it
from interpretune.analysis import schema_to_features
from interpretune.base import _call_itmodule_hook, ITDataModule
from interpretune.runners import SessionRunner, run_step
from interpretune.utils import AllPhases
from interpretune.config import AnalysisRunnerCfg


if TYPE_CHECKING:
    from interpretune.adapters import ITModule

log = logging.getLogger(__name__)

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
