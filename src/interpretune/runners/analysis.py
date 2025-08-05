from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Any, TYPE_CHECKING, Union, List, Optional, Dict, Tuple
import logging
from functools import partialmethod
from pathlib import Path
from inspect import signature

from tqdm.auto import tqdm
from datasets import Dataset

from interpretune.analysis import schema_to_features
from interpretune.base import _call_itmodule_hook, ITDataModule
from interpretune.runners import SessionRunner, run_step
from interpretune.protocol import AllPhases, AnalysisStoreProtocol
from interpretune.config import AnalysisRunnerCfg, AnalysisCfg, init_analysis_cfgs
from interpretune.utils import rank_zero_warn
from interpretune.utils.exceptions import handle_exception_with_debug_dump


if TYPE_CHECKING:
    from interpretune.adapters import ITModule


log = logging.getLogger(__name__)


def analysis_store_generator(module: ITModule, datamodule: ITDataModule,
                             limit_analysis_batches: int = -1,
                             step_fn: str = "analysis_step", max_epochs: int = 1, *args, **kwargs):
    # TODO: should we create separate dataset phase subsplits (per epoch)?
    # TODO: allow for custom dataloader associations
    dataloader = datamodule.test_dataloader()
    test_ctx = {"module": module, "as_generator": True}
    for epoch_idx in range(max_epochs):
        module.current_epoch = epoch_idx
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            if batch_idx >= limit_analysis_batches >= 0:
                break
            # TODO: move previous inference_mode toggle to IT dispatch logic
            yield from run_step(step_fn=step_fn, batch=batch, batch_idx=batch_idx, **test_ctx)
        _call_itmodule_hook(module, hook_name="on_analysis_epoch_end", hook_msg="Running analysis epoch end hooks")

def maybe_init_analysis_cfg(module: "ITModule", analysis_cfg: Optional[AnalysisCfg] = None, **kwargs) -> dict:
    """Initialize analysis configuration if needed and return updated kwargs.

    Args:
        module: The module to set up the analysis configuration for
        analysis_cfg: Optional analysis configuration to use
        **kwargs: Additional keyword arguments

    Returns:
        Updated kwargs with init_analysis_cfg parameters removed
    """
    if analysis_cfg is not None:
        # Set module's analysis_cfg
        module.analysis_cfg = analysis_cfg

        # Extract and pop any AnalysisCfg.apply-specific kwargs
        init_analysis_cfg_params = signature(init_analysis_cfgs).parameters
        init_analysis_cfg_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys())
            if k in init_analysis_cfg_params and k != "analysis_cfgs"
        }

        # Only initialize if not already applied to this module
        if not module.analysis_cfg.applied_to(module):
            init_analysis_cfgs(module, [module.analysis_cfg], **init_analysis_cfg_kwargs)

    return kwargs

def dataset_features_and_format(module: "ITModule", kwargs: dict) -> Tuple[dict, dict, Any, dict]:
    """Generate dataset features and formatting parameters based on module configuration.

    Args:
        module: The module containing analysis configuration

    Returns:
        Tuple containing:
        - features: Dataset features derived from schema
        - it_format_kwargs: Kwargs for interpretune format
        - schema_source: Source of the schema definition
        - serializable_col_cfg: Column configs in serializable format
    """
    # Generate appropriate features based on module configuration and current op context
    # Handle different schema sources based on configuration
    if hasattr(module.analysis_cfg, 'op') and module.analysis_cfg.op is not None:
        features = schema_to_features(module=module, op=module.analysis_cfg.op)
        schema_source = module.analysis_cfg.op.output_schema
    elif hasattr(module.analysis_cfg, 'output_schema') and module.analysis_cfg.output_schema is not None:
        # Use explicitly provided output schema when op is None
        features = schema_to_features(module=module, schema=module.analysis_cfg.output_schema)
        schema_source = module.analysis_cfg.output_schema
    else:
        # For manual analysis steps without any schema, use a default empty features dict
        features = {}
        schema_source = {}

    # Convert ColCfg objects to dicts for JSON serialization
    serializable_col_cfg = {k: v.to_dict() for k, v in schema_source.items()} if schema_source else {}
    it_format_kwargs = dict(col_cfg=serializable_col_cfg)
    kwargs.update(schema_source=schema_source, serializable_col_cfg=serializable_col_cfg)
    return features, it_format_kwargs, kwargs

def generate_analysis_dataset(
    module, features, it_format_kwargs, gen_kwargs, split="test", **kwargs
) -> Dataset:
    """Generate a dataset for analysis using the ITAnalysisFormatter.

    Args:
        module: The module to analyze
        features: Features derived from schema
        it_format_kwargs: Kwargs for interpretune format
        gen_kwargs: Dictionary of generator parameters (module, datamodule, limit_analysis_batches, etc.)
        split: The split to use for dataset generation
        **kwargs: Additional arguments for error context

    Returns:
        Dataset: The generated dataset with interpretune format

    Raises:
        Exception: If dataset generation fails, with detailed debug information
    """
    # TODO: allow split customization rather than hardcode to "test"
    from_gen_kwargs = dict(generator=analysis_store_generator, gen_kwargs=gen_kwargs, features=features, split=split,
                           cache_dir=module.analysis_cfg.output_store.cache_dir)

    # Create dataset with ITAnalysisFormatter
    try:
        dataset = Dataset.from_generator(**from_gen_kwargs).with_format("interpretune", **it_format_kwargs)
        return dataset
    except Exception as e:
        # improve visibility of errors since they can otherwise be obscured by the dataset generator
        context_data = (
            features,                 # Features derived from schema
            from_gen_kwargs,          # Arguments for Dataset.from_generator
            gen_kwargs,               # Arguments for the analysis generator
            it_format_kwargs,         # Arguments for interpretune format
            kwargs,                   # Additional user-provided arguments
        )
        handle_exception_with_debug_dump(e, context_data, "dataset_generation")
        raise  # Re-raise after logging to ensure proper error handling

def core_analysis_loop(module: ITModule, datamodule: ITDataModule,
                      limit_analysis_batches: int = -1,
                      step_fn: str = "analysis_step",
                      max_epochs: int = 1, analysis_cfg: Optional[AnalysisCfg] = None, *args, **kwargs):
    """Create dataset using the ITAnalysisFormatter for optimal handling of analysis data."""
    # Initialize analysis configuration if needed
    kwargs = maybe_init_analysis_cfg(module, analysis_cfg, **kwargs)
    if analysis_cfg is not None:
        step_fn = analysis_cfg.step_fn

    # Run analysis start hooks
    # TODO: execute on_analysis_start/end hooks toggle grad based on functional config rather than op based filter
    _call_itmodule_hook(module, hook_name="on_analysis_start", hook_msg="Running analysis start hooks")

    # Generate features and format parameters
    features, it_format_kwargs, kwargs = dataset_features_and_format(module, kwargs)

    # Create generator kwargs dictionary
    gen_kwargs = dict(module=module, datamodule=datamodule, limit_analysis_batches=limit_analysis_batches,
                      step_fn=step_fn, max_epochs=max_epochs)

    # Generate the dataset
    dataset = generate_analysis_dataset(module, features, it_format_kwargs, gen_kwargs, **kwargs)


    save_dir = Path(module.analysis_cfg.output_store.save_dir)
    dataset.save_to_disk(save_dir)
    # Assign dataset to analysis store
    module.analysis_cfg.output_store.dataset = dataset

    # Run analysis end hooks
    _call_itmodule_hook(module, hook_name="on_analysis_end", hook_msg="Running analysis end hooks")

    return module.analysis_cfg.output_store

class AnalysisRunner(SessionRunner):
    """Trainer subclass with analysis orchestration logic."""
    def __init__(self, run_cfg: AnalysisRunnerCfg | dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        run_cfg = run_cfg if isinstance(run_cfg, AnalysisRunnerCfg) else AnalysisRunnerCfg(**run_cfg)
        super().__init__(run_cfg, *args, **kwargs)
        # Extend supported commands to include analysis
        self.supported_commands = (*self.supported_commands, "analysis")
        self.analysis_results = {}

    def it_init(self):
        super().it_init()
        module = self.run_cfg.module
        # Check if running analysis and the module needs an analysis_step
        if hasattr(self, 'phase') and self.phase == 'analysis':
            if not hasattr(module, 'analysis_step') or getattr(module, '_generated_analysis_step', False):
                if hasattr(module, 'analysis_cfg') and hasattr(module.analysis_cfg, 'op'):
                    # Apply the analysis config to generate the analysis_step
                    module.analysis_cfg.apply(module)
                else:
                    rank_zero_warn(
                        f"Module {module.__class__.__name__} has no analysis_step method and "
                        "no analysis configuration to generate one."
                    )


    def _run(self, phase, loop_fn, step_fn: Optional[str] = None, *args: Any, **kwargs: Any) -> Any | None:
        self.phase = AllPhases[phase]
        phase_artifacts = loop_fn(step_fn=step_fn, **self.run_cfg.__dict__)
        self.it_session_end()
        return phase_artifacts

    analysis = partialmethod(_run, phase="analysis", loop_fn=core_analysis_loop)

    def run_analysis(self,
                     analysis_cfgs: Optional[Union[AnalysisCfg, Any, List[Union[AnalysisCfg, Any]]]] = None,
                     cache_dir: str | Path | None = None,
                     op_output_dataset_path: str | Path | None = None,
                     **kwargs) -> AnalysisStoreProtocol | Dict[str, AnalysisStoreProtocol]:
        """Unified method to run analysis operations based on the provided configuration.

        Args:
            analysis_cfgs: Configuration for analysis operations. Can be:
                - An AnalysisCfg instance
                - An AnalysisOp instance
                - A list of AnalysisCfg or AnalysisOp instances
            cache_dir: Optional override for the cache directory
            op_output_dataset_path: Optional override for the output dataset path
            **kwargs: Additional arguments to pass to the analysis function

        Returns:
            For a single analysis configuration: The result of that analysis
            For multiple analysis configurations: A dictionary mapping configuration names to results
        """
        # Reset analysis results for this run
        self.analysis_results = {}

        # Update configuration parameters if provided
        if cache_dir is not None:
            self.run_cfg.cache_dir = cache_dir
        if op_output_dataset_path is not None:
            self.run_cfg.op_output_dataset_path = op_output_dataset_path

        # Update analysis_cfgs if provided
        if analysis_cfgs is not None:
            self.run_cfg.analysis_cfgs = analysis_cfgs
            # # Process the new analysis_cfgs
            # self.run_cfg._process_analysis_cfgs()

        # Initialize analysis configurations
        init_analysis_cfgs(
            module=self.run_cfg.module,
            analysis_cfgs=self.run_cfg._processed_analysis_cfgs,
            cache_dir=self.run_cfg.cache_dir,
            op_output_dataset_path=self.run_cfg.op_output_dataset_path,
            sae_analysis_targets=self.run_cfg.sae_analysis_targets,
            ignore_manual=self.run_cfg.ignore_manual
        )

        # Check if we have any analysis configurations to process
        if not self.run_cfg._processed_analysis_cfgs:
            raise ValueError("No analysis configurations provided. Please specify analysis_cfgs.")

        # Run each analysis configuration
        for cfg in self.run_cfg._processed_analysis_cfgs:
            result = self._run_analysis_cfg(cfg)
            self.analysis_results[cfg.name] = result

        # Return a single result for a single config, dictionary for multiple
        if len(self.analysis_results) == 1:
            return next(iter(self.analysis_results.values()))
        else:
            return self.analysis_results

    def _run_analysis_cfg(self, analysis_cfg: AnalysisCfg) -> Any:
        """Run a single analysis operation with the provided configuration."""
        # Set active analysis config
        self.run_cfg.module.analysis_cfg = analysis_cfg

        # TODO: Determine if/when we should (re)apply an analysis_cfg given `init_analysis_cfgs` should have already
        #       applied the config. Would probably only if one analysis_cfg updates global state in a way that requires
        #       subsequent ops to be aware of it (a pattern that we are not sure we want to encourage)
        # analysis_cfg.apply(self.run_cfg.module)

        # Run the analysis with the specified step_fn
        result = self.analysis(step_fn=analysis_cfg.step_fn, **self.run_cfg.__dict__)

        return result
