from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
import warnings
from typing import Optional, Generator, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import os
import datetime

from datasets.config import HF_CACHE_HOME
from transformers import BatchEncoding, PreTrainedTokenizerBase

from interpretune.analysis import (SAEAnalysisTargets, resolve_names_filter, _make_simple_cache_hook, OpSchema,
                                   AnalysisOp)
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.config import ITSerializableCfg
from interpretune.protocol import NamesFilter, AnalysisStoreProtocol, AnalysisBatchProtocol, STEP_OUTPUT
from interpretune.utils import DEFAULT_DECODE_KWARGS
from interpretune.utils import rank_zero_warn

IT_ANALYSIS_CACHE_DIR = "interpretune"
DEFAULT_IT_ANALYSIS_CACHE = os.path.join(HF_CACHE_HOME, IT_ANALYSIS_CACHE_DIR)
IT_ANALYSIS_CACHE = Path(os.getenv("IT_ANALYSIS_CACHE_DIR", DEFAULT_IT_ANALYSIS_CACHE))

@dataclass(kw_only=True)
class AnalysisCfg(ITSerializableCfg):
    output_store: AnalysisStoreProtocol = None  # usually constructed on setup()
    input_store: AnalysisStoreProtocol | None = None  # store containing input data from previous op
    op: Optional[Union[str, AnalysisOp, Callable, list[AnalysisOp]]] = None  # op/op chain via generated analysis step
    output_schema: Optional[OpSchema | str | AnalysisOp] = None  # Schema, op, or op name to define schema
    name: Optional[str] = None  # Name for this analysis configuration
    fwd_hooks: list[tuple] = field(default_factory=list)
    bwd_hooks: list[tuple] = field(default_factory=list)
    cache_dict: dict = field(default_factory=dict)
    names_filter: NamesFilter | None = None
    save_prompts: bool = False
    save_tokens: bool = False
    decode_kwargs: dict = field(default_factory=lambda: DEFAULT_DECODE_KWARGS)
    sae_analysis_targets: SAEAnalysisTargets = None
    ignore_manual: bool = False  # When True, ignore existing analysis_step and use op to generate one
    step_fn: str = "analysis_step"  # Name of the method to use/generate for analysis

    def __post_init__(self):
        # Check if ignore_manual is True but no op is provided
        if self.ignore_manual and self.op is None:
            raise ValueError("When ignore_manual is True, an op must be provided to generate the analysis_step")

        resolved_op = None

        # Process output_schema first if it's an op or string
        if self.output_schema is not None and not isinstance(self.output_schema, OpSchema):
            # If output_schema is AnalysisOp-like (using this attribute as heuristic to limit expensive protocol
            # checks), extract its schema
            if hasattr(self.output_schema, 'output_schema'):
                resolved_op = self.output_schema
                self.output_schema = self.output_schema.output_schema
            # If output_schema is a string, resolve to op and extract schema
            elif isinstance(self.output_schema, str):
                resolved_op = DISPATCHER.get_op(self.output_schema) or DISPATCHER.get_by_alias(self.output_schema)
                if resolved_op is None:
                    raise ValueError(f"Unknown operation for schema reference: {self.output_schema}")
                self.output_schema = resolved_op.output_schema

        # Process the operation if provided
        if self.op is not None:
            # Convert list of ops to chain
            if isinstance(self.op, list):
                if len(self.op) == 1:
                    self.op = self.op[0]
                    resolved_op = self.op
                else:
                    # Create a composite operation from the list
                    self.op = DISPATCHER.create_chain_from_ops(self.op)
                    resolved_op = self.op
            # Handle chained operations using dot notation
            elif isinstance(self.op, str):
                if '.' in self.op:
                    # This is a chained operation
                    try:
                        self.op = DISPATCHER.create_chain(self.op)
                        resolved_op = self.op
                    except ValueError as e:
                        raise e
                else:
                    # Try to resolve a single op by name or alias
                    resolved_op = DISPATCHER.get_op(self.op) or DISPATCHER.get_by_alias(self.op)
                    if resolved_op is None:
                        raise ValueError(f"Unknown operation: {self.op}")
                    self.op = resolved_op
            else:
                # If op is already an AnalysisOp or similar, use it directly
                resolved_op = self.op

        # Set name from resolved op if name is not already set
        if self.name is None and resolved_op is not None and hasattr(resolved_op, 'alias'):
            self.name = resolved_op.alias

        # If name still not set (no op was resolved), use timestamp default
        if self.name is None:
            self.name = f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def update(self, **kwargs):
        """Update multiple fields of the dataclass at once."""
        for key, value in kwargs.items():
            if key in self.__annotations__:
                setattr(self, key, value)

    def materialize_names_filter(self, module, fallback_sae_targets: Optional[SAEAnalysisTargets] = None) -> None:
        """Set names_filter using sae_analysis_targets if not already set.

        Args:
            module: The module to construct the names_filter for.
            fallback_sae_targets: Optional fallback SAEAnalysisTargets to use if this config doesn't have one.
        """
        # Skip if names_filter is already set
        if self.names_filter is not None:
            self.names_filter = resolve_names_filter(self.names_filter)
            return

        # Choose the appropriate SAEAnalysisTargets
        sae_targets = self.sae_analysis_targets if self.sae_analysis_targets is not None else fallback_sae_targets

        if sae_targets is not None:
            target_layers = sae_targets.target_layers
            match_fn = sae_targets.sae_hook_match_fn
            self.names_filter = module.construct_names_filter(target_layers, match_fn)
        else:
            raise ValueError("No SAEAnalysisTargets available to create names_filter")
        self.names_filter = resolve_names_filter(self.names_filter)

    def maybe_set_hooks(self) -> None:
        """Set hooks if they're not already set."""
        if not self.fwd_hooks and not self.bwd_hooks:
            self.check_add_default_hooks()

    def prepare_model_ctx(self, module, fallback_sae_targets: Optional[SAEAnalysisTargets] = None) -> None:
        """Configure names_filter and hooks for a specific module.

        Args:
            module: The module to configure for.
            fallback_sae_targets: Optional fallback SAEAnalysisTargets to use if this config doesn't have one.
        """
        # Always materialize names_filter - needed for both op-based and manual analysis
        self.materialize_names_filter(module, fallback_sae_targets)

        # Set hooks based on op or manually if needed
        if self.op is not None:
            self.maybe_set_hooks()

    def reset_analysis_store(self) -> None:
        # TODO: May be able to refactor this out if using per-epoch op dataset subsplits
        # Prepare a new cache for the next epoch preserving save_cfg (for multi-epoch AnalysisSessionCfg instances)
        current_analysis_cls = self.output_store.__class__
        current_save_cfg_cls = self.output_store.save_cfg.__class__
        current_save_cfg = {k: v for k, v in self.output_store.save_cfg.__dict__.items() if k != 'output_store'}
        self.output_store = current_analysis_cls(save_cfg=current_save_cfg_cls(**current_save_cfg))
        assert id(self.output_store) == id(self.output_store.save_cfg.output_store)
        # TODO: maybe use this approach instead of above
        #"""Reset analysis store, preserving configuration."""
        #self.analysis_store.reset()


    def save_batch(self, analysis_batch: AnalysisBatchProtocol, batch: BatchEncoding,
                   tokenizer: PreTrainedTokenizerBase | None = None):
        """Process and yield analysis batch results.

        Uses AnalysisOp.process_batch for consistent processing regardless of
        whether an operation is defined or using a manual analysis step.

        Args:
            analysis_batch: The analysis batch to process
            batch: The raw batch data
            tokenizer: Optional tokenizer for decoding prompts

        Yields:
            Processed analysis batch
        """

        if self.op is not None:
            # When using a defined operation
            analysis_batch = self.op.save_batch(
                analysis_batch,
                batch,
                tokenizer=tokenizer,
                save_prompts=self.save_prompts,
                save_tokens=self.save_tokens,
                decode_kwargs=self.decode_kwargs
            )
        else:
            # For manual analysis steps, use the static method with output_schema
            analysis_batch = AnalysisOp.process_batch(
                analysis_batch,
                batch,
                output_schema=self.output_schema or OpSchema({}),
                tokenizer=tokenizer,
                save_prompts=self.save_prompts,
                save_tokens=self.save_tokens,
                decode_kwargs=self.decode_kwargs
            )

        yield analysis_batch

    def add_default_cache_hooks(
            self,
            include_backward: bool = True
            ) -> tuple[list[tuple], list[tuple]]:
        """Add default caching hooks for forward and optionally backward passes.

        Args:
            names_filter: Filter to determine which layers to hook
            cache_dict: Dictionary to store activation values
            include_backward: Whether to include backward hooks

        Returns:
            Tuple of (forward hooks, backward hooks)
        """
        fwd_hooks = [(self.names_filter, _make_simple_cache_hook(cache_dict=self.cache_dict))]
        self.fwd_hooks = fwd_hooks
        bwd_hooks = []

        if include_backward:
            bwd_hooks = [(self.names_filter, _make_simple_cache_hook(cache_dict=self.cache_dict, is_backward=True))]
            self.bwd_hooks = bwd_hooks

    def check_add_default_hooks(self) -> None:
        """Construct forward and backward hooks based on analysis operation."""
        fwd_hooks, bwd_hooks = [], []

        if self.op is None:
            self.fwd_hooks, self.bwd_hooks = fwd_hooks, bwd_hooks
            return

        if self.op.name == 'logit_diffs_base':
            return fwd_hooks, bwd_hooks

        if self.names_filter is None:
            raise ValueError("names_filter required for non-clean operations")

        if self.op.name == 'logit_diffs_attr_grad':
            self.add_default_cache_hooks()

    def apply(self, module, cache_dir: Optional[str] = None, op_output_dataset_path: Optional[str] = None,
              fallback_sae_targets: Optional[SAEAnalysisTargets] = None):
        """Set up analysis configuration and configure for the given module.

        This method handles both setting up the analysis store and configuring
        the names_filter and hooks for the module. It also injects an analysis_step
        method if one doesn't exist in the module.

        Args:
            module: The module to configure for.
            cache_dir: Optional cache directory.
            op_output_dataset_path: Optional output path.
            fallback_sae_targets: Optional fallback SAEAnalysisTargets to use if this config doesn't have one.
        """
        # Check if module has an analysis step method with the specified name
        has_custom_step = (hasattr(module, self.step_fn) and
                          not getattr(module, f'_generated_{self.step_fn}', False))

        # Only warn about custom step if we're not going to ignore it
        if has_custom_step and not self.ignore_manual:
            warnings.warn(
                f"Module {module.__class__.__name__} already has a {self.step_fn} method. "
                "The provided operation configuration will be used for hooks and filters, "
                f"but the execution flow will be determined by the existing {self.step_fn} method."
            )

        # Generate a new step if no custom step exists OR we're explicitly ignoring manual steps
        if (not has_custom_step or self.ignore_manual) and self.op is not None:
            # Inject a dynamic analysis_step method
            def generated_analysis_step(
                self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0
            ) -> Generator[STEP_OUTPUT, None, None]:
                """Dynamically generated analysis_step method."""
                analysis_batch = None

                # Handle composite ops or chains
                op = self.analysis_cfg.op
                if hasattr(op, 'chain') and op.chain:
                    # Execute the chain of operations
                    for chain_op in op.chain:
                        analysis_batch = chain_op(self, analysis_batch, batch, batch_idx)
                else:
                    # Handle single op case
                    analysis_batch = op(self, analysis_batch, batch, batch_idx)

                yield from self.analysis_cfg.save_batch(analysis_batch, batch, tokenizer=self.datamodule.tokenizer)

            # Add the method to the module with the specified name
            setattr(module, self.step_fn, generated_analysis_step.__get__(module))
            # Mark it as generated using the step_fn name
            setattr(module, f'_generated_{self.step_fn}', True)

        # Always set up the analysis store, even for manual analysis steps
        if not self.output_store:
            # Create the output store if needed
            from interpretune.analysis import AnalysisStore
            self.output_store = AnalysisStore(cache_dir=cache_dir, op_output_dataset_path=op_output_dataset_path)
        elif cache_dir or op_output_dataset_path:
            rank_zero_warn(
                f"The provided cache_dir={cache_dir} and op_output_dataset_path={op_output_dataset_path} "
                "will be ignored in favor of the existing AnalysisStore configuration "
                f"(cache_dir={self.output_store.cache_dir}, "
                f" op_output_dataset_path={self.output_store.op_output_dataset_path})"
            )

        # Always prepare the model context to ensure names_filter is materialized
        self.prepare_model_ctx(module, fallback_sae_targets)


@dataclass(kw_only=True)
class AnalysisArtifactCfg(ITSerializableCfg):
    """Configuration for analysis artifacts and visualizations."""
    latent_effects_graphs: bool = True
    latent_effects_graphs_per_batch: bool = False  # can be overwhelming with many batches
    latents_table_per_sae: bool = True
    top_k_latents_table: int = 2
    top_k_latent_dashboards: int = 1  # (don't set too high, num dashboards = top_k_latent_dashboards * num_hooks * 2)
    top_k_clean_logit_diffs: int = 10

    def __post_init__(self):
        if self.latent_effects_graphs_per_batch and not self.latent_effects_graphs:
            print("Note: Setting latent_effects_graphs to True since latent_effects_graphs_per_batch is True")
            self.latent_effects_graphs = True
