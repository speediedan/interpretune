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
from interpretune.utils import rank_zero_warn, rank_zero_debug

IT_ANALYSIS_CACHE_DIR = "interpretune"
DEFAULT_IT_ANALYSIS_CACHE = os.path.join(HF_CACHE_HOME, IT_ANALYSIS_CACHE_DIR)
IT_ANALYSIS_CACHE = Path(os.getenv("IT_ANALYSIS_CACHE_DIR", DEFAULT_IT_ANALYSIS_CACHE))

@dataclass(kw_only=True)
class AnalysisCfg(ITSerializableCfg):
    output_store: AnalysisStoreProtocol = None  # usually constructed on setup()
    input_store: AnalysisStoreProtocol | None = None  # store containing input data from previous op
    target_op: Optional[Union[str, AnalysisOp, Callable, list[AnalysisOp]]] = None  # input op to be resolved
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
    auto_prune_batch_encoding: bool = True  # Automatically prune encoded batches to only include relevant keys
    _applied_to: dict = field(default_factory=dict)  # Dictionary tracking which modules this cfg has been applied to
    _op: Optional[Union[str, AnalysisOp, Callable, list[AnalysisOp]]] = None  # op via generated analysis step

    @property
    def op(self) -> Optional[Union[str, AnalysisOp, Callable, list[AnalysisOp]]]:
        """Get the operation, unwrapping any OpWrapper if present."""
        if self._op is None:
            return None

        # Unwrap OpWrapper instances if needed
        if hasattr(self._op, '_is_instantiated') and getattr(self._op, '_is_instantiated', False):
            return self._op._instantiated_op

        return self._op

    @op.setter
    def op(self, value: Optional[Union[str, AnalysisOp, Callable, list[AnalysisOp]]]) -> None:
        """Set the operation value."""
        self._op = value

    def __post_init__(self):
        # Process the target_op if provided and set it as the op
        if self.target_op is not None:
            # TODO: consider saving the original target_op value before assigning to self.op (str, op, wrapper, etc.)
            # and then after the op is resolved by the end of the post_init, replace self.target_op with a
            # well-formatted string representation of the original target_op value and what it was resolved to
            # (e.g. "target_op: 'some_op_str' -> op: {str(self.op)}")
            self.op = self.target_op

        if self.ignore_manual and self.op is None:  # Check if ignore_manual is True but no op is provided
            raise ValueError("When ignore_manual is True, an op must be provided to generate the analysis_step")
        if self.output_schema is not None and not isinstance(self.output_schema, OpSchema):
            self.resolve_output_schema()  # Resolve the output schema if it's not already an OpSchema
        if self.op is not None:  # Process the operation if provided
            self.resolve_op()
        if self.name is None and self.op is not None:  # Set name from op if name is not already set
            self.name = self.op.name
        if self.name is None:  # If name still not set (no op was resolved), use timestamp default
            self.name = f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def update(self, **kwargs):
        """Update multiple fields of the dataclass at once."""
        for key, value in kwargs.items():
            if key in self.__annotations__:
                setattr(self, key, value)

    def resolve_output_schema(self) -> None:
        # If output_schema is AnalysisOp-like (using this attribute as heuristic to limit expensive protocol
        # checks), extract its schema
        if hasattr(self.output_schema, 'output_schema'):
            resolved_op = self.output_schema
            self.output_schema = self.output_schema.output_schema
        # If output_schema is a string, resolve to op and extract schema
        elif isinstance(self.output_schema, str):
            resolved_op = DISPATCHER.get_op(self.output_schema)
            self.output_schema = resolved_op.output_schema

    def resolve_op(self) -> None:
        if isinstance(self.op, list):  # Convert list of ops to composition
            if len(self.op) == 1:
                self.op = self.op[0]
                if isinstance(self.op, str):
                    self.op = DISPATCHER.get_op(self.op)
            else:
                # Create a composite operation from the list, ensuring each op in the list is instantiated
                # TODO: consider deferring instantiation of composite ops
                instantiated_ops = []
                for op in self.op:
                    if hasattr(op, '_ensure_instantiated'):
                        instantiated_ops.append(op._ensure_instantiated())
                    else:
                        instantiated_ops.append(op)
                self.op = DISPATCHER.compile_ops(instantiated_ops)
        elif isinstance(self.op, str):  # Handle composite operations using dot notation
            if '.' in self.op:
                # This is a composite operation
                try:
                    self.op = DISPATCHER.compile_ops(self.op)
                except ValueError as e:
                    raise e
            else:
                # Try to resolve a single op by name or alias
                self.op = DISPATCHER.get_op(self.op)

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
        sae_targets = self.sae_analysis_targets or fallback_sae_targets

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

        # Ensure sae_analysis_targets, fallback_sae_targets, or names_filter are set before materializing names_filter
        if not (self.sae_analysis_targets or fallback_sae_targets or self.names_filter):
            rank_zero_debug("None of (sae_analysis_targets, fallback_sae_targets, names_filter) are set. "
                            "Proceeding without materializing names_filter.")
        else:
            # Always materialize names_filter - needed for both op-based and manual analysis
            self.materialize_names_filter(module, fallback_sae_targets)

        # Set hooks based on op or manually if needed
        if self.op is not None:
            self.maybe_set_hooks()


    # TODO: Add a non-generator returning save_batch method at AnalysisCfg level (users already have op-level version)?
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

        # TODO: change these op-based checks to be functionally driven (e.g. uses_default_hooks attribute of ops)
        if self.op.name == 'logit_diffs_base':
            return fwd_hooks, bwd_hooks

        if self.op.name == 'logit_diffs_attr_grad':
            self.add_default_cache_hooks()

        # TODO: add in an op attribute akin to "uses_sae_hooks" to enable names_filter validation resolution etc.
        # if self.op_name in ('logit_diffs_attr_ablation', 'logit_diffs_attr_grad') and self.names_filter is None:
        #     raise ValueError("names_filter required for non-clean operations")



    def applied_to(self, module) -> bool:
        """Check if this configuration has been applied to a specific module.

        Args:
            module: The module to check.

        Returns:
            bool: True if this configuration has been applied to the module, False otherwise.
        """
        # Use module id as key to track unique module instances
        return id(module) in self._applied_to

    def reset_applied_state(self, module=None) -> None:
        """Reset the applied state tracking.

        Args:
            module: Optional specific module to reset. If None, reset for all modules.
        """
        if module is not None:
            # Reset for specific module
            self._applied_to.pop(id(module), None)
        else:
            # Reset for all modules
            self._applied_to.clear()

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
        # Short-circuit if already applied to this module (though apply should be idempotent)
        module_id = id(module)
        if module_id in self._applied_to:
            return

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

                # TODO: move this code to a separate function to allow for reuse outside of the generated method
                # Handle composite ops or compositions
                op = self.analysis_cfg.op
                analysis_batch = op(self, analysis_batch, batch, batch_idx)

                yield from self.analysis_cfg.save_batch(analysis_batch, batch, tokenizer=self.datamodule.tokenizer)

            # TODO: separate some of this more ephemeral state to an AnalysisState object
            # Add the method to the module with a _generated version of the specified step_fn name
            # (to avoid potentially clobbering the manual version)
            setattr(module, f'_generated_{self.step_fn}', generated_analysis_step.__get__(module))
            # update the analysis_cfg step_fn method name to the generated step_fn name
            setattr(self, 'step_fn', f'_generated_{self.step_fn}')

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

        # Mark as applied to this specific module, storing module class name for debugging
        self._applied_to[module_id] = module.__class__.__name__


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
