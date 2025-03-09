"""Definitions of specific analysis operations."""
from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Callable, Literal, Optional, Any
from collections import defaultdict
from functools import partial

import torch
from transformers import BatchEncoding
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float

from interpretune.protocol import AnalysisBatchProtocol
from interpretune.analysis.ops.base import OpSchema, AnalysisBatch, AnalysisOp


def boolean_logits_to_avg_logit_diff(
    logits: Float[torch.Tensor, "batch seq 2"],
    target_indices: torch.Tensor,
    reduction: Literal["mean", "sum"] | None = None,
    keep_as_tensor: bool = True,
) -> list[float] | float:
    """Returns the avg logit diff on a set of prompts, with fixed s2 pos and stuff."""
    incorrect_indices = 1 - target_indices
    correct_logits = torch.gather(logits, 2, torch.reshape(target_indices, (-1,1,1))).squeeze()
    incorrect_logits = torch.gather(logits, 2, torch.reshape(incorrect_indices, (-1,1,1))).squeeze()
    logit_diff = correct_logits - incorrect_logits
    if reduction is not None:
        logit_diff = logit_diff.mean() if reduction == "mean" else logit_diff.sum()
    return logit_diff if keep_as_tensor else logit_diff.tolist()

def ablate_sae_latent(
    sae_acts: torch.Tensor,
    hook: HookPoint,  # required by transformer_lens.hook_points._HookFunctionProtocol
    latent_idx: int | None = None,
    seq_pos: torch.Tensor | None = None,  # batched
) -> torch.Tensor:
    """Ablate a particular latent at a particular sequence position.

    If either argument is None, we ablate at all latents / sequence positions.
    """
    sae_acts[torch.arange(sae_acts.size(0)), seq_pos, latent_idx] = 0.0
    return sae_acts

class LabelsToIDsOp(AnalysisOp):
    """Analysis operation that converts string labels to tensor IDs."""

    def __init__(self, name: str = 'labels_to_ids',
                 description: str = 'Convert label strings to tensor IDs',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None) -> None:
        super().__init__(name, description, output_schema, input_schema)

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Convert labels to IDs."""
        if "labels" in batch:
            label_ids, orig_labels = module.labels_to_ids(batch.pop("labels"))
            if analysis_batch is None:
                analysis_batch = AnalysisBatch(labels=label_ids, orig_labels=orig_labels)
            else:
                analysis_batch.labels = label_ids
                analysis_batch.orig_labels = orig_labels
        return analysis_batch

class GetAnswerIndicesOp(AnalysisOp):
    """Analysis operation that extracts answer indices from batch."""

    def __init__(self, name: str = 'get_answer_indices',
                 description: str = 'Extract answer indices from batch',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None) -> None:
        super().__init__(name, description, output_schema, input_schema)

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Extract answer indices from batch."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Check if answer_indices already exist
        if hasattr(analysis_batch, 'answer_indices') and analysis_batch.answer_indices is not None:
            return analysis_batch

        # Check if we can get from input store
        if module.analysis_cfg.input_store and hasattr(module.analysis_cfg.input_store, 'answer_indices'):
            answer_indices = module.analysis_cfg.input_store.answer_indices[batch_idx]
        else:
            # Otherwise compute it
            tokens = batch["input"].detach().cpu()
            if module.datamodule.tokenizer.padding_side == "left":
                answer_indices = torch.full((tokens.size(0),), -1)
            else:
                nonpadding_mask = tokens != module.datamodule.tokenizer.pad_token_id
                # This could be more robust, test with various datasets and padding strategies
                answer_indices = torch.where(nonpadding_mask, 1, 0).sum(dim=1) - 1

        analysis_batch.update(answer_indices=answer_indices)
        return analysis_batch

class GetAliveLatentsOp(AnalysisOp):
    """Analysis operation that extracts alive latents from cache."""

    def __init__(self, name: str = 'get_alive_latents',
                 description: str = 'Extract alive latents from cache',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None) -> None:
        # Removed reference to ALIVE_LATENTS_SCHEMA
        super().__init__(name, description, output_schema, input_schema)

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Extract alive latents from cache."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Check if alive_latents already exist
        if hasattr(analysis_batch, 'alive_latents') and analysis_batch.alive_latents is not None:
            return analysis_batch

        # Check if we can get from input store
        if module.analysis_cfg.input_store and hasattr(module.analysis_cfg.input_store, 'alive_latents'):
            alive_latents = module.analysis_cfg.input_store.alive_latents[batch_idx]
        elif not hasattr(analysis_batch, 'cache') or analysis_batch.cache is None:
            alive_latents = {}
        else:
            # Extract alive latents from the cache using the answer indices
            cache = analysis_batch.cache
            names_filter = module.analysis_cfg.names_filter
            answer_indices = analysis_batch.answer_indices

            filtered_acts = {name: acts for name, acts in cache.items() if names_filter(name)}
            alive_latents = {}
            for name, acts in filtered_acts.items():
                alive = (acts[torch.arange(acts.size(0)),
                              answer_indices, :] > 0).any(dim=0).nonzero().squeeze().tolist()
                if not isinstance(alive, list):
                    alive = [alive]
                alive_latents[name] = alive

        analysis_batch.update(alive_latents=alive_latents)
        return analysis_batch

class ModelForwardOp(AnalysisOp):
    """Analysis operation that performs a basic forward pass."""

    def __init__(self, name: str = 'model_forward',
                 description: str = 'Basic model forward pass',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None) -> None:
        super().__init__(name, description, output_schema, input_schema)

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Perform a basic forward pass."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Ensure we have answer indices
        if not hasattr(analysis_batch, 'answer_indices') or analysis_batch.answer_indices is None:
            analysis_batch = GetAnswerIndicesOp()(module, analysis_batch, batch, batch_idx)

        # Run forward pass
        answer_logits = module(**batch)
        analysis_batch.update(answer_logits=answer_logits)
        return analysis_batch

class ModelCacheForwardOp(AnalysisOp):
    """Analysis operation that performs a forward pass with cache."""

    def __init__(self, name: str = 'model_cache_forward',
                 description: str = 'Model forward pass with cache',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None) -> None:
        super().__init__(name, description, output_schema, input_schema)

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Perform a forward pass with cache."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Run with cache and SAEs
        answer_logits, cache = module.model.run_with_cache_with_saes(
            **batch, saes=module.sae_handles,
            names_filter=module.analysis_cfg.names_filter
        )

        # Get answer indices and alive latents
        analysis_batch = GetAnswerIndicesOp()(module, analysis_batch, batch, batch_idx)
        analysis_batch.update(cache=cache)
        analysis_batch = GetAliveLatentsOp()(module, analysis_batch, batch, batch_idx)

        analysis_batch.update(answer_logits=answer_logits)
        return analysis_batch

class ModelAblationOp(AnalysisOp):
    """Analysis operation that performs an ablation study."""

    def __init__(self, name: str = 'model_ablation',
                 description: str = 'Model ablation analysis',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None,
                 ablate_latent_fn: Callable = ablate_sae_latent) -> None:
        super().__init__(name, description, output_schema, input_schema)
        self.ablate_latent_fn = ablate_latent_fn

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Perform an ablation study."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Ensure we have answer indices and alive latents
        if not hasattr(analysis_batch, 'answer_indices') or analysis_batch.answer_indices is None:
            analysis_batch = GetAnswerIndicesOp()(module, analysis_batch, batch, batch_idx)

        if not hasattr(analysis_batch, 'alive_latents') or analysis_batch.alive_latents is None:
            assert module.analysis_cfg.input_store and hasattr(module.analysis_cfg.input_store, 'alive_latents'), \
                "alive_latents required for ablation op"
            analysis_batch = GetAliveLatentsOp()(module, analysis_batch, batch, batch_idx)

        answer_indices = analysis_batch.answer_indices
        alive_latents = analysis_batch.alive_latents

        # Run ablation for each latent
        per_latent_logits: dict[str, dict[Any, torch.Tensor]] = defaultdict(dict)
        for name, alive in alive_latents.items():
            for latent_idx in alive:
                fwd_hooks_cfg = [(name, partial(self.ablate_latent_fn,
                                                latent_idx=latent_idx,
                                                seq_pos=answer_indices))]
                answer_logits = module.model.run_with_hooks_with_saes(
                    **batch, saes=module.sae_handles,
                    clear_contexts=True, fwd_hooks=fwd_hooks_cfg
                )
                per_latent_logits[name][latent_idx] = answer_logits[
                    torch.arange(batch["input"].size(0)), answer_indices, :
                ]

        analysis_batch.update(answer_logits=per_latent_logits)
        return analysis_batch

class ModelGradientOp(AnalysisOp):
    """Analysis operation that performs gradient-based attribution."""

    def __init__(self, name: str = 'model_gradient',
                 description: str = 'Model gradient-based attribution',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None,
                 logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff) -> None:
        super().__init__(name, description, output_schema, input_schema)
        self.logit_diff_fn = logit_diff_fn

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Perform gradient-based attribution."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Ensure we have answer indices
        if not hasattr(analysis_batch, 'answer_indices') or analysis_batch.answer_indices is None:
            analysis_batch = GetAnswerIndicesOp()(module, analysis_batch, batch, batch_idx)

        answer_indices = analysis_batch.answer_indices

        # if we're running a manual analysis_step context, we may need to manually set hooks
        module.analysis_cfg.add_default_cache_hooks()
        # Verify hooks are configured
        assert all((module.analysis_cfg.fwd_hooks, module.analysis_cfg.bwd_hooks)), \
            "fwd_hooks and bwd_hooks required for gradient-based attribution op"

        # TODO: In the future, we will likely use IT dispatch logic to control toggling autograd/inference mode etc.
        #       but for now controlling manually here
        # Run with hooks and compute gradients
        with torch.set_grad_enabled(True):
            with module.model.saes(saes=module.sae_handles):
                with module.model.hooks(
                    fwd_hooks=module.analysis_cfg.fwd_hooks, bwd_hooks=module.analysis_cfg.bwd_hooks
                ):
                    answer_logits = module.model(**batch)
                    answer_logits = torch.squeeze(
                        answer_logits[torch.arange(batch["input"].size(0)), answer_indices],
                        dim=1
                    )

                    # Compute loss and logit differences
                    loss = module.loss_fn(answer_logits, analysis_batch.labels)
                    answer_logits = module.standardize_logits(answer_logits)
                    per_example_answers, _ = torch.max(answer_logits, dim=-2)
                    preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
                    logit_diffs = self.logit_diff_fn(answer_logits, target_indices=analysis_batch.orig_labels)

                    # Compute gradients
                    logit_diffs.sum().backward()
                    if logit_diffs.dim() == 0:
                        logit_diffs.unsqueeze_(0)
        analysis_batch.update(
            answer_logits=answer_logits,
            answer_indices=answer_indices,
            logit_diffs=logit_diffs,
            preds=preds,
            loss=loss,
            grad_cache=module.analysis_cfg.cache_dict  # Store the gradient cache
        )
        return analysis_batch

class GradientAttributionOp(AnalysisOp):
    """Analysis operation that computes attribution values from gradients."""

    def __init__(self, name: str = 'gradient_attribution',
                 description: str = 'Compute attribution values from gradients',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None) -> None:
        super().__init__(name, description, output_schema, input_schema)

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Compute attribution values from gradients."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Ensure required inputs exist
        required_inputs = ['answer_indices', 'logit_diffs']
        for key in required_inputs:
            if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
                raise ValueError(f"Missing required input '{key}' for gradient attribution")

        # Get cached activations (forwards) and gradients (backwards) from analysis_cfg.cache_dict
        from transformer_lens import ActivationCache
        cache_dict = module.analysis_cfg.cache_dict  # Use module's cache_dict instead of grad_cache
        batch_cache_dict = ActivationCache(cache_dict, module.model)
        batch_sz = batch["input"].size(0)

        # Get alive latents using GetAliveLatentsOp  # TODO: clean this up so no temp batch is required
        # Create a temporary analysis batch with the cache for GetAliveLatentsOp
        temp_batch = AnalysisBatch(
            cache=batch_cache_dict,
            answer_indices=analysis_batch.answer_indices
        )
        temp_batch = GetAliveLatentsOp()(module, temp_batch, batch, batch_idx)
        analysis_batch.alive_latents = temp_batch.alive_latents

        # Compute attribution values and correct activations
        attribution_values: dict[str, torch.Tensor] = {}
        correct_activations: dict[str, torch.Tensor] = {}

        # Process each forward hook
        for fwd_name in [name for name in batch_cache_dict.keys()
                         if module.analysis_cfg.names_filter(name) and not name.endswith("_grad")]:
            # Check if we have gradient information for this hook
            grad_name = f"{fwd_name}_grad"
            if grad_name not in batch_cache_dict:
                continue

            # Initialize attribution tensor
            attribution_values[fwd_name] = torch.zeros(batch_sz, module.sae_handles[0].cfg.d_sae)

            # Get activations and gradients at the answer indices
            fwd_hook_acts = batch_cache_dict[fwd_name][torch.arange(batch_sz), analysis_batch.answer_indices]
            bwd_hook_grads = batch_cache_dict[grad_name][torch.arange(batch_sz), analysis_batch.answer_indices]

            # Ensure tensors have the right shape (add batch dimension if needed)
            for t in [fwd_hook_acts, bwd_hook_grads]:
                if t.dim() == 2:
                    t.unsqueeze_(1)

            # Extract correct activations (for examples with positive logit differences)
            correct_activations[fwd_name] = torch.squeeze(
                fwd_hook_acts[(analysis_batch.logit_diffs > 0), :, :], dim=1
            )

            # Calculate attribution as activations × gradients for the alive latents
            alive_indices = analysis_batch.alive_latents[fwd_name]
            attribution_values[fwd_name][:, alive_indices] = torch.squeeze(
                (bwd_hook_grads[:, :, alive_indices] *
                 fwd_hook_acts[:, :, alive_indices]).cpu(), dim=1
            )

        # Update the analysis batch with results
        analysis_batch.update(
            attribution_values=attribution_values,
            correct_activations=correct_activations
        )

        return analysis_batch

class LogitDiffsOp(AnalysisOp):
    """Analysis operation that computes logit differences."""
    def __init__(self, name: str, description: str,
                 output_schema: OpSchema,
                 input_schema: Optional[OpSchema] = None,
                 logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff) -> None:
        # This op requires explicit schema since it's instantiated with different schemas
        super().__init__(name, description, output_schema, input_schema)
        self.logit_diff_fn = logit_diff_fn

    def get_loss_preds_diffs(self, module: torch.nn.Module, analysis_batch: AnalysisBatchProtocol,
                             answer_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor]:
        # TODO: prob should remove dependency on module handle here with other composable ops
        loss = module.loss_fn(answer_logits, analysis_batch.labels)
        answer_logits = module.standardize_logits(answer_logits)
        per_example_answers, _ = torch.max(answer_logits, dim=-2)
        preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
        # Use the op's logit_diff_fn instead of analysis_cfg
        logit_diffs = self.logit_diff_fn(answer_logits, target_indices=analysis_batch.orig_labels)
        return loss, logit_diffs, preds, answer_logits

    def __call__(self, module: torch.nn.Module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Execute the logit differences operation."""
        logits, indices = analysis_batch.answer_logits, analysis_batch.answer_indices
        answer_logits = torch.squeeze(logits[torch.arange(batch["input"].size(0)), indices], dim=1)
        loss, logit_diffs, preds, answer_logits = self.get_loss_preds_diffs(module, analysis_batch, answer_logits)
        if logit_diffs.dim() == 0:
            logit_diffs.unsqueeze_(0)
        analysis_batch.update(loss=loss, logit_diffs=logit_diffs, preds=preds, answer_logits=answer_logits)
        return analysis_batch

class SAECorrectActivationsOp(AnalysisOp):
    """Analysis operation that computes correct activations from SAE outputs."""
    def __init__(self, name: str = 'sae_correct_acts',
                 description: str = 'Compute correct activations from SAE cache',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None) -> None:
        super().__init__(name, description, output_schema, input_schema)

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Extract correct activations for examples with positive logit diffs."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        correct_activations: dict[str, torch.Tensor] = {}
        logit_diffs, indices = analysis_batch.logit_diffs, analysis_batch.answer_indices
        logit_diffs = logit_diffs.cpu()

        # Filter cache keys using module's names_filter
        for name in analysis_batch.cache.keys():
            if module.analysis_cfg.names_filter(name):
                correct_activations[name] = analysis_batch.cache[name][logit_diffs > 0, indices[logit_diffs > 0], :]

        analysis_batch.update(correct_activations=correct_activations)
        return analysis_batch

class AblationAttributionOp(AnalysisOp):
    """Analysis operation that computes attribution values using latent ablation."""
    def __init__(self, name: str = 'ablation_attribution',
                 description: str = 'Compute attribution using latent ablation',
                 output_schema: Optional[OpSchema] = None,
                 input_schema: Optional[OpSchema] = None,
                 logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff) -> None:
        super().__init__(name, description, output_schema, input_schema)
        self.logit_diff_fn = logit_diff_fn

    def get_loss_preds_diffs(self, module, analysis_batch: AnalysisBatchProtocol,
                             answer_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor]:
        loss = module.loss_fn(answer_logits, analysis_batch.labels)
        answer_logits = module.standardize_logits(answer_logits)
        per_example_answers, _ = torch.max(answer_logits, dim=-2)
        preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
        logit_diffs = self.logit_diff_fn(answer_logits, target_indices=analysis_batch.orig_labels)
        return loss, logit_diffs, preds, answer_logits

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Compute attribution values using latent ablation."""
        if analysis_batch is None:
            analysis_batch = AnalysisBatch()

        # Ensure we have required inputs
        required_inputs = ['answer_logits', 'answer_indices', 'alive_latents', 'logit_diffs']
        for key in required_inputs:
            if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
                raise ValueError(f"Missing required input '{key}' for ablation attribution")

        # Initialize result structures
        attribution_values: dict[str, torch.Tensor] = {}
        per_latent = {"loss": defaultdict(dict), "logit_diffs": defaultdict(dict),
                      "preds": defaultdict(dict), "answer_logits": defaultdict(dict)}

        # Process per-latent logits for each hook
        for act_name, logits in analysis_batch.answer_logits.items():
            attribution_values[act_name] = torch.zeros(batch["input"].size(0), module.sae_handles[0].cfg.d_sae)
            for latent_idx in analysis_batch.alive_latents[act_name]:
                # Calculate metrics for this latent
                loss_preds_diffs = self.get_loss_preds_diffs(module, analysis_batch, logits[latent_idx])

                # Store per-latent metrics
                for metric_name, value in zip(per_latent.keys(), loss_preds_diffs):
                    per_latent[metric_name][act_name][latent_idx] = value

                # Calculate attribution values
                example_mask = (per_latent["logit_diffs"][act_name][latent_idx] > 0).cpu()
                per_latent["logit_diffs"][act_name][latent_idx] = (
                    per_latent["logit_diffs"][act_name][latent_idx][example_mask].detach().cpu()
                )

                base_diffs = analysis_batch.logit_diffs
                for t in [example_mask, base_diffs]:
                    if t.dim() == 0:
                        t.unsqueeze_(0)
                base_diffs = base_diffs.cpu()

                # Attribution is difference between base and ablated performance
                attribution_values[act_name][example_mask, latent_idx] = (
                    base_diffs[example_mask] - per_latent["logit_diffs"][act_name][latent_idx]
                )

        # Update analysis batch with results
        for key in per_latent:
            analysis_batch.update(**{key: per_latent[key]})
        analysis_batch.update(attribution_values=attribution_values)

        return analysis_batch
