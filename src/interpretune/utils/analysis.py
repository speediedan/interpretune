from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Literal, Tuple, List, Union
from collections import defaultdict
from functools import partial
from enum import auto

from jaxtyping import Float
import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint, NamesFilter
from IPython.display import IFrame, display

from interpretune.utils.import_utils import _SL_AVAILABLE
from interpretune.adapters.sae_lens import SAELensModule
from interpretune.base.config.shared import AutoStrEnum


if _SL_AVAILABLE:
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

else:
    def get_pretrained_saes_directory():
        raise NotImplementedError("sae_lens not available")


def construct_names_filter(
    module: SAELensModule,
    target_hooks: List[Tuple[int, str]] | Tuple[List[int] | int, List[str] | str]
) -> NamesFilter:
    available_hooks = {f'{handle.cfg.hook_name}.{key}' for handle in module.sae_handles
                          for key in handle.hook_dict.keys()}
    def hook_filter(layer, name):
        return (hook for hook in available_hooks if f"blocks.{layer}." in hook and name in hook)

    if isinstance(target_hooks, tuple):  # handle tuple of (layer, hook_name) lists or single layer/hook_names
        target_layers, target_hook_names = target_hooks
        if isinstance(target_layers, int):
            target_layers = [target_layers]
        if isinstance(target_hook_names, str):
            target_hook_names = [target_hook_names]
        names_filter = [hook for layer in target_layers for n in target_hook_names for hook in hook_filter(layer, n)]
    else:  # list of (layer, hook_name) tuples
        names_filter = [hook for layer, n in target_hooks for hook in hook_filter(layer, n)]
    return names_filter


# similar to logic in `transformer_lens.hook_points.get_caching_hooks` but accessible to other functions
def resolve_names_filter(names_filter: Optional[NamesFilter]) -> Callable[[str], bool]:
    if names_filter is None:
        names_filter = lambda name: True
    elif isinstance(names_filter, str):
        filter_str = names_filter
        names_filter = lambda name: name == filter_str
    elif isinstance(names_filter, list):
        filter_list = names_filter
        names_filter = lambda name: name in filter_list
    elif callable(names_filter):
        names_filter = names_filter
    else:
        raise ValueError("names_filter must be a string, list of strings, or function")
    assert callable(names_filter)
    return names_filter


DEFAULT_DECODE_KWARGS = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}

class AnalysisMode(AutoStrEnum):
    clean_no_sae = auto()
    clean_w_sae = auto()
    attr_patching = auto()
    ablation = auto()


@dataclass(kw_only=True)
class AnalysisBatch:
    """Contains all analysis results for a single batch."""
    logit_diffs: torch.Tensor | Dict[str, Dict[int, torch.Tensor]] = None
    answer_logits: torch.Tensor | Dict[str, Dict[int, torch.Tensor]] = None
    loss: torch.Tensor | Dict[str, Dict[int, torch.Tensor]] = None
    labels: torch.Tensor = None
    orig_labels: torch.Tensor = None
    preds: torch.Tensor | Dict[str, Dict[int, torch.Tensor]] = None
    cache: ActivationCache = None
    grad_cache: ActivationCache = None
    answer_indices: torch.Tensor = None
    alive_latents: dict[str, list[int]] = None
    correct_activations: dict[str, torch.Tensor] = None
    ablation_effects: dict[str, torch.Tensor] = field(default_factory=dict)
    attribution_values: dict[str, torch.Tensor] = field(default_factory=dict)
    tokens: torch.Tensor = None
    prompts: list[str] = None

    def update(self, **kwargs):
        """Update multiple fields of the dataclass at once."""
        for key, value in kwargs.items():
            if key in self.__annotations__:
                setattr(self, key, value)

    def to_cpu(self):
        """Detach and move all AnalysisBatch field tensors to CPU."""
        def maybe_detach_to_cpu(val):
            if isinstance(val, torch.Tensor):
                return val.detach().cpu()
            elif isinstance(val, dict):
                return {k: maybe_detach_to_cpu(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [maybe_detach_to_cpu(v) for v in val]
            return val

        for key, val in self.__dict__.items():
            setattr(self, key, maybe_detach_to_cpu(val))

@dataclass(kw_only=True)
class SaveCfg:
    analysis_cache: Optional["AnalysisCache"] = None
    prompts: bool = False
    tokens: bool = False
    cache: bool = False
    grad_cache: bool = False
    decode_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        default_decode_kwargs = DEFAULT_DECODE_KWARGS.copy()
        for k, v in default_decode_kwargs.items():
            self.decode_kwargs.setdefault(k, v)

    def wrap_summary(self, analysis_batch: AnalysisBatch, batch: BatchEncoding,
                     tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        if self.prompts:
            assert tokenizer is not None, "Tokenizer is required to decode prompts"
            analysis_batch.prompts = tokenizer.batch_decode(batch['input'], **self.decode_kwargs)
        if self.tokens:
            analysis_batch.tokens = batch['input'].detach().cpu()
        for cache_type, should_keep in [("cache", self.cache), ("grad_cache", self.grad_cache)]:
            if not should_keep and getattr(analysis_batch, cache_type, None):
                setattr(analysis_batch, cache_type, None)
        analysis_batch.to_cpu()
        self.analysis_cache.batches.append(analysis_batch)

class SAEAnalysisDict(dict):
    """Dictionary for SAE-specific data where values must be torch.Tensor or list[torch.Tensor]."""

    def __setitem__(self, key: str, value: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        if not isinstance(value, (torch.Tensor, list)):
            raise TypeError("Values must be torch.Tensor or list[torch.Tensor]")
        if isinstance(value, list) and not all(isinstance(v, torch.Tensor) for v in value):
            raise TypeError("All list elements must be torch.Tensor")
        super().__setitem__(key, value)

    @property
    def shapes(self) -> dict[str, Union[torch.Size, list[torch.Size]]]:
        """Return shapes for each tensor or list of tensors in the dictionary.

        Returns:
            Dictionary mapping SAE names to either single tensor shapes or lists of tensor shapes
        """
        shapes = {}
        for sae, values in self.items():
            if isinstance(values, torch.Tensor):
                shapes[sae] = values.shape
            elif isinstance(values, list):
                shapes[sae] = [t.shape for t in values]
        return shapes

    def batch_join(self, across_saes: bool = False, join_fn: Callable = torch.cat
                   ) -> Union["SAEAnalysisDict", list[torch.Tensor]]:
        """Join field values either by SAE or across SAEs.

        Args:
            join_across_saes: If True, joins values across SAEs for each batch.
                                If False, joins batches for each SAE separately.
            join_fn: Function to use for joining (default: torch.cat)

        Returns:
            If join_across_saes=True: List of tensors, one per batch, with values joined across SAEs
            If join_across_saes=False: SAEAnalysisDict with batches joined for each SAE
        """
        if across_saes:
            # Get number of batches from first SAE's values
            num_batches = len(next(iter(self.values())))

            # For each batch, collect and join tensors from all SAEs
            result = []
            for batch_idx in range(num_batches):
                batch_tensors = []
                for sae_values in self.values():
                    batch_tensors.append(sae_values[batch_idx])
                result.append(join_fn(batch_tensors))
            return result
        else:
            # Join batches for each SAE separately
            result = SAEAnalysisDict()
            for k, v in self.items():
                result[k] = join_fn(v, dim=0)
            return result

    def apply_op_by_sae(self, operation: Union[Callable, str],
                        *args, **kwargs) -> "SAEAnalysisDict":
        """Apply an operation to each tensor value while preserving SAE keys.

        Args:
            operation: Either callable or string name of torch.Tensor method
            *args, **kwargs: Additional arguments passed to the operation

        Returns:
            SAEAnalysisDict: New dictionary mapping SAE names to operated tensor values

        Examples:
            # Apply mean
            my_dict.batch_join().apply_op_by_sae('mean', dim=0)

            # Apply custom function
            my_dict.batch_join().apply_op_by_sae(torch.mean, dim=0)
        """
        result = SAEAnalysisDict()

        for k, v in self.items():
            if isinstance(operation, str):
                result[k] = getattr(v, operation)(*args, **kwargs)
            else:
                result[k] = operation(v, *args, **kwargs)

        return result


# TODO: maybe keep as tensors via cat for future analysis rather than separate batches
@dataclass(kw_only=True)
class AnalysisCache:
    batches: list[AnalysisBatch] = field(default_factory=list)
    save_cfg: SaveCfg = field(default_factory=SaveCfg)

    def __post_init__(self):
        self.save_cfg.analysis_cache = self

    def save(self, analysis_batch: AnalysisBatch, batch: BatchEncoding,
             tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.save_cfg.wrap_summary(analysis_batch=analysis_batch, batch=batch, tokenizer=tokenizer)

    def __getattr__(self, name: str) -> list:
        """Get list of values across all batches for a given field."""
        if name not in AnalysisBatch.__annotations__:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        return [getattr(batch, name) for batch in self.batches if getattr(batch, name) is not None]

    def by_sae(self, field_name: str, stack_latents: bool = True) -> SAEAnalysisDict:
        """Transform batch-oriented field values into per-SAE lists of batch values.

        Args:
            field_name: Name of the field to process (e.g. 'correct_activations', 'preds')
            stack_latents: Whether to stack latent values using torch.stack for nested dictionary fields

        Returns:
            SAEAnalysisDict: Dictionary mapping SAE names to lists of batch values.
            For nested dictionary fields, latent values within each batch are stacked if stack_latents=True.

        Raises:
            TypeError: If the values are not dictionaries and thus cannot be transformed into an SAEAnalysisDict
        """
        values = getattr(self, field_name)
        assert values, f"No values found for field {field_name}"
        if not isinstance(values[0], dict):
            raise TypeError(
                f"Values for field {field_name} must be dictionaries to be transformed into an SAEAnalysisDict"
            )

        result = SAEAnalysisDict()
        sae_names = values[0].keys()

        for sae in sae_names:
            if isinstance(values[0][sae], dict) and stack_latents:
                # Stack latent tensors for each batch
                batch_tensors = []
                for batch in values:
                    latent_tensors = [t for t in batch[sae].values()]
                    batch_tensors.append(torch.stack(latent_tensors))
                result[sae] = batch_tensors
            else:
                # Handle both non-nested and non-stacked nested cases
                result[sae] = [batch[sae] for batch in values]
        return result

def _make_simple_cache_hook(cache_dict: dict, is_backward: bool = False) -> Callable:
    """Create a hook function that caches activations.

    Args:
        is_backward: Whether this is a backward hook
        cache_dict: Dictionary to store cached activations
    """
    def cache_hook(act, hook):
        assert hook.name is not None
        hook_name = hook.name
        if is_backward:
            hook_name += "_grad"
        cache_dict[hook_name] = act.detach()
    return cache_hook

@dataclass(kw_only=True)
class AnalysisCfg:
    analysis_cache: AnalysisCache = field(default_factory=AnalysisCache)
    mode: Union[str, AnalysisMode] = AnalysisMode.clean_no_sae
    ablate_latent_fn: Optional[Callable] = None  # set to default ablate_sae_latent
    logit_diff_fn: Optional[Callable] = None  # set to default boolean_logits_to_avg_logit_diff
    # each tuple of (hook_name/((str) -> bool)/sequence(hook_names), hook_fn) (i.e. NamesFilter, hook_fn) will be
    # used by transformer_lens.hook_points.hook context manager to enable the relevant hooks for each direction
    # (which ultimately are converted to LensHandles at the pytorch interface)
    fwd_hooks: List[Tuple[str | Callable, Callable]] = field(default_factory=list)
    bwd_hooks: List[Tuple[str | Callable, Callable]] = field(default_factory=list)
    cache_dict: Dict = field(default_factory=dict)  # handle for fwd_hook/bwd_hook construction in attr_patching mode
    # note, may need to be validated if fwd_hook/bwd_hooks are separately/explicitly defined
    names_filter: Optional[NamesFilter] = None
    base_logit_diffs: List = field(default_factory=list)  # only used for ablation mode currently
    # optional cached values to avoid recomputation, only relevant for ablation mode currently
    alive_latents: list[dict] = field(default_factory=list)
    answer_indices: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = AnalysisMode(self.mode)
        self.names_filter = resolve_names_filter(self.names_filter)
        # Only auto-construct hooks if none provided explicitly
        if not self.fwd_hooks and not self.bwd_hooks:
            self.fwd_hooks, self.bwd_hooks = self.check_add_default_hooks(
                mode=self.mode,
                names_filter=self.names_filter,
                cache_dict=self.cache_dict,
            )
        if self.logit_diff_fn is None:
            self.logit_diff_fn = boolean_logits_to_avg_logit_diff
        if self.ablate_latent_fn is None:
            self.ablate_latent_fn = ablate_sae_latent
        # TODO: prob a good idea to add a sanity check that names_filter is provided when needed for the relevant mode
        #       when fwd_hooks/bwd_hooks are explicitly defined, prob makes sense to validate that names_filter is a
        #       subset of fwd_hook and possibly bwd_hook ({name}_grad) names for the relevant modes

    def check_add_default_hooks(
            self, mode: Union[str, AnalysisMode], names_filter: Optional[str | Callable],
            cache_dict: Optional[dict]
            ) -> Tuple[List[Tuple[str | Callable, Callable]], List[Tuple[str | Callable, Callable]]]:
        """Construct forward and backward hooks based on analysis mode."""
        if isinstance(mode, str):
            mode = AnalysisMode(mode)

        fwd_hooks, bwd_hooks = [], []

        if mode == AnalysisMode.clean_no_sae:
            return fwd_hooks, bwd_hooks
        if names_filter is None:
            raise ValueError("names_filter required for non-clean modes")
        if mode == AnalysisMode.attr_patching:
            fwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict))
            )
            bwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict, is_backward=True))
            )
        return fwd_hooks, bwd_hooks

def compute_correct(analysis_obj: AnalysisCache | AnalysisCfg,
                    mode: Union[str, AnalysisMode, None] = None) -> tuple[int, float, Optional[list]]:
    """Compute correct prediction statistics for a given analysis mode.

    Args:
        log_summs: Either an AnalysisCache containing prediction results or an AnalysisCfg object
        mode: Analysis mode to compute statistics for. Only required if passing an AnalysisCache

    Returns:
        Tuple of (total_correct, percentage_correct, batch_predictions) for the given mode
        batch_predictions is None except for ablation mode where it contains the modal predictions
    """
    # Handle input type and get mode
    if isinstance(analysis_obj, AnalysisCfg):
        analysis_cache = analysis_obj.analysis_cache
        mode = analysis_obj.mode
    else:
        if mode is None:
            raise ValueError("mode argument required when passing AnalysisCache")
        analysis_cache = analysis_obj

    if isinstance(mode, str):
        mode = AnalysisMode(mode)

    batch_preds = (
        [b.mode(dim=0).values.cpu() for b in analysis_cache.by_sae('preds').batch_join(across_saes=True)]
        if mode == AnalysisMode.ablation else analysis_cache.preds
    )
    correct_statuses = [
        (labels == preds).nonzero().unique().size(0)
        for labels, preds in zip(analysis_cache.orig_labels, batch_preds)
    ]
    total_correct = sum(correct_statuses)
    percentage_correct = total_correct / (len(torch.cat(analysis_cache.orig_labels))) * 100
    return (total_correct, percentage_correct, batch_preds if mode == AnalysisMode.ablation else None)

def get_latents_and_indices(module: SAELensModule, analysis_cfg: AnalysisCfg, batch: BatchEncoding, batch_idx: int,
                            analysis_batch: Optional[AnalysisBatch] = None,
                            cache: Optional[ActivationCache] = None) -> Tuple[torch.Tensor, dict]:
    if getattr(analysis_batch, 'answer_indices', None) is not None:
        answer_indices = analysis_batch.answer_indices
    elif analysis_cfg.answer_indices:
        answer_indices = analysis_cfg.answer_indices[batch_idx]
    else:
        answer_indices = resolve_answer_indices(module, batch)

    if analysis_cfg.alive_latents:
        alive_latents = analysis_cfg.alive_latents[batch_idx]
    elif not cache:
        alive_latents = {}
    else:
        alive_latents = batch_alive_latents(answer_indices, cache, analysis_cfg.names_filter)

    if analysis_batch is not None:
        analysis_batch.update(answer_indices=answer_indices, alive_latents=alive_latents)
    else:
        return answer_indices, alive_latents

def get_loss_preds_diffs(module: SAELensModule, analysis_batch: AnalysisBatch, analysis_cfg: AnalysisCfg,
                            answer_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                  torch.Tensor]:
    """Standardize logits and compute loss, predictions and logit differences."""
    loss = module.loss_fn(answer_logits, analysis_batch.labels)
    answer_logits = module.standardize_logits(answer_logits)
    per_example_answers, _ = torch.max(answer_logits, dim=-2)
    preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
    logit_diffs = analysis_cfg.logit_diff_fn(answer_logits, target_indices=analysis_batch.orig_labels)
    return loss, logit_diffs, preds, answer_logits

def run_with_ctx(module: SAELensModule, analysis_batch: AnalysisBatch, analysis_cfg: AnalysisCfg, batch: BatchEncoding,
                 batch_idx: int, **kwargs):
    if analysis_cfg.mode == AnalysisMode.clean_no_sae:
        answer_logits = module(**batch)
        answer_indices, alive_latents = get_latents_and_indices(module, analysis_cfg, batch, batch_idx)
        analysis_batch.update(answer_indices=answer_indices, alive_latents=alive_latents)
    elif analysis_cfg.mode == AnalysisMode.clean_w_sae:
        answer_logits, cache = module.model.run_with_cache_with_saes(**batch, saes=module.sae_handles,
                                                     names_filter=analysis_cfg.names_filter)
        answer_indices, alive_latents = get_latents_and_indices(module, analysis_cfg, batch, batch_idx, cache=cache)
        analysis_batch.update(cache=cache, answer_indices=answer_indices, alive_latents=alive_latents)
    elif analysis_cfg.mode == AnalysisMode.ablation:
        assert analysis_cfg.alive_latents, "alive_latents required for ablation mode"
        answer_indices, alive_latents = get_latents_and_indices(module, analysis_cfg, batch, batch_idx)
        per_latent_logits = defaultdict(dict)
        for name, alive in alive_latents.items():
            for latent_idx in alive:
                fwd_hooks_cfg = [(name, partial(analysis_cfg.ablate_latent_fn, latent_idx=latent_idx,
                                                seq_pos=answer_indices))]
                answer_logits = module.model.run_with_hooks_with_saes(**batch, saes=module.sae_handles,
                                                                    clear_contexts=True, fwd_hooks=fwd_hooks_cfg)
                per_latent_logits[name][latent_idx] = answer_logits[torch.arange(batch["input"].size(0)),
                                                                    answer_indices, :]
        analysis_batch.update(alive_latents=alive_latents, answer_indices=answer_indices)
        answer_logits = per_latent_logits
    elif analysis_cfg.mode == AnalysisMode.attr_patching:
        assert all((analysis_cfg.fwd_hooks, analysis_cfg.bwd_hooks)), \
        "fwd_hooks and bwd_hooks required for attr_patching mode"
        answer_indices, _ = get_latents_and_indices(module, analysis_cfg, batch, batch_idx)
        with module.model.saes(saes=module.sae_handles):
            with module.model.hooks(fwd_hooks=analysis_cfg.fwd_hooks, bwd_hooks=analysis_cfg.bwd_hooks):
                answer_logits = module.model(**batch)
                answer_logits = torch.squeeze(answer_logits[torch.arange(batch["input"].size(0)), answer_indices],
                                              dim=1)
                loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(module, analysis_batch, analysis_cfg,
                                                                               answer_logits)
                logit_diffs.sum().backward()
                if logit_diffs.dim() == 0:
                    logit_diffs.unsqueeze_(0)
        analysis_batch.update(answer_indices=answer_indices, logit_diffs=logit_diffs, preds=preds, loss=loss)
    else:
        answer_logits = module.model.run_with_saes(**batch, saes=module.sae_handles)
    analysis_batch.update(answer_logits=answer_logits)

def calc_ablation_effects(module: SAELensModule, analysis_batch: AnalysisBatch, analysis_cfg: AnalysisCfg,
                          batch: BatchEncoding, batch_idx: int) -> Tuple[dict, dict]:
    assert analysis_cfg.base_logit_diffs, "base_logit_diffs required for ablation mode"
    if batch_idx >= len(analysis_cfg.base_logit_diffs):
        raise IndexError(f"Batch index {batch_idx} out of range for base_logit_diffs")
    ablation_effects = {}
    per_latent = {'loss': defaultdict(dict),'logit_diffs': defaultdict(dict),'preds': defaultdict(dict),
                    'answer_logits': defaultdict(dict)}
    for act_name, logits in analysis_batch.answer_logits.items():
        ablation_effects[act_name] = torch.zeros(batch["input"].size(0), module.sae_handles[0].cfg.d_sae)
        for latent_idx in analysis_batch.alive_latents[act_name]:
            loss_preds_diffs = get_loss_preds_diffs(module, analysis_batch, analysis_cfg, logits[latent_idx])
            for metric_name, value in zip(per_latent.keys(), loss_preds_diffs):
                per_latent[metric_name][act_name][latent_idx] = value

            example_mask = (per_latent['logit_diffs'][act_name][latent_idx] > 0).cpu()
            per_latent['logit_diffs'][act_name][latent_idx] = (
                per_latent['logit_diffs'][act_name][latent_idx][example_mask].detach().cpu()
            )
            base_diffs = torch.as_tensor(analysis_cfg.base_logit_diffs[batch_idx])
            for t in [example_mask, base_diffs]:
                if t.dim() == 0:
                    t.unsqueeze_(0)
            base_diffs = base_diffs.cpu()
            ablation_effects[act_name][example_mask, latent_idx] = (
                base_diffs[example_mask] - per_latent['logit_diffs'][act_name][latent_idx]
            )
    per_latent_results = {key: per_latent[key] for key in ['loss', 'logit_diffs', 'preds', 'answer_logits']}
    return per_latent_results, ablation_effects

def calc_attribution_values(module: SAELensModule, analysis_batch: AnalysisBatch, analysis_cfg: AnalysisCfg,
                            batch: BatchEncoding, batch_idx: int) -> Tuple[dict, dict]:
    attribution_values = {}
    correct_activations = {}
    batch_cache_dict = ActivationCache(analysis_cfg.cache_dict, module.model)
    get_latents_and_indices(module, analysis_cfg, batch, batch_idx, cache=batch_cache_dict,
                            analysis_batch=analysis_batch)
    for fwd_name in [name for name in batch_cache_dict if analysis_cfg.names_filter(name)]:
        attribution_values[fwd_name] = torch.zeros(batch["input"].size(0), module.sae_handles[0].cfg.d_sae)
        fwd_hook_acts = batch_cache_dict[fwd_name][torch.arange(batch["input"].size(0)), analysis_batch.answer_indices]
        bwd_hook_grads = batch_cache_dict[f'{fwd_name}_grad'][torch.arange(batch["input"].size(0)),
                                                              analysis_batch.answer_indices]
        for t in [fwd_hook_acts, bwd_hook_grads]:
            if t.dim() == 2:
                t.unsqueeze_(1)
        correct_activations[fwd_name] = torch.squeeze(fwd_hook_acts[(analysis_batch.logit_diffs > 0), :, :], dim=1)
        attribution_values[fwd_name][:, analysis_batch.alive_latents[fwd_name]] = \
            torch.squeeze(
                (bwd_hook_grads[:, :, analysis_batch.alive_latents[fwd_name]] *
                 fwd_hook_acts[:, :, analysis_batch.alive_latents[fwd_name]]
                 ).cpu(), dim=1)
    return attribution_values, correct_activations

def calc_clean_diffs(module: SAELensModule, analysis_batch: AnalysisBatch, analysis_cfg: AnalysisCfg,
                     batch: BatchEncoding) -> None:
    logits, indices = analysis_batch.answer_logits, analysis_batch.answer_indices
    answer_logits = torch.squeeze(logits[torch.arange(batch["input"].size(0)), indices], dim=1)
    loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(module, analysis_batch, analysis_cfg, answer_logits)
    if logit_diffs.dim() == 0:
        logit_diffs.unsqueeze_(0)
    analysis_batch.update(loss=loss, logit_diffs=logit_diffs, preds=preds, answer_logits=answer_logits)
    if analysis_cfg.mode == AnalysisMode.clean_w_sae:
        correct_activations = {}
        logit_diffs = logit_diffs.cpu()
        for name in analysis_batch.cache.keys():
            if analysis_cfg.names_filter(name):
                correct_activations[name] = analysis_batch.cache[name][logit_diffs > 0, indices[logit_diffs > 0], :]
        analysis_batch.update(correct_activations=correct_activations)

def loss_and_logit_diffs(module: SAELensModule, analysis_batch: AnalysisBatch, analysis_cfg: AnalysisCfg,
                         batch: BatchEncoding, batch_idx: int) -> None:
    if analysis_cfg.mode in [AnalysisMode.clean_w_sae, AnalysisMode.clean_no_sae]:
        calc_clean_diffs(module, analysis_batch, analysis_cfg, batch)
    elif analysis_cfg.mode == AnalysisMode.ablation:
        per_latent_results, ablation_effects = calc_ablation_effects(
            module=module, analysis_batch=analysis_batch, analysis_cfg=analysis_cfg, batch=batch, batch_idx=batch_idx)
        analysis_batch.update(**per_latent_results, ablation_effects=ablation_effects)
    elif analysis_cfg.mode == AnalysisMode.attr_patching:
        attribution_values, correct_activations = calc_attribution_values(
            module=module, analysis_batch=analysis_batch, analysis_cfg=analysis_cfg, batch=batch, batch_idx=batch_idx)
        analysis_batch.update(attribution_values=attribution_values, correct_activations=correct_activations)
    else:
        raise ValueError(f"Unsupported analysis mode {analysis_cfg.mode}")

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

def resolve_answer_indices(module: SAELensModule, batch):
    tokens = batch['input'].detach().cpu()
    if module.datamodule.tokenizer.padding_side == "left":
        return torch.full((tokens.size(0),), -1)
    nonpadding_mask = tokens != module.datamodule.tokenizer.pad_token_id
    answer_indices = torch.where(nonpadding_mask, 1, 0).sum(dim=1) - 1
    return answer_indices

def batch_alive_latents(answer_indices, cache, names_filter):
    # TODO: decide whether to check supported sae activation hooks here or just leave to the user to ensure correctness
    filtered_acts = {name: acts for name, acts in cache.items() if names_filter(name)}
    alive_latents = {}
    for name, acts in filtered_acts.items():
        alive_latents[name] = (acts[torch.arange(acts.size(0)),
                                    answer_indices, :] > 0).any(dim=0).nonzero().squeeze().tolist()
        if not isinstance(alive_latents[name], list):
            alive_latents[name] = [alive_latents[name]]
    return alive_latents

def ablate_sae_latent(
    sae_acts: torch.Tensor,
    hook: HookPoint,
    latent_idx: int | None = None,
    seq_pos: torch.Tensor | None = None,  # batched
) -> torch.Tensor:
    """Ablate a particular latent at a particular sequence position.

    If either argument is None, we ablate at all latents / sequence positions.
    """
    sae_acts[torch.arange(sae_acts.size(0)), seq_pos, latent_idx] = 0.0
    return sae_acts

def display_dashboard(sae_release="gpt2-small-res-jb", sae_id="blocks.9.hook_resid_pre", latent_idx=0, width=800,
                      height=600):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]
    embed_cfg= "embed=true&embedexplanation=true&embedplots=true&embedtest=true"
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?{embed_cfg}&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))
