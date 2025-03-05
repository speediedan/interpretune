from typing import Optional, Sequence, Dict
from dataclasses import dataclass, field
from pathlib import Path
import os

from datasets.config import HF_CACHE_HOME
from transformers import BatchEncoding, PreTrainedTokenizerBase

import interpretune as it

from interpretune.analysis import (SAEAnalysisTargets, ANALYSIS_OPS, AnalysisOp, AnalysisStore, resolve_names_filter,
                                   _make_simple_cache_hook)
from interpretune.config import ITSerializableCfg
from interpretune.protocol import NamesFilter, AnalysisStoreProtocol, AnalysisBatchProtocol
from interpretune.utils import DEFAULT_DECODE_KWARGS, MisconfigurationException
from interpretune.utils import rank_zero_warn

IT_ANALYSIS_CACHE_DIR = "interpretune"
DEFAULT_IT_ANALYSIS_CACHE = os.path.join(HF_CACHE_HOME, IT_ANALYSIS_CACHE_DIR)
IT_ANALYSIS_CACHE = Path(os.getenv("IT_ANALYSIS_CACHE_DIR", DEFAULT_IT_ANALYSIS_CACHE))


# TODO: move update to a dataclass multi-field update mixin if keeping this a dataclass
@dataclass(kw_only=True)
class AnalysisCfg(ITSerializableCfg):
    output_store: AnalysisStoreProtocol = None  # usually constructed on setup()
    input_store: AnalysisStoreProtocol | None = None  # store containing input data from previous op
    op: str | AnalysisOp = field(default_factory=lambda: it.logit_diffs_base)  # Renamed from mode
    fwd_hooks: list[tuple] = field(default_factory=list)
    bwd_hooks: list[tuple] = field(default_factory=list)
    cache_dict: dict = field(default_factory=dict)
    names_filter: NamesFilter | None = None
    save_prompts: bool = False
    save_tokens: bool = False
    decode_kwargs: dict = field(default_factory=lambda: DEFAULT_DECODE_KWARGS)
    sae_analysis_targets: SAEAnalysisTargets = None

    def __post_init__(self):
        # Convert string op to AnalysisOp
        if isinstance(self.op, str):
            self.op = ANALYSIS_OPS[self.op]

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
            self.fwd_hooks, self.bwd_hooks = self.check_add_default_hooks(
                op=self.op,
                names_filter=self.names_filter,
                cache_dict=self.cache_dict,
            )

    def prepare_model_ctx(self, module, fallback_sae_targets: Optional[SAEAnalysisTargets] = None) -> None:
        """Configure names_filter and hooks for a specific module.

        Args:
            module: The module to configure for.
            fallback_sae_targets: Optional fallback SAEAnalysisTargets to use if this config doesn't have one.
        """
        self.materialize_names_filter(module, fallback_sae_targets)
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
        yield self.op.save_batch(analysis_batch, batch, tokenizer=tokenizer, save_prompts=self.save_prompts,
                           save_tokens=self.save_tokens, decode_kwargs=self.decode_kwargs)

    def check_add_default_hooks(
            self, op: AnalysisOp, names_filter: NamesFilter | None,
            cache_dict: dict | None
            ) -> tuple[list[tuple], list[tuple]]:
        """Construct forward and backward hooks based on analysis operation."""
        fwd_hooks, bwd_hooks = [], []

        if op.name == 'logit_diffs.base':
            return fwd_hooks, bwd_hooks
        if names_filter is None:
            raise ValueError("names_filter required for non-clean operations")
        if op.name == 'logit_diffs.attribution.grad_based':
            fwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict))
            )
            bwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict, is_backward=True))
            )
        return fwd_hooks, bwd_hooks

    def apply(self, module, cache_dir: Optional[str] = None, op_output_dataset_path: Optional[str] = None,
              fallback_sae_targets: Optional[SAEAnalysisTargets] = None):
        """Set up analysis configuration and configure for the given module.

        This method handles both setting up the analysis store and configuring
        the names_filter and hooks for the module.

        Args:
            module: The module to configure for.
            cache_dir: Optional cache directory.
            op_output_dataset_path: Optional output path.
            fallback_sae_targets: Optional fallback SAEAnalysisTargets to use if this config doesn't have one.
        """
        if not self.output_store:
            # create the output store if needed
            self.output_store = AnalysisStore(cache_dir=cache_dir, op_output_dataset_path=op_output_dataset_path)
        elif cache_dir or op_output_dataset_path:
            rank_zero_warn(
                f"The provided cache_dir={cache_dir} and op_output_dataset_path={op_output_dataset_path} "
                "will be ignored in favor of the existing AnalysisStore configuration "
                f"(cache_dir={self.output_store.cache_dir}, "
                f" op_output_dataset_path={self.output_store.op_output_dataset_path})"
            )

        # Then configure for the module to set names_filter and hooks
        self.prepare_model_ctx(module, fallback_sae_targets)
        # return self


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


@dataclass(kw_only=True)
class AnalysisSetCfg(ITSerializableCfg):
    # Accept any sequence as input and convert to tuple in __post_init__
    # TODO: ultimately we want to support an arbitrary DAG of ops
    analysis_ops: Sequence[AnalysisOp] | None = None
    # currently allow executing a sequence of Ops (generating cfgs) or a sequence of AnalysisCfg objects if passed
    analysis_cfgs: Dict[AnalysisOp, AnalysisCfg] = field(default_factory=dict)

    def __post_init__(self):
        if not self.analysis_ops and not self.analysis_cfgs:
            raise MisconfigurationException("Either analysis_ops or analysis_cfgs must be provided")
        # Ensure analysis_ops is stored as a tuple if provided
        if self.analysis_ops is not None and not isinstance(self.analysis_ops, tuple):
            self.analysis_ops = tuple(self.analysis_ops)
        if self.analysis_ops:
            self.validate_analysis_order()

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
