from typing import Callable, Optional
from dataclasses import dataclass, field

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from interpretune.base.analysis import (AnalysisStore, ablate_sae_latent, boolean_logits_to_avg_logit_diff,
                                        resolve_names_filter, _make_simple_cache_hook)
from interpretune.base.contract.analysis import (NamesFilter, AnalysisStoreProtocol, AnalysisBatchProtocol, AnalysisOp,
                                                 ANALYSIS_OPS, DEFAULT_DECODE_KWARGS)
from interpretune.base.config.shared import ITSerializableCfg

# TODO: move update to a dataclass multi-field update mixin if keeping this a dataclass
@dataclass(kw_only=True)
class AnalysisCfg(ITSerializableCfg):
    analysis_store: AnalysisStoreProtocol = None  # usually constructed on setup()
    op: str | AnalysisOp = field(default_factory=lambda: ANALYSIS_OPS['clean_no_sae'])  # Renamed from mode
    ablate_latent_fn: Callable = ablate_sae_latent
    logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff
    fwd_hooks: list[tuple] = field(default_factory=list)
    bwd_hooks: list[tuple] = field(default_factory=list)
    cache_dict: dict = field(default_factory=dict)
    names_filter: NamesFilter | None = None
    # TODO: after refactor, these three fields should probably be moved to a more general ref_artifacts field for
    #       artifacts of a previous analysis operation required for the current one
    base_logit_diffs: list = field(default_factory=list)
    alive_latents: list[dict] = field(default_factory=list)
    answer_indices: list[torch.Tensor] = field(default_factory=list)
    # Save configuration fields
    save_prompts: bool = False
    save_tokens: bool = False
    cache_dir: Optional[str] = None
    op_output_dataset_path: Optional[str] = None
    decode_kwargs: dict = field(default_factory=lambda: DEFAULT_DECODE_KWARGS)

    def __post_init__(self):
        # Convert string op to AnalysisOp
        if isinstance(self.op, str):
            self.op = ANALYSIS_OPS[self.op]

        self.names_filter = resolve_names_filter(self.names_filter)
        if not self.fwd_hooks and not self.bwd_hooks:
            self.fwd_hooks, self.bwd_hooks = self.check_add_default_hooks(
                op=self.op,
                names_filter=self.names_filter,
                cache_dict=self.cache_dict,
            )
        if self.logit_diff_fn is None:
            self.logit_diff_fn = boolean_logits_to_avg_logit_diff
        if self.ablate_latent_fn is None:
            self.ablate_latent_fn = ablate_sae_latent

    def update(self, **kwargs):
        """Update multiple fields of the dataclass at once."""
        for key, value in kwargs.items():
            if key in self.__annotations__:
                setattr(self, key, value)

    def reset_analysis_store(self) -> None:
        # TODO: May be able to refactor this out if using per-epoch op dataset subsplits
        # Prepare a new cache for the next epoch preserving save_cfg (for multi-epoch AnalysisSessionCfg instances)
        current_analysis_cls = self.analysis_store.__class__
        current_save_cfg_cls = self.analysis_store.save_cfg.__class__
        current_save_cfg = {k: v for k, v in self.analysis_store.save_cfg.__dict__.items() if k != 'analysis_store'}
        self.analysis_store = current_analysis_cls(save_cfg=current_save_cfg_cls(**current_save_cfg))
        assert id(self.analysis_store) == id(self.analysis_store.save_cfg.analysis_store)
        # TODO: maybe use this approach instead of above
        #"""Reset analysis store, preserving configuration."""
        #self.analysis_store.reset()

    def setup(self):
        # Use shared cache_dir for all ops; create op-specific output directory only
        op_output_path = self.op_output_dataset_path / self.op.name
        op_output_path.mkdir(exist_ok=True, parents=True)
        analysis_dirs = {"cache_dir": self.cache_dir, "op_output_dataset_path": op_output_path}
        # Create analysis store
        self.analysis_store = AnalysisStore(**analysis_dirs)

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

        if op.name == 'clean_no_sae':
            return fwd_hooks, bwd_hooks
        if names_filter is None:
            raise ValueError("names_filter required for non-clean operations")
        if op.name == 'attr_patching':
            fwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict))
            )
            bwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict, is_backward=True))
            )
        return fwd_hooks, bwd_hooks
