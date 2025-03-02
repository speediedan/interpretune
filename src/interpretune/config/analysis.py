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
from interpretune.protocol import NamesFilter, AnalysisStoreProtocol, AnalysisBatchProtocol, SAEAnalysisProtocol
from interpretune.utils import DEFAULT_DECODE_KWARGS, MisconfigurationException

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

    def update(self, **kwargs):
        """Update multiple fields of the dataclass at once."""
        for key, value in kwargs.items():
            if key in self.__annotations__:
                setattr(self, key, value)

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

    def setup(self):
        # Use shared cache_dir for all ops; create op-specific output directory only
        op_output_path = self.op_output_dataset_path / self.op.name
        op_output_path.mkdir(exist_ok=True, parents=True)
        analysis_dirs = {"cache_dir": self.cache_dir, "op_output_dataset_path": op_output_path}
        # Create analysis store
        self.output_store = AnalysisStore(**analysis_dirs)

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
