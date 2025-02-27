from typing import Optional
from dataclasses import dataclass, field

from transformers import BatchEncoding, PreTrainedTokenizerBase

from interpretune.base.analysis import AnalysisStore, resolve_names_filter, _make_simple_cache_hook
from interpretune.base.contract.analysis import NamesFilter, AnalysisStoreProtocol, AnalysisBatchProtocol
from interpretune.base.ops import ANALYSIS_OPS, AnalysisOp
from interpretune.base.config.shared import ITSerializableCfg
from interpretune.utils.tokenization import DEFAULT_DECODE_KWARGS

# TODO: move update to a dataclass multi-field update mixin if keeping this a dataclass
@dataclass(kw_only=True)
class AnalysisCfg(ITSerializableCfg):
    output_store: AnalysisStoreProtocol = None  # usually constructed on setup()
    input_store: AnalysisStoreProtocol | None = None  # store containing input data from previous op
    op: str | AnalysisOp = field(default_factory=lambda: ANALYSIS_OPS['logit_diffs.base'])  # Renamed from mode
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
