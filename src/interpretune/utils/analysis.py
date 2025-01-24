from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformer_lens import ActivationCache
from jaxtyping import Float
from typing_extensions import Literal
from transformer_lens.hook_points import HookPoint
from IPython.display import IFrame, display
from interpretune.utils.import_utils import _SL_AVAILABLE

if _SL_AVAILABLE:
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
else:
    def get_pretrained_saes_directory():
        raise NotImplementedError("sae_lens not available")


DEFAULT_DECODE_KWARGS = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}


@dataclass(kw_only=True)
class SaveCfg:
    #out_summ: Optional[AnalysisCache] = None
    analysis_cache: Optional["AnalysisCache"] = None
    #summ_map: Dict = field(default_factory=dict)
    prompts: bool = False
    tokens: bool = False
    caches: bool = False
    grad_caches: bool = False
    decode_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        default_decode_kwargs = DEFAULT_DECODE_KWARGS.copy()
        for k, v in default_decode_kwargs.items():
            self.decode_kwargs.setdefault(k, v)

    def wrap_summary(self, step_summ: Dict, batch: BatchEncoding, cache: Optional[ActivationCache] = None,
        grad_cache: Optional[ActivationCache] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        if self.prompts:
            assert tokenizer is not None, "Tokenizer is required to decode prompts"
            step_summ["prompts"] = tokenizer.batch_decode(batch['input'], **self.decode_kwargs)
        if self.tokens:
            step_summ["tokens"] = batch['input'].detach().cpu()
        if self.caches:
            step_summ["caches"] = cache
        if self.grad_caches:
            step_summ["grad_caches"] = grad_cache
        if self.analysis_cache:
            for key, val in step_summ.items():
                getattr(self.analysis_cache, key).append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)
        else:
            return step_summ


# TODO: maybe keep as tensors via cat for future analysis rather than separate batches
@dataclass(kw_only=True)
class AnalysisCache:
    logit_diffs: list[torch.Tensor] = field(default_factory=list)
    answer_logits: list[torch.Tensor] = field(default_factory=list)
    loss: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    orig_labels: list[torch.Tensor] = field(default_factory=list)
    preds: list[torch.Tensor] = field(default_factory=list)
    caches: list[ActivationCache] = field(default_factory=list)
    grad_caches: list[ActivationCache] = field(default_factory=list)
    alive_latents: list[int] = field(default_factory=list)
    answer_indices: list[torch.Tensor] = field(default_factory=list)
    correct_activations: list[torch.Tensor] = field(default_factory=list)
    ablation_effects: list[torch.Tensor] = field(default_factory=list)
    attribution_values: list[torch.Tensor] = field(default_factory=list)
    tokens: list[torch.Tensor] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    save_cfg: SaveCfg = field(default_factory=SaveCfg)

    def __post_init__(self):
        self.save_cfg.analysis_cache = self

    def save(self, step_summ: Dict, batch: BatchEncoding, cache: Optional[ActivationCache] = None,
             grad_cache: Optional[ActivationCache] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        wrap_map = dict(step_summ=step_summ, batch=batch, cache=cache, grad_cache=grad_cache, tokenizer=tokenizer)
        self.save_cfg.wrap_summary(**wrap_map)

# def save_summary(self, batch: BatchEncoding,
#                     save_cfg: SaveCfg = SaveCfg(),
#                        cache: Optional[ActivationCache] = None,
#                        grad_cache: Optional[ActivationCache] = None) -> None:
#     if save_cfg.prompts:
#         save_cfg.summ_map["prompts"] = self.datamodule.tokenizer.batch_decode(batch['input'], **DEFAULT_DECODE_KWARGS)
#     if save_cfg.tokens:
#         save_cfg.summ_map["tokens"] = batch['input'].detach().cpu()
#     if save_cfg.caches:
#         save_cfg.summ_map["caches"] = cache
#     if save_cfg.grad_caches:
#         save_cfg.summ_map["grad_caches"] = grad_cache
#     if save_cfg.out_summ:
#         for key, val in save_cfg.summ_map.items():
#             getattr(save_cfg.out_summ, key).append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)
#     else:
#         return save_cfg.summ_map

def boolean_logits_to_avg_logit_diff(
    logits: Float[torch.Tensor, "batch seq 2"],
    target_indices: torch.Tensor,
    reduction: Literal["mean", "sum"] | None = "mean",
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

def resolve_answer_indices(tokens):
    nonpadding_mask = tokens != 50256
    answer_indices = torch.where(nonpadding_mask, 1, 0).sum(dim=1) - 1
    return answer_indices

def batch_alive_latents(answer_indices, cache, hook_names):
    acts = cache[hook_names]
    alive_latents = (acts[torch.arange(acts.size(0)), answer_indices, :] > 0).any(dim=0).nonzero().squeeze().tolist()
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
