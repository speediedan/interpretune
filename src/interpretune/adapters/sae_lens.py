import torch
from transformer_lens import ActivationCache
from typing import Any, TypeAlias
from collections.abc import Sequence, Callable
from dataclasses import dataclass, field
from functools import reduce, partial
from copy import deepcopy
from collections import defaultdict
from typing_extensions import override

from sae_lens.sae import SAE, SAEConfig
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from transformer_lens.hook_points import NamesFilter
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens.utils import get_device as tl_get_device

from interpretune.adapters.registration import CompositionRegistry
from interpretune.adapters.lightning import LightningDataModule, LightningModule, LightningAdapter
from interpretune.adapters.transformer_lens import ITLensConfig, BaseITLensModule, TLensAttributeMixin
from interpretune.base.config.shared import ITSerializableCfg, Adapter
from interpretune.base.components.core import CoreHelperAttributes
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import BaseITModule
from interpretune.utils.data_movement import move_data_to_device
from interpretune.utils.logging import rank_zero_warn, rank_zero_info
from interpretune.utils.patched_tlens_generate import generate as patched_generate
from interpretune.utils.exceptions import MisconfigurationException
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import IFrame, display
from interpretune.base.contract.analysis import AnalysisBatchProtocol, AnalysisStoreProtocol, ANALYSIS_OPS


################################################################################
# SAE Lens Configuration Encapsulation
################################################################################

@dataclass(kw_only=True)
class InstantiatedSAE:
    handle: SAE
    original_cfg: dict[str, Any] = field(default_factory=dict)
    sparsity: dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class SAELensFromPretrainedConfig(ITSerializableCfg):
    release: str
    sae_id: str
    device: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:  # align with TL default device resolution
            self.device = str(tl_get_device())

@dataclass(kw_only=True)
class SAELensCustomConfig(ITSerializableCfg):
    cfg: SAEConfig | dict[str, Any]
    # TODO: may add additional custom behavior handling attributes here
    def __post_init__(self) -> None:
        if not isinstance(self.cfg, SAEConfig):
            # TODO: add a PR to SAELens to allow SAEConfig to ref torch dtype and device objects instead of str repr
            # ensure the user provided a valid dtype (should be handled by SAEConfig ideally)
            # if self.cfg.get('dtype', None) and not isinstance(self.cfg['dtype'], torch.dtype):
            #     self.cfg['dtype'] = _resolve_torch_dtype(self.cfg['dtype'])
            self.cfg = SAEConfig.from_dict(self.cfg)

SAECfgType: TypeAlias = SAELensFromPretrainedConfig | SAELensCustomConfig

@dataclass(kw_only=True)
class SAELensConfig(ITLensConfig):
    sae_cfgs: SAECfgType | Sequence[SAECfgType]
    add_saes_on_init: bool = False  # TODO: may push this down to SAE config level instead of setting for all saes
    # use_error_term: bool = False  # TODO: add support for use_error_term with on_init stateful SAEs

    @property
    def normalized_sae_cfg_refs(self) -> list[str]:
        normalized_names = []
        for sae_cfg in self.sae_cfgs:
            if isinstance(sae_cfg, SAELensFromPretrainedConfig):
                normalized_names.append(sae_cfg.sae_id)
            elif isinstance(sae_cfg, SAELensCustomConfig):
                normalized_names.append(sae_cfg.cfg.hook_name)
        return normalized_names

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sae_cfgs:
            raise MisconfigurationException(
                "At least one `SAELensFromPretrainedConfig` or `SAELensCustomConfig` must be provided to "
                "initialize a HookedSAETransformer and use SAE Lens."
            )
        if isinstance(self.sae_cfgs, (SAELensFromPretrainedConfig, SAELensCustomConfig)):
            self.sae_cfgs = [self.sae_cfgs]
        self._sync_sl_tl_device_cfg()

    def _sync_sl_tl_device_cfg(self):
        tl_device = self.tl_cfg.cfg.device if hasattr(self.tl_cfg, 'cfg') else self.tl_cfg.device
        for sae_cfg in self.sae_cfgs:
            if hasattr(sae_cfg, 'cfg'):
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg.cfg, tl_device=str(tl_device))
            else:
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg, tl_device=str(tl_device))

    def _sync_sl_tl_default_device(self, sae_cfg_obj: SAECfgType, tl_device):
        if sae_cfg_obj.device and tl_device:
            rank_zero_warn(
                f"This SAEConfig's device type ('{sae_cfg_obj.device}') does not match the configured TL device "
                f"('{tl_device}'). Setting the device type for this SAE to match the specified TL device "
                f"('{tl_device}')."
            )
            setattr(sae_cfg_obj, 'device', tl_device)
        else:
            rank_zero_warn(
                "An SAEConfig device type was not provided. Setting the device type to match the currently specified "
                f"TL device type: '{tl_device}'."
            )
            setattr(sae_cfg_obj, 'device', tl_device)


################################################################################
# Mixins to support SAE Lens in different adapter contexts
################################################################################

class SAELensAttributeMixin(TLensAttributeMixin):
    @property
    def sae_cfgs(self) -> SAEConfig | None:
        try:
            # TODO: probably will need to add a separate sae_cfg property here as well that points to the configured
            #       SAEConfig
            cfg = reduce(getattr, "it_cfg.sae_cfgs".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a `SAEConfig` reference (has it been set yet?): {ae}")
            cfg = None
        return cfg

    @property
    def sae_handles(self) -> list[SAE]:
        return [sae.handle for sae in self.saes]


class BaseSAELensModule(BaseITLensModule):
    def __init__(self, *args, **kwargs):
        # using cooperative inheritance, so initialize attributes that may be required in base init methods
        self.saes: list[InstantiatedSAE] = []
        super().__init__(*args, **kwargs)
        HookedSAETransformer.generate = patched_generate

    def _convert_hf_to_tl(self) -> None:
        # if datamodule is not attached yet, attempt to retrieve tokenizer handle directly from provided it_cfg
        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer
        hf_preconversion_config = deepcopy(self.model.config)  # capture original hf config before conversion
        self.model = HookedSAETransformer.from_pretrained(hf_model=self.model, tokenizer=tokenizer_handle,
                                                          **self.it_cfg.tl_cfg.__dict__)
        self.model.config = hf_preconversion_config
        self.instantiate_saes()

    def instantiate_saes(self) -> None:
        for sae_cfg in self.it_cfg.sae_cfgs:
            assert isinstance(sae_cfg, (SAELensFromPretrainedConfig, SAELensCustomConfig))
            original_cfg, sparsity = None, None
            if isinstance(sae_cfg, SAELensFromPretrainedConfig):
                handle, original_cfg, sparsity = SAE.from_pretrained(**sae_cfg.__dict__)
            else:
                handle = SAE(cfg=sae_cfg.cfg)
            self.saes.append(added_sae := InstantiatedSAE(handle=handle, original_cfg=original_cfg, sparsity=sparsity))
            if self.it_cfg.add_saes_on_init:
                self.model.add_sae(added_sae.handle)

    def tl_config_model_init(self) -> None:
        self.model = HookedSAETransformer(tokenizer=self.it_cfg.tokenizer, **self.it_cfg.tl_cfg.__dict__)
        self.instantiate_saes()

    def _capture_hyperparameters(self) -> None:
        self._it_state._init_hparams = {"sae_cfgs": deepcopy(self.it_cfg.sae_cfgs)}
        super()._capture_hyperparameters()

    def set_input_require_grads(self) -> None:
        # not currently supported by ITLensModule
        rank_zero_info("Setting input require grads not currently supported by BASESAELensModule.")


################################################################################
# SAE Lens Module Composition
################################################################################

class SAELensAdapter(SAELensAttributeMixin):

    @classmethod
    @override
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.sae_lens),
            composition_classes=(ITDataModule,),
            description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.sae_lens),
            composition_classes=(ITDataModule, LightningDataModule),
            description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.sae_lens),
            composition_classes=(SAELensModule,),
            description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.core, Adapter.sae_lens),
            composition_classes=(SAELensConfig,),
            description="SAE Lens configuration that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.sae_lens),
            composition_classes=(
                SAELensAttributeMixin,
                BaseSAELensModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
            description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.lightning, Adapter.sae_lens),
            composition_classes=(SAELensConfig,),
            description="SAE Lens adapter that can be composed with core and l...",
        )

    def batch_to_device(self, batch) -> BatchEncoding:
        move_data_to_device(batch, self.input_device)
        return batch

class SAEAnalysisMixin:

    def construct_names_filter(
        self,
        target_layers: list[int],
        sae_hook_match_fn: Callable[[str, list[int] | None], bool]
    ) -> NamesFilter:
        available_hooks = {
            f'{handle.cfg.hook_name}.{key}' for handle in self.sae_handles
            for key in handle.hook_dict.keys()
        }
        names_filter = [
            hook for hook in available_hooks
            if sae_hook_match_fn(in_name=hook, layers=target_layers if target_layers else None)
        ]
        return names_filter

    def get_latents_and_indices(
            self, batch: dict[str, Any], batch_idx: int, analysis_batch: AnalysisBatchProtocol | None = None,
            cache: AnalysisStoreProtocol | None = None) -> tuple[torch.Tensor, dict[str, Any]] | None:

        # Determine answer_indices
        if getattr(analysis_batch, "answer_indices", None) is not None:
            answer_indices = analysis_batch.answer_indices
        elif self.analysis_cfg.answer_indices:
            answer_indices = self.analysis_cfg.answer_indices[batch_idx]
        else:
            answer_indices = self.resolve_answer_indices(batch)
        # Determine alive_latents
        if self.analysis_cfg.alive_latents:
            alive_latents = self.analysis_cfg.alive_latents[batch_idx]
        elif not cache:
            alive_latents = {}
        else:
            alive_latents = self.batch_alive_latents(answer_indices, cache, self.analysis_cfg.names_filter)
        if analysis_batch is not None:
            analysis_batch.update(answer_indices=answer_indices, alive_latents=alive_latents)
            return None
        else:
            return answer_indices, alive_latents

    def get_loss_preds_diffs(self, analysis_batch: AnalysisBatchProtocol,
                             answer_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor]:
        loss = self.loss_fn(answer_logits, analysis_batch.labels)
        answer_logits = self.standardize_logits(answer_logits)
        per_example_answers, _ = torch.max(answer_logits, dim=-2)
        preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
        logit_diffs = self.analysis_cfg.logit_diff_fn(answer_logits, target_indices=analysis_batch.orig_labels)
        return loss, logit_diffs, preds, answer_logits

    def run_with_ctx(
        self,
        analysis_batch: AnalysisBatchProtocol,
        batch: dict[str, Any],
        batch_idx: int,
        **kwargs: Any
    ) -> None:
        if self.analysis_cfg.op == ANALYSIS_OPS["clean_no_sae"]:
            answer_logits = self(**batch)
            answer_indices, alive_latents = self.get_latents_and_indices(batch, batch_idx)
            analysis_batch.update(answer_indices=answer_indices, alive_latents=alive_latents)
        elif self.analysis_cfg.op == ANALYSIS_OPS["clean_w_sae"]:
            answer_logits, cache = self.model.run_with_cache_with_saes(
                **batch, saes=self.sae_handles,
                names_filter=self.analysis_cfg.names_filter
            )
            answer_indices, alive_latents = self.get_latents_and_indices(batch, batch_idx, cache=cache)
            analysis_batch.update(cache=cache, answer_indices=answer_indices, alive_latents=alive_latents)
        elif self.analysis_cfg.op == ANALYSIS_OPS["ablation"]:
            assert self.analysis_cfg.alive_latents, "alive_latents required for ablation op"
            answer_indices, alive_latents = self.get_latents_and_indices(batch, batch_idx)
            per_latent_logits: dict[str, dict[Any, torch.Tensor]] = defaultdict(dict)
            for name, alive in alive_latents.items():
                for latent_idx in alive:
                    fwd_hooks_cfg = [(name, partial(self.analysis_cfg.ablate_latent_fn,
                                                    latent_idx=latent_idx,
                                                    seq_pos=answer_indices))]
                    answer_logits = self.model.run_with_hooks_with_saes(
                        **batch, saes=self.sae_handles,
                        clear_contexts=True, fwd_hooks=fwd_hooks_cfg
                    )
                    per_latent_logits[name][latent_idx] = answer_logits[
                        torch.arange(batch["input"].size(0)), answer_indices, :
                    ]
            analysis_batch.update(alive_latents=alive_latents, answer_indices=answer_indices)
            answer_logits = per_latent_logits
        elif self.analysis_cfg.op == ANALYSIS_OPS["attr_patching"]:
            assert all((self.analysis_cfg.fwd_hooks, self.analysis_cfg.bwd_hooks)), (
                "fwd_hooks and bwd_hooks required for attr_patching op"
            )
            answer_indices, _ = self.get_latents_and_indices(batch, batch_idx)
            with self.model.saes(saes=self.sae_handles):
                with self.model.hooks(
                    fwd_hooks=self.analysis_cfg.fwd_hooks, bwd_hooks=self.analysis_cfg.bwd_hooks
                ):
                    answer_logits = self.model(**batch)
                    answer_logits = torch.squeeze(
                        answer_logits[torch.arange(batch["input"].size(0)), answer_indices],
                        dim=1
                    )
                    loss, logit_diffs, preds, answer_logits = self.get_loss_preds_diffs(
                        analysis_batch, answer_logits
                    )
                    logit_diffs.sum().backward()
                    if logit_diffs.dim() == 0:
                        logit_diffs.unsqueeze_(0)
            analysis_batch.update(answer_indices=answer_indices, logit_diffs=logit_diffs,
                                  preds=preds, loss=loss)
        else:
            answer_logits = self.model.run_with_saes(**batch, saes=self.sae_handles)
        analysis_batch.update(answer_logits=answer_logits)

    def calc_ablation_effects(self, analysis_batch: AnalysisBatchProtocol, batch: dict[str, Any],
                              batch_idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self.analysis_cfg.base_logit_diffs, "base_logit_diffs required for ablation mode"
        if batch_idx >= len(self.analysis_cfg.base_logit_diffs):
            raise IndexError(f"Batch index {batch_idx} out of range for base_logit_diffs")
        attribution_values: dict[str, torch.Tensor] = {}
        per_latent = {"loss": defaultdict(dict), "logit_diffs": defaultdict(dict), "preds": defaultdict(dict),
                      "answer_logits": defaultdict(dict)}
        for act_name, logits in analysis_batch.answer_logits.items():
            attribution_values[act_name] = torch.zeros(batch["input"].size(0), self.sae_handles[0].cfg.d_sae)
            for latent_idx in analysis_batch.alive_latents[act_name]:
                loss_preds_diffs = self.get_loss_preds_diffs(analysis_batch, logits[latent_idx])
                for metric_name, value in zip(per_latent.keys(), loss_preds_diffs):
                    per_latent[metric_name][act_name][latent_idx] = value
                example_mask = (per_latent["logit_diffs"][act_name][latent_idx] > 0).cpu()
                per_latent["logit_diffs"][act_name][latent_idx] = (
                    per_latent["logit_diffs"][act_name][latent_idx][example_mask].detach().cpu()
                )
                base_diffs = torch.as_tensor(self.analysis_cfg.base_logit_diffs[batch_idx])
                for t in [example_mask, base_diffs]:
                    if t.dim() == 0:
                        t.unsqueeze_(0)
                base_diffs = base_diffs.cpu()
                attribution_values[act_name][example_mask, latent_idx] = (
                    base_diffs[example_mask] - per_latent["logit_diffs"][act_name][latent_idx]
                )
        per_latent_results = {key: per_latent[key]
                              for key in ["loss", "logit_diffs", "preds", "answer_logits"]}
        return per_latent_results, attribution_values

    def calc_attribution_values(
        self, analysis_batch: AnalysisBatchProtocol, batch: dict[str, Any], batch_idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        attribution_values: dict[str, torch.Tensor] = {}
        correct_activations: dict[str, torch.Tensor] = {}
        batch_cache_dict = ActivationCache(self.analysis_cfg.cache_dict, self.model)
        batch_sz = batch["input"].size(0)
        self.get_latents_and_indices(batch, batch_idx, cache=batch_cache_dict, analysis_batch=analysis_batch)
        for fwd_name in [name for name in batch_cache_dict if self.analysis_cfg.names_filter(name)]:
            attribution_values[fwd_name] = torch.zeros(batch_sz, self.sae_handles[0].cfg.d_sae)
            fwd_hook_acts = batch_cache_dict[fwd_name][torch.arange(batch_sz), analysis_batch.answer_indices]
            bwd_hook_grads = batch_cache_dict[f"{fwd_name}_grad"][torch.arange(batch_sz), analysis_batch.answer_indices]
            for t in [fwd_hook_acts, bwd_hook_grads]:
                if t.dim() == 2:
                    t.unsqueeze_(1)
            correct_activations[fwd_name] = torch.squeeze(fwd_hook_acts[(analysis_batch.logit_diffs > 0), :, :], dim=1)
            attribution_values[fwd_name][:, analysis_batch.alive_latents[fwd_name]] = torch.squeeze(
                (bwd_hook_grads[:, :, analysis_batch.alive_latents[fwd_name]] *
                 fwd_hook_acts[:, :, analysis_batch.alive_latents[fwd_name]]).cpu(), dim=1
            )
        return attribution_values, correct_activations

    def calc_clean_diffs(
        self, analysis_batch: AnalysisBatchProtocol, batch: dict[str, Any]
    ) -> None:
        logits, indices = analysis_batch.answer_logits, analysis_batch.answer_indices
        answer_logits = torch.squeeze(logits[torch.arange(batch["input"].size(0)), indices], dim=1)
        loss, logit_diffs, preds, answer_logits = self.get_loss_preds_diffs(analysis_batch, answer_logits)
        if logit_diffs.dim() == 0:
            logit_diffs.unsqueeze_(0)
        analysis_batch.update(loss=loss, logit_diffs=logit_diffs, preds=preds, answer_logits=answer_logits)
        if self.analysis_cfg.op == ANALYSIS_OPS['clean_w_sae']:
            correct_activations: dict[str, torch.Tensor] = {}
            logit_diffs = logit_diffs.cpu()
            for name in analysis_batch.cache.keys():
                if self.analysis_cfg.names_filter(name):
                    correct_activations[name] = analysis_batch.cache[name][logit_diffs > 0, indices[logit_diffs > 0], :]
            analysis_batch.update(correct_activations=correct_activations)

    def loss_and_logit_diffs(self, analysis_batch: AnalysisBatchProtocol, batch: dict[str, Any],
                             batch_idx: int) -> None:
        if self.analysis_cfg.op in [ANALYSIS_OPS["clean_w_sae"], ANALYSIS_OPS["clean_no_sae"]]:
            self.calc_clean_diffs(analysis_batch, batch)
        elif self.analysis_cfg.op == ANALYSIS_OPS["ablation"]:
            per_latent_results, attribution_values = self.calc_ablation_effects(analysis_batch, batch, batch_idx)
            analysis_batch.update(**per_latent_results, attribution_values=attribution_values)
        elif self.analysis_cfg.op == ANALYSIS_OPS["attr_patching"]:
            attribution_values, correct_activations = self.calc_attribution_values(analysis_batch, batch, batch_idx)
            analysis_batch.update(attribution_values=attribution_values, correct_activations=correct_activations)
        else:
            raise ValueError(f"Unsupported analysis op {self.analysis_cfg.op}")

    def resolve_answer_indices(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: allow selecting multiple answer indices for each example (analagous generation step where
        # we max over max_new_tokens)
        tokens = batch["input"].detach().cpu()
        if self.datamodule.tokenizer.padding_side == "left":
            return torch.full((tokens.size(0),), -1)
        nonpadding_mask = tokens != self.datamodule.tokenizer.pad_token_id
        # TODO: this heuristic could be more robust, test with a variety of datasets and padding strategies
        answer_indices = torch.where(nonpadding_mask, 1, 0).sum(dim=1) - 1
        return answer_indices

    @staticmethod
    def display_latent_dashboards(
            metrics: Any, title: str, sae_release: str,
            hook_to_sae_id: Callable[[str], str] = lambda hook: f"blocks.{hook.split('.')[1]}.hook_z",
            top_k: int = 1) -> None:
        """Print top positive and negative latent dashboards for all hooks in metrics."""
        analysis_dict = metrics.total_effect
        activation_counts = metrics.num_samples_active
        for hook_name, total_values in analysis_dict.items():
            print(f"\n{title} for {hook_name}:")
            directions = {"positive": total_values.topk(top_k),
                          "negative": total_values.topk(top_k, largest=False)}
            for direction, (values, indices) in directions.items():
                print(f"\n{direction}:")
                for value, idx in zip(values, indices):
                    effect_str = (f"#{idx} had total effect {value:.2f} and was active in "
                                  f"{activation_counts[hook_name][idx]} examples")
                    print(effect_str)
                    SAEAnalysisMixin.display_dashboard(
                        sae_release=sae_release,
                        sae_id=hook_to_sae_id(hook_name),
                        latent_idx=int(idx)
                    )
    @staticmethod
    def display_dashboard(sae_release: str = "gpt2-small-res-jb", sae_id: str = "blocks.9.hook_resid_pre",
                          latent_idx: int = 0, width: int = 800, height: int = 600) -> None:
        release = get_pretrained_saes_directory()[sae_release]
        neuronpedia_id = release.neuronpedia_id[sae_id]
        embed_cfg = "embed=true&embedexplanation=true&embedplots=true&embedtest=true"
        url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?{embed_cfg}&height=300"
        print(url)
        display(IFrame(url, width=width, height=height))

    def batch_alive_latents(self, answer_indices: torch.Tensor, cache: AnalysisStoreProtocol,
                            names_filter: Callable[[str], bool]) -> dict[str, Any]:
        filtered_acts = {name: acts for name, acts in cache.items() if names_filter(name)}
        alive_latents: dict[str, Any] = {}
        for name, acts in filtered_acts.items():
            alive = (acts[torch.arange(acts.size(0)), answer_indices, :] > 0).any(dim=0).nonzero().squeeze().tolist()
            if not isinstance(alive, list):
                alive = [alive]
            alive_latents[name] = alive
        return alive_latents

class SAELensModule(SAEAnalysisMixin, SAELensAdapter, CoreHelperAttributes, BaseSAELensModule):
    ...
