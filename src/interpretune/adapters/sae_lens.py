from __future__ import annotations
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from copy import deepcopy
from typing_extensions import override

from IPython.display import IFrame, display
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.sae import SAE, SAEConfig
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from transformer_lens.hook_points import NamesFilter
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.adapters import (CompositionRegistry, LightningDataModule, LightningModule, LightningAdapter,
                                   BaseITLensModule, TLensAttributeMixin)
from interpretune.base import CoreHelperAttributes, ITDataModule, BaseITModule
from interpretune.config import SAELensFromPretrainedConfig, SAELensCustomConfig, SAELensConfig
from interpretune.utils import move_data_to_device, rank_zero_warn, rank_zero_info, patched_generate
from interpretune.protocol import Adapter


@dataclass(kw_only=True)
class InstantiatedSAE:
    handle: SAE
    original_cfg: dict[str, Any] = field(default_factory=dict)
    sparsity: dict[str, Any] = field(default_factory=dict)

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
        pruned_cfg = self._prune_tl_cfg_dict()  # avoid edge case where conflicting keys haven't already been pruned
        self.model = HookedSAETransformer.from_pretrained(hf_model=self.model, tokenizer=tokenizer_handle, **pruned_cfg)
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

class SAELensModule(SAEAnalysisMixin, SAELensAdapter, CoreHelperAttributes, BaseSAELensModule):
    ...
