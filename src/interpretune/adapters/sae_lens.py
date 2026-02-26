from __future__ import annotations
from typing import Any, cast, TYPE_CHECKING
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from copy import deepcopy

from IPython.display import IFrame, display
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.saes.sae import SAE, SAEConfig
from sae_lens.saes.standard_sae import StandardSAE
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from transformer_lens.hook_points import NamesFilter
from transformers import PreTrainedModel

from interpretune.adapters import (
    CompositionRegistry,
    LightningDataModule,
    LightningModule,
    LightningAdapter,
    BaseITLensModule,
    TLensAttributeMixin,
    NNsightAttributeMixin,
    BaseNNsightModule,
)
from interpretune.base import CoreHelperAttributes, ITDataModule, BaseITModule
from interpretune.config import SAELensFromPretrainedConfig, SAELensCustomConfig, SAELensConfig
from interpretune.utils import rank_zero_warn, rank_zero_info
from interpretune.protocol import Adapter

if TYPE_CHECKING:
    from interpretune.analysis.backends import ModelBackend


@dataclass(kw_only=True)
class InstantiatedSAE:
    handle: SAE
    original_cfg: dict[str, Any] = field(default_factory=dict)
    sparsity: dict[str, Any] = field(default_factory=dict)


################################################################################
# Mixins to support SAE Lens in different adapter contexts
################################################################################


class SAELensAttributeMixin:
    """Backend-agnostic mixin for SAE Lens attribute access.

    Provides consistent attribute access patterns for SAE-related state.
    """

    @property
    def sae_cfgs(self) -> SAEConfig | None:
        try:
            # TODO: probably will need to add a separate sae_cfg property here as well that points to the configured
            #       SAEConfig
            cfg = reduce(getattr, "it_cfg.sae_cfgs".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a `SAEConfig` reference (has it been set yet?): {ae}")
            cfg = None
        return cfg  # type: ignore[return-value]

    @property
    def sae_handles(self) -> list[SAE]:
        return [sae.handle for sae in self.saes]  # type: ignore[attr-defined]  # provided by mixing class


class BaseSAELensModule(BaseITModule):
    """Backend-agnostic base module for SAE Lens.

    Provides SAE/transcoder loading and state management that works with both TransformerLens and NNsight backends, e.g.
    when using the TransformerLens backend, compose with BaseITLensModule via the adapter composition system.
    """

    def __init__(self, *args, **kwargs):
        # using cooperative inheritance, so initialize attributes that may be required in base init methods
        self.saes: list[InstantiatedSAE] = []
        self._model_backend: ModelBackend | None = getattr(self, "_model_backend", None)
        super().__init__(*args, **kwargs)

    @property
    def model_backend(self) -> ModelBackend:
        assert self._model_backend is not None, "model_backend not set — ensure a backend mixin is composed"
        return self._model_backend

    def instantiate_saes(self) -> None:
        for sae_cfg in self.it_cfg.sae_cfgs:
            assert isinstance(sae_cfg, (SAELensFromPretrainedConfig, SAELensCustomConfig))
            original_cfg, sparsity = None, None
            if isinstance(sae_cfg, SAELensFromPretrainedConfig):
                handle, original_cfg, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(**sae_cfg.__dict__)
            else:
                # TODO: enable configuration of SAE subclass to use
                handle = StandardSAE(cfg=sae_cfg.cfg)  # type: ignore[arg-type]
                original_cfg = original_cfg or {}
                sparsity = sparsity or {}
            self.saes.append(added_sae := InstantiatedSAE(handle=handle, original_cfg=original_cfg, sparsity=sparsity))  # type: ignore[arg-type]
            if self.it_cfg.add_saes_on_init and hasattr(self.model, "add_sae"):
                self.model.add_sae(added_sae.handle)  # type: ignore[operator]

    def _capture_hyperparameters(self) -> None:
        self._it_state._init_hparams = {"sae_cfgs": deepcopy(self.it_cfg.sae_cfgs)}
        super()._capture_hyperparameters()

    def set_input_require_grads(self) -> None:
        rank_zero_info("Setting input require grads not currently supported by BaseSAELensModule.")


class SAELensTLModuleMixin(TLensAttributeMixin):
    """Mixin that provides TransformerLens-specific functionality for SAE Lens.

    This mixin is composed with BaseSAELensModule and BaseITLensModule when using the TransformerLens backend to
    provide _convert_hf_to_tl, tl_config_model_init, and TLModelBackend initialization.

    Supports model_wrapper selection via ``it_cfg.model_wrapper``:
    - ``"hooked_transformer"`` → ``HookedSAETransformer.from_pretrained()`` (default)
    - ``"transformer_bridge"`` → ``SAETransformerBridge`` (not yet implemented)
    """

    def __init__(self, *args, **kwargs):
        from interpretune.analysis.backends.transformer_lens import TLModelBackend

        self._model_backend = TLModelBackend()
        super().__init__(*args, **kwargs)

    def _convert_hf_to_tl(self) -> None:
        """Convert HF model to HookedSAETransformer (or SAETransformerBridge when available)."""
        model_wrapper = getattr(self.it_cfg, "model_wrapper", "hooked_transformer")
        if model_wrapper == "transformer_bridge":
            raise NotImplementedError(
                "SAETransformerBridge is not yet implemented. Use model_wrapper='hooked_transformer' "
                "(or omit it for the default) until TransformerBridge SAE support is available."
            )
        # HookedSAETransformer path (existing behavior)
        # if datamodule is not attached yet, attempt to retrieve tokenizer handle directly from provided it_cfg
        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer  # type: ignore[attr-defined]
        hf_preconversion_config = deepcopy(self.model.config)  # type: ignore[attr-defined]  # capture original hf config before conversion
        pruned_cfg = self._prune_tl_cfg_dict()  # type: ignore[attr-defined]  # from BaseITLensModule
        self.model = HookedSAETransformer.from_pretrained(
            hf_model=cast(PreTrainedModel, self.model), tokenizer=tokenizer_handle, **pruned_cfg
        )
        self.model.config = hf_preconversion_config
        self.instantiate_saes()  # type: ignore[attr-defined]  # from BaseSAELensModule

    def tl_config_model_init(self) -> None:
        """Initialize model from TL config (custom/non-pretrained path)."""
        # Filter out IT-specific keys (e.g., 'use_bridge') that HookedSAETransformer doesn't accept
        pruned_cfg = self._prune_tl_cfg_dict()  # type: ignore[attr-defined]  # from BaseITLensModule
        self.model = HookedSAETransformer(tokenizer=self.it_cfg.tokenizer, **pruned_cfg)
        self.instantiate_saes()  # type: ignore[attr-defined]  # from BaseSAELensModule


class SAELensNNsightModuleMixin(NNsightAttributeMixin):
    """Mixin that provides NNsight-specific functionality for SAE Lens.

    This mixin is composed with BaseSAELensModule and BaseNNsightModule when using the NNsight backend to handle NNsight
    LanguageModel initialization and SAE loading.
    """

    def auto_model_init(self) -> None:
        """Initialize model using NNsight backend for SAE Lens.

        This overrides BaseNNsightModule's auto_model_init to additionally load SAEs after the NNsight LanguageModel is
        initialized.
        """
        self._init_nnsight_for_sae_lens()

    def _init_nnsight_for_sae_lens(self) -> None:
        """Initialize NNsight LanguageModel and load SAEs.

        This method handles NNsight-specific initialization logic, then loads SAEs using sae_lens (pure PyTorch).
        Also initializes the ``NNsightModelBackend`` with a ``HookNameResolver`` for the model's architecture.
        """
        import os

        from nnsight import LanguageModel

        from interpretune.analysis.backends.hook_mapping import HookNameResolver
        from interpretune.analysis.backends.nnsight import NNsightModelBackend
        from interpretune.utils import rank_zero_debug

        nnsight_cfg = self.nnsight_cfg  # type: ignore[attr-defined]  # from NNsightAttributeMixin
        if nnsight_cfg is None:
            raise ValueError("nnsight_cfg must be set for NNsight SAE Lens model initialization")

        model_name = nnsight_cfg.model_name or self.it_cfg.model_name_or_path  # type: ignore[attr-defined]
        if model_name is None:
            raise ValueError("model_name must be specified in nnsight_cfg or model_name_or_path in it_cfg")

        rank_zero_info(f"Initializing NNsight LanguageModel for SAE Lens with model: {model_name}")

        nnsight_kwargs = nnsight_cfg.get_nnsight_kwargs()

        # Handle authentication token
        if self.it_cfg.os_env_model_auth_key:  # type: ignore[attr-defined]
            access_token = os.environ.get(self.it_cfg.os_env_model_auth_key.upper())  # type: ignore[attr-defined]
            if access_token:
                nnsight_kwargs["token"] = access_token
                rank_zero_debug("Using authentication token from environment for NNsight model")

        # Handle NDIF API key for remote execution
        if nnsight_cfg.remote:
            api_key = nnsight_cfg.api_key or os.environ.get("NDIF_API_KEY")
            if api_key:
                nnsight_kwargs["api_key"] = api_key

        # Initialize NNsight LanguageModel
        self.model = LanguageModel(model_name, **nnsight_kwargs)

        # Store tokenizer reference
        if self.it_cfg.tokenizer is None and hasattr(self.model, "tokenizer"):  # type: ignore[attr-defined]
            self.it_cfg.tokenizer = self.model.tokenizer  # type: ignore[attr-defined]

        rank_zero_info("NNsight LanguageModel initialized for SAE Lens")

        # Load SAEs (pure PyTorch, backend-agnostic)
        self.instantiate_saes()  # type: ignore[attr-defined]  # from BaseSAELensModule

        # Initialize NNsight model backend with hook resolver
        hf_model = NNsightModelBackend._get_hf_model(self.model)
        hf_config = getattr(hf_model, "config", None)
        architectures = getattr(hf_config, "architectures", None) if hf_config else None
        if architectures:
            model_arch = architectures[0]
        else:
            # Fallback: use the class name of the underlying HF model
            model_arch = type(hf_model).__name__

        resolver = HookNameResolver(model_arch)
        self._model_backend = NNsightModelBackend(resolver)  # type: ignore[attr-defined]
        rank_zero_info(f"NNsight model backend initialized with architecture: {model_arch}")

    def _capture_hyperparameters(self) -> None:
        """Capture hyperparameters for NNsight SAE Lens module.

        This overrides BaseNNsightModule's _capture_hyperparameters to avoid requiring nnsight_cfg on the base module.
        SAE Lens NNsight modules use SAELensConfig which manages nnsight_cfg.
        """
        self._it_state._init_hparams = {"sae_cfgs": deepcopy(self.it_cfg.sae_cfgs)}  # type: ignore[attr-defined]
        if self.it_cfg.nnsight_cfg is not None:  # type: ignore[attr-defined]
            self._it_state._init_hparams.update({"nnsight_cfg": deepcopy(self.it_cfg.nnsight_cfg)})  # type: ignore[attr-defined]
        # Skip BaseNNsightModule._capture_hyperparameters and call BaseITModule's version
        BaseITModule._capture_hyperparameters(self)  # type: ignore[arg-type]


################################################################################
# SAE Lens Module Composition
################################################################################


class SAELensAdapter(SAELensAttributeMixin):
    @property
    def has_sae_analysis_cfg(self) -> bool:
        """Return True if this object has an analysis_run_cfg and implements construct_names_filter.

        As in other places, we favor hasattr structural check instead of runtime protocol for LatentAnalysisProtocol.
        """
        return (
            hasattr(self, "analysis_run_cfg")
            and bool(getattr(self, "analysis_run_cfg", None))
            and hasattr(self, "construct_names_filter")
        )

    def _make_setup_dirs(self) -> None:
        # Call the parent implementation to perform core directory and hook setup.
        # BaseITModule is always present in the composition hierarchy for SAE-enabled modules.
        # TODO: standardize and document how we handle these static type checker challenges. We currently use a mix of,
        #       assertions, type checker directives and casting, e.g. cast(BaseITModule, self)._make_setup_dirs()
        super()._make_setup_dirs()  # type: ignore[attr-defined]

        # SAE-specific analysis initialization: only run when the light-weight structural checks pass
        if self.has_sae_analysis_cfg:
            # local import to avoid import cycles when adapters are imported early
            from interpretune.protocol import LatentAnalysisProtocol, AnalysisRunnerProtocol
            from interpretune.config import init_analysis_cfgs

            # Cast self to AnalysisRunnerProtocol to satisfy typing and access analysis_run_cfg
            runner_holder = cast(AnalysisRunnerProtocol, self)
            analysis_run_cfg = runner_holder.analysis_run_cfg

            init_analysis_cfgs(
                module=cast(LatentAnalysisProtocol, self),  # type: ignore[arg-type]  # protocol compatibility
                analysis_cfgs=analysis_run_cfg._processed_analysis_cfgs,
                cache_dir=analysis_run_cfg.cache_dir,
                op_output_dataset_path=analysis_run_cfg.op_output_dataset_path,
                latent_analysis_targets=analysis_run_cfg.latent_analysis_targets,
                ignore_manual=analysis_run_cfg.ignore_manual,
            )

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        # ======================================================================
        # Backend-agnostic registrations: (core, sae_lens) — defaults to TL backend
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule,),
            description="SAE Lens adapter (default TL backend) that can be composed with core...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensTLModule,),
            description="SAE Lens adapter (default TL backend) that can be composed with core...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.core, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensConfig,),
            description="SAE Lens configuration that can be composed with core...",
        )

        # ======================================================================
        # TransformerLens backend registrations: (core, transformer_lens, sae_lens)
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.transformer_lens, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule,),
            description="SAE Lens adapter with explicit TransformerLens backend...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.transformer_lens, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensTLModule,),
            description="SAE Lens adapter with explicit TransformerLens backend...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.core, Adapter.transformer_lens, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensConfig,),
            description="SAE Lens configuration with explicit TransformerLens backend...",
        )

        # ======================================================================
        # Lightning + default TL backend registrations: (lightning, sae_lens)
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule, LightningDataModule),
            description="SAE Lens adapter (default TL backend) with Lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(
                SAELensTLModuleMixin,
                SAELensAttributeMixin,
                BaseSAELensModule,
                BaseITLensModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
            description="SAE Lens adapter (default TL backend) with Lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.lightning, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensConfig,),
            description="SAE Lens configuration with Lightning...",
        )

        # ======================================================================
        # Lightning + explicit TL backend registrations: (lightning, transformer_lens, sae_lens)
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.transformer_lens, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule, LightningDataModule),
            description="SAE Lens adapter with explicit TransformerLens backend and Lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.transformer_lens, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(
                SAELensTLModuleMixin,
                SAELensAttributeMixin,
                BaseSAELensModule,
                BaseITLensModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
            description="SAE Lens adapter with explicit TransformerLens backend and Lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.lightning, Adapter.transformer_lens, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensConfig,),
            description="SAE Lens configuration with explicit TransformerLens backend and Lightning...",
        )

        # ======================================================================
        # NNsight backend registrations: (core, nnsight, sae_lens)
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.nnsight, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule,),
            description="SAE Lens adapter with NNsight backend...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.nnsight, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensNNsightModule,),
            description="SAE Lens adapter with NNsight backend...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.core, Adapter.nnsight, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensConfig,),
            description="SAE Lens configuration with NNsight backend...",
        )

        # ======================================================================
        # Lightning + NNsight backend registrations: (lightning, nnsight, sae_lens)
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.nnsight, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule, LightningDataModule),
            description="SAE Lens adapter with NNsight backend and Lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.nnsight, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(
                SAELensNNsightModuleMixin,
                SAELensAttributeMixin,
                BaseSAELensModule,
                BaseNNsightModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
            description="SAE Lens adapter with NNsight backend and Lightning...",
        )
        adapter_ctx_registry.register(
            Adapter.sae_lens,
            component_key="module_cfg",
            adapter_combination=(Adapter.lightning, Adapter.nnsight, Adapter.sae_lens),  # type: ignore[arg-type]
            composition_classes=(SAELensConfig,),
            description="SAE Lens configuration with NNsight backend and Lightning...",
        )


class SAELensAnalysisMixin:
    def construct_names_filter(
        self, target_layers: int | list[int] | None, sae_hook_match_fn: Callable[[str, int | list[int] | None], bool]
    ) -> NamesFilter:
        available_hooks = {
            f"{handle.cfg.metadata.hook_name}.{key}"
            for handle in self.sae_handles  # type: ignore[attr-defined]  # provided by mixing class
            for key in handle.hook_dict.keys()
        }
        layers_arg = [target_layers] if isinstance(target_layers, int) else target_layers
        names_filter = [hook for hook in available_hooks if sae_hook_match_fn(hook, layers_arg)]
        return names_filter

    @staticmethod
    def display_latent_dashboards(
        metrics: Any,
        title: str,
        sae_release: str,
        hook_to_sae_id: Callable[[str], str] = lambda hook: f"blocks.{hook.split('.')[1]}.hook_z",
        top_k: int = 1,
    ) -> None:
        """Print top positive and negative latent dashboards for all hooks in metrics."""
        analysis_dict = metrics.total_effect
        activation_counts = metrics.num_samples_active
        for hook_name, total_values in analysis_dict.items():
            print(f"\n{title} for {hook_name}:")
            directions = {"positive": total_values.topk(top_k), "negative": total_values.topk(top_k, largest=False)}
            for direction, (values, indices) in directions.items():
                print(f"\n{direction}:")
                for value, idx in zip(values, indices):
                    effect_str = (
                        f"#{idx} had total effect {value:.2f} and was active in "
                        f"{activation_counts[hook_name][idx]} examples"
                    )
                    print(effect_str)
                    SAELensAnalysisMixin.display_dashboard(
                        sae_release=sae_release, sae_id=hook_to_sae_id(hook_name), latent_idx=int(idx)
                    )

    @staticmethod
    def display_dashboard(
        sae_release: str = "gpt2-small-res-jb",
        sae_id: str = "blocks.9.hook_resid_pre",
        latent_idx: int = 0,
        width: int = 800,
        height: int = 600,
    ) -> None:
        release = get_pretrained_saes_directory()[sae_release]
        neuronpedia_id = release.neuronpedia_id[sae_id]
        embed_cfg = "embed=true&embedexplanation=true&embedplots=true&embedtest=true"
        url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?{embed_cfg}&height=300"
        print(url)
        display(IFrame(url, width=width, height=height))


class SAELensTLModule(
    SAELensTLModuleMixin,
    SAELensAnalysisMixin,
    SAELensAdapter,
    CoreHelperAttributes,
    BaseSAELensModule,
    BaseITLensModule,
): ...


class SAELensNNsightModule(
    SAELensNNsightModuleMixin,
    SAELensAnalysisMixin,
    SAELensAdapter,
    CoreHelperAttributes,
    BaseSAELensModule,
    BaseNNsightModule,
): ...
