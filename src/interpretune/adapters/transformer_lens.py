import os
import re
from typing import Type, cast, Any, Mapping
import inspect
from functools import reduce, partial
from copy import deepcopy

import torch
from transformers import PretrainedConfig as HFPretrainedConfig, PreTrainedModel
from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig, TransformerBridgeConfig
from transformer_lens.utilities.multi_gpu import get_best_available_device
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
    map_default_transformer_lens_config,
)
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.adapters import CompositionRegistry, LightningDataModule, LightningModule, LightningAdapter
from interpretune.adapters.model_view import ModelView, CanonicalModelView
from interpretune.base import CoreHelperAttributes, ITDataModule, BaseITModule
from interpretune.base.components.mixins import _import_class
from interpretune.utils import move_data_to_device, rank_zero_warn, rank_zero_info, rank_zero_debug, _FTS_AVAILABLE
from interpretune.utils.import_utils import _resolve_dtype
from interpretune.protocol import Adapter


################################################################################
# Mixins to support TransformerLens in different adapter contexts
################################################################################


class TLensAttributeMixin:
    @property
    def tl_cfg(self) -> HookedTransformerConfig | TransformerBridgeConfig | None:
        try:
            cfg = reduce(getattr, "model.cfg".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a TransformerLens config reference (has it been set yet?): {ae}")
            cfg = None
        return cfg  # type: ignore[return-value]

    # TODO: we aren't using IT's Property Composition feature for TLens yet, but might be worth enabling it
    @property
    def device(self) -> torch.device | None:
        device: torch.device | None = None
        try:
            device = (
                getattr(self._it_state, "_device", None)  # type: ignore[attr-defined]  # provided by mixing class
                or getattr(self.tl_cfg, "device", None)
                or reduce(getattr, "model.device".split("."), self)
            )
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a device reference (has it been set yet?): {ae}")
            device = None
        return device

    @device.setter
    def device(self, value: str | torch.device | None) -> None:
        if value is not None and not isinstance(value, torch.device):
            value = torch.device(value)
        self._it_state._device = value  # type: ignore[attr-defined]  # provided by mixing class

    def get_tl_device(self) -> torch.device | None:
        """Get the best available device based on TransformerLens config."""
        try:
            if self.tl_cfg is None:
                return None
            # get_best_available_device works with both HookedTransformerConfig and TransformerBridgeConfig
            # at runtime, though type signature only declares HookedTransformerConfig
            device = get_best_available_device(self.tl_cfg)  # type: ignore[arg-type]
        except (AttributeError, AssertionError) as ae:
            rank_zero_warn(f"Problem determining appropriate device from TransformerLens config. Received: {ae}")
            device = None
        return device

    @property
    def output_device(self) -> torch.device | None:
        return self.get_tl_device()  # type: ignore[attr-defined]  # provided by mixing class

    @property
    def input_device(self) -> torch.device | None:
        return self.get_tl_device()

    def batch_to_device(self, batch) -> BatchEncoding:
        """Move a batch to the TL input device if one is available.

        Implemented on the TL mixin so that any composition that exposes TL properties
        (e.g., `input_device`) can reuse consistent behavior and satisfy static typing.
        """
        device = self.input_device
        if device is not None:
            move_data_to_device(batch, device)
        return batch


class BaseITLensModule(BaseITModule):
    """Base module for TransformerLens integration.

    Supports both:
    - TransformerBridge (v3, default): Wraps HF models without weight conversion, more memory efficient
    - HookedTransformer (legacy): Converts HF model weights, provides traditional TL interface

    Set `use_bridge=False` in tl_cfg to use legacy HookedTransformer path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = None

    def auto_model_init(self) -> None:
        """Can be overridden by subclasses to automatically initialize model from a configuration (e.g.
        hf_from_pretrained_cfg, tl_from_config etc.).

        Supports both TransformerBridge and HookedTransformer.
        """
        if self.it_cfg.hf_from_pretrained_cfg:
            self.hf_pretrained_model_init()
        else:
            self.tl_config_model_init()

    def hf_pretrained_model_init(self) -> None:
        # for TL, only a subset of the HF pretrained init flow used since the model is replaced with
        # HookedTransformer or TransformerBridge
        access_token = (
            os.environ[self.it_cfg.os_env_model_auth_key.upper()] if self.it_cfg.os_env_model_auth_key else None
        )
        quantization_config = super()._hf_configure_quantization()
        super()._update_hf_pretrained_cfg(quantization_config)
        cust_config, _ = super()._hf_gen_cust_config(access_token)
        self.model = self.hf_configured_model_init(cust_config, access_token)

        # Choose between TransformerBridge (v3, default) and legacy HookedTransformer
        if self.it_cfg.tl_cfg.use_bridge:
            self._convert_hf_to_bridge()
        else:
            self._convert_hf_to_tl()

    def hf_configured_model_init(
        self, cust_config: HFPretrainedConfig, access_token: str | None = None
    ) -> torch.nn.Module:
        # usually makes sense to init the HookedTransfomer (empty) and pretrained HF model weights on cpu
        # versus moving them both to GPU (may make sense to explore meta device usage for model definition
        # in the future, only materializing parameter by parameter during loading from pretrained weights
        # to eliminate need for two copies in memory)
        # TODO: add warning that TransformerLens only specifying a single device via device is supported
        #       (though the model will automatically be moved to multiple devices if n_devices > 1)
        cust_config.num_labels = self.it_cfg.num_labels
        assert self.it_cfg.hf_from_pretrained_cfg is not None, (
            "cannot init hf_configured_model if hf_from_pretrained_cfg is None"
        )
        assert self.it_cfg.model_class is not None, "cannot init hf_configured_model if model_class is None"
        if (dmap := self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs.get("device_map", None)) != "cpu":
            rank_zero_warn(
                "Overriding `device_map` passed to TransformerLens to transform pretrained weights on"
                f" cpu prior to moving the model to target device: {dmap}"
            )
            self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs["device_map"] = "cpu"
        # We use nominal typing against HF's PreTrainedModel to satisfy static checkers without a local
        # Protocol definition. We should switch to structural typing if/when HF offers a protocol definition.
        assert inspect.isclass(self.it_cfg.model_class) and issubclass(self.it_cfg.model_class, PreTrainedModel), (
            "model_class must be a PreTrainedModel subclass"
        )

        model_cls = cast(Type[PreTrainedModel], self.it_cfg.model_class)
        restore_async_load_env = False
        if os.name == "nt" and "HF_DEACTIVATE_ASYNC_LOAD" not in os.environ:
            # Transformers v5 uses async thread-based tensor materialization by default.
            # Force sequential loading on Windows to avoid the upstream crash seen in CI.
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
            restore_async_load_env = True

        try:
            model = model_cls.from_pretrained(
                **self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs,
                config=cust_config,
                token=access_token,
            )
        finally:
            if restore_async_load_env:
                del os.environ["HF_DEACTIVATE_ASYNC_LOAD"]
        # perhaps explore initializing on the meta device and then materializing as needed layer by layer during
        # loading/processing into hookedtransformer
        # with torch.device("meta"):
        #     model = self.it_cfg.model_class(config=cust_config)  # token=access_token)
        return model

    def tl_config_model_init(self) -> None:
        # TODO: add note to documentation that we currently require tl_cfg to be not None (either from pretrained or
        #       custom config) based, so model_init will not be used. To fully customize TL behavior, override this
        #       method and init config-based HookedTransformer as desired
        # TODO: suppress messages from tl about no tokenizer here, we're deferring the tokenizer attach until setup
        # Note: TransformerBridge requires an HF model, so config-based init always uses HookedTransformer
        if self.it_cfg.tl_cfg.use_bridge:
            rank_zero_warn(
                "TransformerBridge requires an HF model and cannot be initialized from config alone. "
                "Falling back to legacy HookedTransformer for config-based initialization."
            )
        # Filter out IT-specific keys that HookedTransformer doesn't accept
        tl_kwargs = {k: v for k, v in self.it_cfg.tl_cfg.__dict__.items() if k not in ["use_bridge"]}
        self.model = HookedTransformer(tokenizer=self.it_cfg.tokenizer, **tl_kwargs)

    def _prune_tl_cfg_dict(self, normalize_device: bool = False, prune_list: list | None = None) -> dict:
        """Prunes the tl_cfg dictionary by removing IT-specific and HF-specific keys that shouldn't be passed to
        HookedTransformer/TransformerBridge constructors.

        Returns:
            dict: The pruned dictionary
        """
        prune_list = prune_list or ["hf_model", "tokenizer", "use_bridge"]
        pruned_dict = deepcopy(self.it_cfg.tl_cfg.__dict__)

        for key in prune_list:
            if key in pruned_dict:
                if pruned_dict[key] is not None and key not in ["use_bridge"]:
                    rank_zero_warn(f"Found non-None value for '{key}' in tl_cfg. This may cause issues.")
                del pruned_dict[key]

        if normalize_device and "device" in pruned_dict and isinstance(pruned_dict["device"], str):
            pruned_dict["device"] = torch.device(pruned_dict["device"])
        return pruned_dict

    def _prepare_hf_to_tl_conversion(self) -> tuple[Any, Any]:
        """Prepare common state for HF -> TransformerLens conversion.

        Returns a tuple of (tokenizer_handle, hf_preconversion_config).
        """
        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer
        assert self.model is not None, "Model must be loaded before conversion"
        hf_preconversion_config = deepcopy(self.model.config)
        return tokenizer_handle, hf_preconversion_config

    def _convert_hf_to_tl(self) -> None:
        # TODO: decide whether to pass remaining hf_from_pretrained_cfg args to HookedTransformer
        # (other than `dtype` which should already have been processed and removed, `device_map` should also be
        # removed before passing to HookedTransformer)
        tokenizer_handle, hf_preconversion_config = self._prepare_hf_to_tl_conversion()
        pruned_cfg = self._prune_tl_cfg_dict()  # avoid edge case where conflicting keys haven't already been pruned
        self.model = HookedTransformer.from_pretrained(
            hf_model=cast(PreTrainedModel, self.model), tokenizer=tokenizer_handle, **pruned_cfg
        )
        self.model.config = hf_preconversion_config

    def _convert_hf_to_bridge(self) -> None:
        """Convert HF model to TransformerBridge (v3 architecture).

        TransformerBridge wraps the HF model without weight conversion, providing memory efficiency and better HF
        ecosystem compatibility.

        If the tl_cfg is an ITLensBridgeConfig, this method will:
        - Apply any transformer_bridge_config_overrides to the TransformerBridgeConfig
        - Call enable_compatibility_mode() with the specified kwargs if enable_compatibility_mode=True
        """
        from interpretune.config.transformer_lens import ITLensBridgeConfig

        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer
        hf_model = self.model
        hf_preconversion_config = deepcopy(hf_model.config)  # capture original hf config before conversion

        # Map HF config to TransformerLens config
        tl_config = map_default_transformer_lens_config(hf_model.config)

        # Determine architecture from HF config
        architecture = determine_architecture_from_hf_config(hf_model.config)

        # Convert to TransformerBridgeConfig
        bridge_config = TransformerBridgeConfig.from_dict(tl_config.__dict__)
        bridge_config.architecture = architecture

        # Apply any device/dtype overrides from IT config
        pruned_cfg = self._prune_tl_cfg_dict()
        if "device" in pruned_cfg:
            bridge_config.device = pruned_cfg["device"]
        if "dtype" in pruned_cfg:
            # TransformerBridgeConfig.dtype expects torch.dtype; pruned_cfg may contain a string (e.g. "float32")
            bridge_config.dtype = _resolve_dtype(pruned_cfg["dtype"]) or bridge_config.dtype

        # Apply ITLensBridgeConfig-specific overrides if using that config type
        if isinstance(self.it_cfg.tl_cfg, ITLensBridgeConfig):
            if self.it_cfg.tl_cfg.transformer_bridge_config_overrides:
                for key, value in self.it_cfg.tl_cfg.transformer_bridge_config_overrides.items():
                    if hasattr(bridge_config, key):
                        setattr(bridge_config, key, value)
                        rank_zero_debug(f"Applied TransformerBridgeConfig override: {key}={value}")
                    else:
                        rank_zero_warn(
                            f"transformer_bridge_config_overrides key '{key}' not found in "
                            f"TransformerBridgeConfig, ignoring."
                        )

        # Create adapter and bridge
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
        self.model = TransformerBridge(model=hf_model, adapter=adapter, tokenizer=tokenizer_handle)

        # Debug shim: log component mapping for troubleshooting checkpoint restoration
        if hasattr(adapter, "component_mapping"):
            component_mapping = adapter.component_mapping
            if component_mapping is not None:
                hf_prefixes = {getattr(c, "name", "?").split(".")[0] for c in component_mapping.values()}
                rank_zero_debug(f"TransformerBridge component_mapping keys: {list(component_mapping.keys())}")
                rank_zero_debug(f"TransformerBridge HF top-level prefixes: {hf_prefixes}")

        # Enable compatibility mode if requested (ITLensBridgeConfig only)
        if isinstance(self.it_cfg.tl_cfg, ITLensBridgeConfig) and self.it_cfg.tl_cfg.enable_compatibility_mode:
            compat_kwargs = self.it_cfg.tl_cfg.enable_compatibility_mode_kwargs or {}
            rank_zero_info(f"Enabling TransformerBridge compatibility mode with kwargs: {compat_kwargs}")
            self.model.enable_compatibility_mode(**compat_kwargs)

        # Move model to device if move_to_device is enabled (default True)
        if pruned_cfg.get("move_to_device", True) and "device" in pruned_cfg:
            device = pruned_cfg["device"]
            rank_zero_info(f"Moving TransformerBridge to device: {device}")
            self.model.to(device)

        # Preserve original HF config for reference
        self.model.config = hf_preconversion_config

    def _capture_hyperparameters(self) -> None:
        """Capture and serialize hyperparameters for model checkpointing.

        Serializes three types of configurations:
        1. HF PretrainedConfig: Original HF model config (via superclass)
        2. TL Model Config: Actual TransformerLens config (HookedTransformerConfig or TransformerBridgeConfig)
        3. IT TL Config: Interpretune-specific settings (ITLensFromPretrainedConfig or ITLensCustomConfig)
        """
        # Override unsupported from pretrained options
        if self.hf_cfg:
            self.hf_cfg.lora_cfg = None  # type: ignore[assignment]  # config flexibility
            # TODO NEXT: enable bnb now that it's supported by TransformerLens
            self.hf_cfg.bitsandbytesconfig = None  # type: ignore[assignment]  # config flexibility
        # TODO: refactor the captured config here to only add tl_from_pretrained, other added in superclass
        # Serialize the actual TransformerLens config from the initialized model
        # Works for both HookedTransformer (legacy) and TransformerBridge (v3)
        tl_model_cfg = self._make_config_serializable(self.model.cfg, ["device", "dtype"])

        # Add architecture flag for clarity on which path was used
        if hasattr(tl_model_cfg, "__dict__"):
            tl_model_cfg.__dict__["_used_bridge"] = self.it_cfg.tl_cfg.use_bridge

        self._it_state._init_hparams.update({"tl_model_cfg": tl_model_cfg})

        # Serialize IT-specific TransformerLens settings
        self._it_state._init_hparams.update({"it_tl_cfg": self.it_cfg.tl_cfg})

        # Call superclass to serialize HF PretrainedConfig (hf_preconversion_config)
        super()._capture_hyperparameters()

    def set_input_require_grads(self) -> None:
        # not currently supported by ITLensModule
        rank_zero_info("Setting input require grads not currently supported by ITLensModule.")


################################################################################
# TransformerLens Module Composition
################################################################################


class TransformerLensAdapter(TLensAttributeMixin):
    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.transformer_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule,),
            description="TransformerLens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.transformer_lens),  # type: ignore[arg-type]
            composition_classes=(ITDataModule, LightningDataModule),
            description="TransformerLens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.transformer_lens),  # type: ignore[arg-type]
            composition_classes=(ITLensModule,),
            description="TransformerLens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.transformer_lens),  # type: ignore[arg-type]
            composition_classes=(
                TLensAttributeMixin,
                BaseITLensModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
            description="TransformerLens adapter that can be composed with core and l...",
        )

    def batch_to_device(self, batch) -> BatchEncoding:
        device = self.input_device
        if device is not None:
            move_data_to_device(batch, device)
        return batch


class ITLensModule(TransformerLensAdapter, CoreHelperAttributes, BaseITLensModule): ...


if _FTS_AVAILABLE:
    from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

    # TODO: will need to gate this on Lightning being available as well once FTS can use raw torch
    # Strategy adapter for TransformerLens Bridge integration
    class TransformerBridgeStrategyAdapter(StrategyAdapter):
        """Strategy adapter to support TransformerLens Bridge naming translation.

        Enables fine-tuning schedules to use clean TL-style parameter names (e.g., blocks.9.attn.W_Q)
        instead of verbose canonical names (e.g., model.blocks.9._original_component.attn.q._original_component.weight).

        Handles checkpoint key translation between HF-style and TL-style formats:
        - Checkpoint keys use HF prefixes: model.transformer.h.N, model.transformer.wte, etc.
        - Runtime keys use TL prefixes: model.blocks.N, model.embed, etc.

        Uses the TransformerBridge adapter's component_mapping for architecture-agnostic conversion.

        Args:
            model_view: Optional ModelView instance or class (str/Type) for parameter naming transformation.
                If None, uses canonical naming (default behavior).
            model_view_cfg: Optional dict of configuration parameters to pass to ModelView constructor.
                For example: {'implicit_ln_thaw': False} for TLNamesModelView.
                Ignored if model_view is already an instance.
            use_tl_names: Convenience flag. If True, automatically creates TLNamesModelView instance.
                Cannot be used together with model_view parameter.
        """

        def __init__(
            self,
            model_view: None | str | Type[ModelView] | ModelView = None,
            model_view_cfg: dict[str, Any] | None = None,
            use_tl_names: bool = False,
            *args,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)

            # Validate mutually exclusive parameters
            if model_view is not None and use_tl_names:
                raise ValueError("Cannot specify both 'model_view' and 'use_tl_names'. Use one or the other.")

            # Store initialization parameters for deferred model_view creation
            # (model_view needs access to adapter, so we create it in on_before_init_fts)
            # If nothing specified, we'll use CanonicalModelView (identity transformation)
            self._model_view_init: None | str | Type[ModelView] | ModelView = model_view
            self._model_view_cfg: dict[str, Any] = model_view_cfg or {}
            self._use_tl_names: bool = use_tl_names
            self.model_view: ModelView | None = None

            # Always use translation function (even for canonical mode via CanonicalModelView)
            self.exec_ft_phase = partial(StrategyAdapter.base_ft_phase, translation_func=self.logical_param_translation)

        def _ensure_model_view_initialized(self) -> None:
            """Ensure model_view is initialized before use.

            This is called from methods that may be invoked before on_before_init_fts(), such as gen_ft_schedule() when
            gen_ft_sched_only=True.
            """
            if self.model_view is not None:
                return  # Already initialized

            # Initialize model_view on demand (or triggered by on_before_init_fts hasn't run yet)
            if self._use_tl_names:
                self.model_view = TLNamesModelView(self, **self._model_view_cfg)
            elif self._model_view_init is not None:
                if isinstance(self._model_view_init, str):
                    view_class = _import_class(self._model_view_init)
                    self.model_view = view_class(self, **self._model_view_cfg)
                elif isinstance(self._model_view_init, type):
                    self.model_view = self._model_view_init(self, **self._model_view_cfg)
                elif isinstance(self._model_view_init, ModelView):
                    if self._model_view_cfg:
                        rank_zero_warn(
                            "model_view_cfg provided but model_view is already an instance. "
                            "Configuration will be ignored."
                        )
                    self.model_view = self._model_view_init
                else:
                    raise TypeError(
                        f"model_view must be None, str, Type[ModelView], or ModelView instance. "
                        f"Got: {type(self._model_view_init)}"
                    )
            else:
                self.model_view = CanonicalModelView(self, **self._model_view_cfg)

            # Build parameter mapping
            assert self.model_view is not None
            self.model_view.build_param_mapping()

        def on_before_init_fts(self) -> None:
            # we patch our wrapped TransformerBridge module to use the TransformerBridge's state_dict and
            # load_state_dict methods to handle the specialized key translation logic
            assert self.pl_module is not None
            setattr(self.pl_module, "state_dict", self.pl_module.model.state_dict)  # type: ignore[attr-defined]
            setattr(self.pl_module, "load_state_dict", self.pl_module.model.load_state_dict)  # type: ignore[attr-defined]

            # Ensure model_view is initialized (may already be initialized if gen_ft_sched_only=True)
            self._ensure_model_view_initialized()

        def fts_optim_transform(self, orig_pl: list[str], inspect_only: bool = False) -> list[str]:
            """Transform parameter names to canonical names for optimizer.

            Delegates transformation to the active model view.

            Args:
                orig_pl: List of parameter names from schedule
                inspect_only: If True, only validate mapping without transforming

            Returns:
                List of canonical parameter names for optimizer
            """
            assert self.model_view is not None
            return self.model_view.transform_to_canonical(orig_pl, inspect_only=inspect_only)

        def logical_param_translation(self, param_names: list[str]) -> list[str]:
            """Translate canonical parameter names to model view names.

            Delegates transformation to the active model view.

            Args:
                param_names: List of canonical parameter names from optimizer

            Returns:
                List of model view parameter names
            """
            assert self.model_view is not None
            return self.model_view.transform_from_canonical(param_names)

        def get_named_params_for_schedule_validation(self) -> dict[str, torch.nn.Parameter]:
            """Get named parameters for schedule validation.

            Delegates to the active model view for parameter naming.

            Returns:
                dict[str, torch.nn.Parameter]: A dictionary mapping parameter names to parameter tensors.
            """
            self._ensure_model_view_initialized()
            assert self.model_view is not None
            return self.model_view.get_named_params()  # type: ignore[return-value]

        def validate_ft_sched(self) -> tuple[int, int]:
            """Validate the fine-tuning schedule.

            Delegates to the active model view's validation method.

            Returns:
                tuple[int, int]: A tuple of ints specifying:
                    1. The depth of the final scheduled phase
                    2. The maximum epoch watermark explicitly specified in the schedule
            """
            self._ensure_model_view_initialized()
            rank_zero_debug(
                f"[TransformerBridgeStrategyAdapter.validate_ft_sched] Using {self.model_view.__class__.__name__} "
                f"for schedule validation (module: {self.pl_module.__class__.__name__})"
            )
            # Delegate to model view's validation (which calls base StrategyAdapter implementation)
            assert self.model_view is not None
            return self.model_view.validate_schedule()

        def gen_ft_schedule(self, dump_loc: str | os.PathLike) -> os.PathLike | None:
            """Generate fine-tuning schedule using active model view naming.

            Delegates to the active model view's generation method.

            Args:
                dump_loc: Directory to write the generated schedule

            Returns:
                Path to the generated schedule YAML file
            """
            self._ensure_model_view_initialized()
            rank_zero_debug(
                f"[TransformerBridgeStrategyAdapter.gen_ft_schedule] Using {self.model_view.__class__.__name__} "
                f"for schedule generation (module: {self.pl_module.__class__.__name__})"
            )
            assert self.model_view is not None
            return self.model_view.gen_schedule(dump_loc)

        def lightning_module_state_dict(self) -> dict[str, Any]:
            """Override lightning_module_state_dict to use TransformerBridge state_dict to avoid dup keys."""
            assert (
                self.pl_module is not None
                and hasattr(self.pl_module, "model")
                and hasattr(self.pl_module.model, "state_dict")
            )
            return self.pl_module.model.state_dict()  # type: ignore[attr-defined,no-any-return]

        def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
            assert self.pl_module is not None
            self.pl_module.model.load_state_dict(checkpoint["state_dict"], strict=strict)  # type: ignore[attr-defined]

    class TLNamesModelView(ModelView):
        """TransformerLens-style parameter naming strategy.

        Provides clean TL-style names (e.g., blocks.9.attn.W_Q) instead of verbose
        canonical names. Optionally includes implicit LayerNorm thawing since TL
        nomenclature doesn't include LayerNorm parameters.

        Args:
            adapter: The strategy adapter instance
            implicit_ln_thaw: If True (default), automatically thaws LayerNorm parameters
                when attention or MLP blocks are thawed. If False, LayerNorms are not
                implicitly thawed and must be explicitly included in schedules if needed.

        Note:
            Users needing fine-grained LayerNorm control can either use canonical mode
            or set implicit_ln_thaw=False and explicitly manage LayerNorm parameters.
        """

        def __init__(self, adapter: "StrategyAdapter", implicit_ln_thaw: bool = True):
            super().__init__(adapter)
            self.implicit_ln_thaw = implicit_ln_thaw
            self._tl_to_canonical_mapping: dict[str, list[str]] | None = None
            self._canonical_to_tl_mapping: dict[str, str] | None = None
            self._unmapped_canonical_params: set | None = None

        def build_param_mapping(self) -> None:
            """Build bidirectional parameter name mappings using component structure tracing.

            Creates mappings between TL-style names (e.g., blocks.9.attn.W_Q) and canonical names
            (e.g., model.blocks.9._original_component.attn.q._original_component.weight).

            Uses tl_named_parameters() as the source of truth for TL naming, then traces
            back to the underlying component tensors via data_ptr() to find canonical matches.
            This approach works for any architecture since it introspects the actual bridge
            component structure rather than relying on hardcoded pattern lists.

            Note on LayerNorm parameters:
                Canonical LayerNorm parameters (ln_1, ln_2, ln_final) are always present in
                named_parameters() but are NOT mapped to TL parameters. This is expected:
                - TransformerLens (if configured to) folds LayerNorm into subsequent layers mathematically
                - The canonical params remain (weight=1, bias=0 when folded) but have no TL equivalent
                - These canonical LayerNorm params appear in `unmapped_canonical` and can be
                  handled specially during schedule validation
            """
            rank_zero_info("Building TL-style to canonical parameter name mapping...")

            bridge = self.pl_module.model

            # Get TL-style parameter names (source of truth for TL naming)
            tl_params = dict(bridge.tl_named_parameters())  # type: ignore[attr-defined]

            # Get canonical parameters
            canonical_params = dict(self.pl_module.named_parameters())

            # Build index of canonical params by data_ptr for efficient lookup
            canonical_by_ptr: dict[int, list[str]] = {}
            for name, tensor in canonical_params.items():
                ptr = tensor.data_ptr()
                if ptr not in canonical_by_ptr:
                    canonical_by_ptr[ptr] = []
                canonical_by_ptr[ptr].append(name)

            # Build mappings using component structure tracing
            self._tl_to_canonical_mapping = {}
            self._canonical_to_tl_mapping = {}

            for tl_name in tl_params.keys():
                # Get the underlying component tensor (not the TL view)
                underlying_tensor = self._get_underlying_component_tensor(tl_name, bridge)

                if underlying_tensor is not None:
                    ptr = underlying_tensor.data_ptr()
                    if ptr in canonical_by_ptr:
                        canonical_names = canonical_by_ptr[ptr]
                        self._tl_to_canonical_mapping[tl_name] = canonical_names
                        for canonical_name in canonical_names:
                            self._canonical_to_tl_mapping[canonical_name] = tl_name
                    else:
                        # Underlying tensor exists but no canonical match - unexpected
                        rank_zero_debug(f"TL param '{tl_name}' has underlying tensor but no canonical match")
                        self._tl_to_canonical_mapping[tl_name] = []
                else:
                    # No underlying tensor found - might be a virtual param
                    self._tl_to_canonical_mapping[tl_name] = []

            # Validate mapping completeness
            unmapped_tl = [name for name, mapping in self._tl_to_canonical_mapping.items() if not mapping]
            unmapped_canonical = set(canonical_params.keys()) - set(self._canonical_to_tl_mapping.keys())

            if unmapped_tl:
                # Report any unmapped TL params - this is unexpected since we trace through components
                rank_zero_warn(
                    f"Warning: {len(unmapped_tl)} TL-style parameters could not be mapped to canonical params. "
                    f"First few: {unmapped_tl[:5]}"
                )

            # Store unmapped canonical params for later use in validation
            # Expected unmapped canonical params vary by architecture, e.g.:
            # - LayerNorm params (ln_1, ln_2, ln_final) - TL often folds these into subsequent layers, irrespective of
            #   folding though, does not expose them when following the TL naming convention
            # - Combined QKV params - TL exposes separate W_Q, W_K, W_V instead of joint e.g. c_attn
            self._unmapped_canonical_params = unmapped_canonical

            if unmapped_canonical:
                rank_zero_debug(
                    f"{len(unmapped_canonical)} canonical parameters have no TL-style equivalent. "
                    f"First few: {list(unmapped_canonical)[:5]}"
                )

            rank_zero_info(
                f"Mapping complete: {len(self._tl_to_canonical_mapping)} TL-style → "
                f"{sum(len(v) for v in self._tl_to_canonical_mapping.values())} canonical parameters"
            )

        def transform_to_canonical(self, param_names: list[str], inspect_only: bool = False) -> list[str]:
            """Transform TL-style parameter names to canonical names for optimizer.

            If implicit_ln_thaw=True, this method also appends LayerNorm parameters
            for the layers being thawed (since TL nomenclature doesn't include LayerNorm).
            If implicit_ln_thaw=False, only the explicitly specified TL params are transformed.

            Args:
                param_names: List of parameter names from schedule (TL-style)
                inspect_only: If True, only validate mapping without transforming

            Returns:
                List of canonical parameter names for optimizer
            """
            assert self._tl_to_canonical_mapping is not None, (
                "Parameter mapping not initialized. Call build_param_mapping() first."
            )

            canonical_params = []
            for tl_name in param_names:
                if tl_name not in self._tl_to_canonical_mapping:
                    raise ValueError(
                        f"TL-style parameter '{tl_name}' not found in mapping. "
                        f"Available TL names: {list(self._tl_to_canonical_mapping.keys())[:10]}..."
                    )

                # A TL parameter may map to multiple canonical parameters (views)
                canonical_names = self._tl_to_canonical_mapping[tl_name]
                if not canonical_names:
                    raise ValueError(f"TL-style parameter '{tl_name}' has no canonical mapping")

                canonical_params.extend(canonical_names)

            # Append implicit LayerNorm params if enabled
            # TL nomenclature doesn't include LayerNorm, but canonical params need them for training
            implicit_ln_params = []
            if self.implicit_ln_thaw:
                implicit_ln_params = self._get_implicit_layernorm_params(param_names)
                canonical_params.extend(implicit_ln_params)

            if not inspect_only:
                if self.implicit_ln_thaw:
                    rank_zero_debug(
                        f"Transformed {len(param_names)} TL-style params → {len(canonical_params)} canonical params "
                        f"(including {len(implicit_ln_params)} implicit LayerNorm params)"
                    )
                else:
                    rank_zero_debug(
                        f"Transformed {len(param_names)} TL-style params → {len(canonical_params)} canonical params "
                        f"(implicit_ln_thaw=False, no LayerNorm params added)"
                    )

            return canonical_params

        def transform_from_canonical(self, param_names: list[str]) -> list[str]:
            """Translate canonical parameter names to TL-style names.

            Args:
                param_names: List of canonical parameter names from optimizer

            Returns:
                List of TL-style parameter names (or canonical if no mapping exists)
            """
            assert self._canonical_to_tl_mapping is not None, (
                "Parameter mapping not initialized. Call build_param_mapping() first."
            )

            tl_params = []
            for canonical_name in param_names:
                if canonical_name not in self._canonical_to_tl_mapping:
                    # Some canonical parameters might not have TL equivalents (e.g., LayerNorm)
                    rank_zero_debug(
                        f"Canonical parameter '{canonical_name}' not found in reverse mapping, keeping as-is"
                    )
                    tl_params.append(canonical_name)
                else:
                    tl_params.append(self._canonical_to_tl_mapping[canonical_name])

            # Remove duplicates while preserving order
            seen = set()
            unique_tl_params = []
            for name in tl_params:
                if name not in seen:
                    seen.add(name)
                    unique_tl_params.append(name)

            return unique_tl_params

        def get_named_params(self) -> dict[str, torch.Tensor]:
            """Get named parameters for schedule validation.

            Returns TL-style parameter names from the TransformerBridge.

            Returns:
                dict[str, torch.Tensor]: A dictionary mapping TL-style names to parameter tensors.
            """
            return dict(self.pl_module.model.tl_named_parameters())  # type: ignore[attr-defined]

        def gen_schedule(self, dump_loc: str | os.PathLike) -> os.PathLike | None:
            """Generate fine-tuning schedule using TL-style parameter names.

            Generates schedule with clean TL-style names (e.g., blocks.9.attn.W_Q).

            Args:
                dump_loc: Directory to write the generated schedule

            Returns:
                Path to the generated schedule YAML file
            """
            from finetuning_scheduler.fts_supporters import ScheduleImplMixin

            # Generate schedule with TL-style names
            rank_zero_debug("TLNamesModelView.gen_schedule() called")
            rank_zero_info(f"Generating TL-style fine-tuning schedule for {self.pl_module.__class__.__name__}")

            param_lists: list = []
            cur_group: list = []

            # Use TL-style parameter names
            model_params = list(self.pl_module.model.tl_named_parameters())[::-1]  # type: ignore[attr-defined]

            # Apply 2-parameters per-level heuristic
            for i, (n, _) in enumerate(model_params):
                if i % 2 == 0:
                    cur_group = []
                    cur_group.append(n)
                else:
                    cur_group.append(n)
                    param_lists.append(cur_group)

            if len(model_params) % 2 == 1:
                param_lists.append([model_params[-1][0]])

            # Build schedule config
            layer_config = {}
            for i, param_l in enumerate(param_lists):
                layer_config[i] = {"params": param_l}

            schedule_name = f"{self.pl_module.__class__.__name__}_ft_schedule.yaml"
            assert dump_loc is not None
            return ScheduleImplMixin.save_schedule(schedule_name, layer_config, dump_loc)

        def validate_schedule(self) -> tuple[int, int]:
            """Validate the fine-tuning schedule with TL-style parameter mapping diagnostics.

            Logs diagnostic information about the parameter mappings before delegating
            to the standard validation. This helps debug schedule issues and understand
            which canonical LayerNorm params are unmapped.

            Returns:
                tuple[int, int]: A tuple of ints specifying:
                    1. The depth of the final scheduled phase
                    2. The maximum epoch watermark explicitly specified in the schedule
            """
            rank_zero_debug("TLNamesModelView.validate_schedule() called")
            if self._tl_to_canonical_mapping is not None and self._canonical_to_tl_mapping is not None:
                # Log mapping diagnostics for debugging
                rank_zero_debug(
                    f"TL-style schedule validation - TL→Canonical mapping summary:\n"
                    f"  Total TL params: {len(self._tl_to_canonical_mapping)}\n"
                    f"  Total mapped canonical params: {len(self._canonical_to_tl_mapping)}"
                )

                # Log unmapped canonical params (expected to include LayerNorms)
                if self._unmapped_canonical_params:
                    # Categorize unmapped canonical params
                    ln_params = [p for p in self._unmapped_canonical_params if "ln_" in p or "ln_final" in p]
                    other_params = [p for p in self._unmapped_canonical_params if p not in ln_params]

                    rank_zero_debug(
                        f"Unmapped canonical parameters ({len(self._unmapped_canonical_params)} total):\n"
                        f"  LayerNorm params: {len(ln_params)} (expected - TL nomenclature does not include these)\n"
                        f"  Other params: {len(other_params)} (e.g., combined QKV)"
                    )

                    if ln_params:
                        rank_zero_debug(f"  LayerNorm sample: {ln_params[:3]}")
                    if other_params:
                        rank_zero_debug(f"  Other sample: {other_params[:3]}")

            # Delegate to base StrategyAdapter validation
            from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

            return StrategyAdapter.validate_ft_sched(self.adapter)

        # Private helper methods for component structure tracing

        def _get_underlying_component_tensor(self, tl_name: str, bridge: Any) -> torch.Tensor | None:
            """Get the underlying component tensor for a TL-style parameter name.

            Maps TL names like 'blocks.0.attn.W_Q' to the actual component tensor
            (bridge.blocks[0].attn.q.weight), which shares data_ptr with canonical params.

            The TL tensors from tl_named_parameters() may be views/reshapes (via einops),
            but the underlying component tensors are the actual nn.Parameter objects.

            Args:
                tl_name: TL-style parameter name from tl_named_parameters()
                bridge: TransformerBridge instance

            Returns:
                The underlying component tensor if found, None otherwise.
            """
            parts = tl_name.split(".")

            try:
                if parts[0] == "blocks":
                    layer_idx = int(parts[1])
                    block = bridge.blocks[layer_idx]
                    component_type = parts[2]  # attn, mlp, ln1, ln2
                    param_name = parts[3]  # W_Q, b_Q, W_in, w, etc.

                    if component_type == "attn":
                        return self._get_attn_component_tensor(block.attn, param_name)
                    elif component_type == "mlp":
                        return self._get_mlp_component_tensor(block.mlp, param_name)
                    elif component_type in ("ln1", "ln2"):
                        ln = block.ln_1 if component_type == "ln1" else getattr(block, "ln_2", None)
                        if ln is not None:
                            return self._get_ln_component_tensor(ln, param_name)

                elif parts[0] == "embed":
                    return getattr(bridge.embed, "weight", None)

                elif parts[0] == "pos_embed":
                    pos_embed = getattr(bridge, "pos_embed", None)
                    if pos_embed is not None:
                        return getattr(pos_embed, "weight", None)

                elif parts[0] == "unembed":
                    if parts[1] == "W_U":
                        return getattr(bridge.unembed, "weight", None)
                    elif parts[1] == "b_U":
                        return getattr(bridge.unembed, "bias", None)

                elif parts[0] == "ln_final":
                    ln_final = getattr(bridge, "ln_final", None)
                    if ln_final is not None:
                        return self._get_ln_component_tensor(ln_final, parts[1])

            except (AttributeError, IndexError, KeyError, ValueError) as e:
                rank_zero_debug(f"Failed to trace TL name '{tl_name}': {e}")

            return None

        def _get_attn_component_tensor(self, attn: Any, param_name: str) -> torch.Tensor | None:
            """Get attention component tensor.

            Maps TL attention param names to underlying component tensors:
            - W_Q -> attn.q.weight
            - W_K -> attn.k.weight
            - W_V -> attn.v.weight
            - W_O -> attn.o.weight
            - b_Q -> attn.q.bias
            - etc.
            """
            # Map param suffix to (component_attr, tensor_attr)
            mapping = {
                "W_Q": ("q", "weight"),
                "W_K": ("k", "weight"),
                "W_V": ("v", "weight"),
                "W_O": ("o", "weight"),
                "b_Q": ("q", "bias"),
                "b_K": ("k", "bias"),
                "b_V": ("v", "bias"),
                "b_O": ("o", "bias"),
            }

            if param_name in mapping:
                comp_name, attr_name = mapping[param_name]
                comp = getattr(attn, comp_name, None)
                if comp is not None:
                    return getattr(comp, attr_name, None)

            return None

        def _get_mlp_component_tensor(self, mlp: Any, param_name: str) -> torch.Tensor | None:
            """Get MLP component tensor.

            Maps TL MLP param names to underlying component tensors:
            - W_in -> mlp.in.weight (or mlp.input.weight)
            - W_out -> mlp.out.weight
            - W_gate -> mlp.gate.weight
            - b_in -> mlp.in.bias
            - etc.
            """
            if param_name == "W_in":
                comp = getattr(mlp, "in", None) or getattr(mlp, "input", None)
                return getattr(comp, "weight", None) if comp else None
            elif param_name == "b_in":
                comp = getattr(mlp, "in", None) or getattr(mlp, "input", None)
                return getattr(comp, "bias", None) if comp else None
            elif param_name == "W_out":
                comp = getattr(mlp, "out", None)
                return getattr(comp, "weight", None) if comp else None
            elif param_name == "b_out":
                comp = getattr(mlp, "out", None)
                return getattr(comp, "bias", None) if comp else None
            elif param_name == "W_gate":
                comp = getattr(mlp, "gate", None)
                return getattr(comp, "weight", None) if comp else None
            elif param_name == "b_gate":
                comp = getattr(mlp, "gate", None)
                return getattr(comp, "bias", None) if comp else None

            return None

        def _get_ln_component_tensor(self, ln: Any, param_name: str) -> torch.Tensor | None:
            """Get LayerNorm/RMSNorm component tensor.

            Maps TL LayerNorm param names to underlying component tensors:
            - w -> ln.weight
            - b -> ln.bias
            """
            if param_name in ("w", "weight"):
                return getattr(ln, "weight", None)
            elif param_name in ("b", "bias"):
                return getattr(ln, "bias", None)

            return None

        def _get_implicit_layernorm_params(self, tl_param_names: list[str]) -> list[str]:
            """Get implicit LayerNorm canonical params for the layers referenced by TL params.

            TL-style nomenclature doesn't include LayerNorm parameters, but canonical training
            needs them. This method extracts the layer indices from TL param names and returns
            the corresponding canonical LayerNorm params.

            For each layer index found in tl_param_names:
            - Adds ln_1 params (weight/bias) for that block
            - Adds ln_2 params (weight/bias) for that block

            Also handles embeddings:
            - If any embed/unembed params are present, includes ln_final

            Args:
                tl_param_names: List of TL-style parameter names being thawed

            Returns:
                List of canonical LayerNorm parameter names to implicitly thaw
            """
            if not self._unmapped_canonical_params:
                return []

            implicit_ln_params: list[str] = []
            layer_indices_seen: set = set()
            has_embed_params = False

            # Extract layer indices from TL param names
            block_pattern = re.compile(r"^blocks\.(\d+)\.")

            for tl_name in tl_param_names:
                match = block_pattern.match(tl_name)
                if match:
                    layer_indices_seen.add(int(match.group(1)))
                elif tl_name.startswith(("embed.", "pos_embed.", "unembed.")):
                    has_embed_params = True

            # Find matching LayerNorm params from unmapped canonical params
            for canonical_name in sorted(self._unmapped_canonical_params):
                # Check for block LayerNorm params (ln_1, ln_2)
                block_ln_match = re.search(r"blocks\.(\d+)\..*?(ln_1|ln_2)", canonical_name)
                if block_ln_match:
                    layer_idx = int(block_ln_match.group(1))
                    if layer_idx in layer_indices_seen:
                        implicit_ln_params.append(canonical_name)
                        continue

                # Check for ln_final (associated with embeddings/unembed)
                if has_embed_params and "ln_final" in canonical_name:
                    implicit_ln_params.append(canonical_name)

            return implicit_ln_params

else:
    TransformerBridgeStrategyAdapter = object  # type: ignore[misc,assignment]
    TLNamesModelView = object  # type: ignore[misc,assignment]
