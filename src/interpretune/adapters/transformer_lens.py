import os
from typing import Optional, Type, cast, Union, Dict, Any, List, Mapping
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
from interpretune.base import CoreHelperAttributes, ITDataModule, BaseITModule
from interpretune.utils import move_data_to_device, rank_zero_warn, rank_zero_info, rank_zero_debug, _FTS_AVAILABLE
from interpretune.protocol import Adapter


################################################################################
# Mixins to support TransformerLens in different adapter contexts
################################################################################


class TLensAttributeMixin:
    @property
    def tl_cfg(self) -> Optional[Union[HookedTransformerConfig, TransformerBridgeConfig]]:
        try:
            cfg = reduce(getattr, "model.cfg".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a TransformerLens config reference (has it been set yet?): {ae}")
            cfg = None
        return cfg  # type: ignore[return-value]

    # TODO: we aren't using IT's Property Composition feature for TLens yet, but might be worth enabling it
    @property
    def device(self) -> Optional[torch.device]:
        device: Optional[torch.device] = None
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
    def device(self, value: Optional[str | torch.device]) -> None:
        if value is not None and not isinstance(value, torch.device):
            value = torch.device(value)
        self._it_state._device = value  # type: ignore[attr-defined]  # provided by mixing class

    def get_tl_device(self) -> Optional[torch.device]:
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
    def output_device(self) -> Optional[torch.device]:
        return self.get_tl_device()  # type: ignore[attr-defined]  # provided by mixing class

    @property
    def input_device(self) -> Optional[torch.device]:
        return self.get_tl_device()


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
        self, cust_config: HFPretrainedConfig, access_token: Optional[str] = None
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
        model = model_cls.from_pretrained(
            **self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs,
            config=cust_config,
            token=access_token,
        )
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

    def _prune_tl_cfg_dict(self, prune_list: Optional[list] = None) -> dict:
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

        return pruned_dict

    def _convert_hf_to_tl(self) -> None:
        # TODO: decide whether to pass remaining hf_from_pretrained_cfg args to HookedTransformer
        # (other than `dtype` which should already have been processed and removed, `device_map` should also be
        # removed before passing to HookedTransformer)
        # if datamodule is not attached yet, attempt to retrieve tokenizer handle directly from provided it_cfg
        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer
        hf_preconversion_config = deepcopy(self.model.config)  # capture original hf config before conversion
        pruned_cfg = self._prune_tl_cfg_dict()  # avoid edge case where conflicting keys haven't already been pruned
        self.model = HookedTransformer.from_pretrained(
            hf_model=cast(PreTrainedModel, self.model), tokenizer=tokenizer_handle, **pruned_cfg
        )
        self.model.config = hf_preconversion_config

    def _convert_hf_to_bridge(self) -> None:
        """Convert HF model to TransformerBridge (v3 architecture).

        TransformerBridge wraps the HF model without weight conversion, providing memory efficiency and better HF
        ecosystem compatibility.
        """
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
            bridge_config.dtype = pruned_cfg["dtype"]

        # Create adapter and bridge
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)

        # Inspect HF model before TransformerBridge instantiation
        inspect_state_dict_or_params(
            hf_model.state_dict(),
            "hf_model.state_dict()",
            context_message="Before TransformerBridge instantiation",
            prefix_filter=None,
            num_keys_sample=10,
            log_debug=True,
        )
        inspect_state_dict_or_params(
            hf_model,
            "hf_model.named_parameters()",
            context_message="Before TransformerBridge instantiation",
            prefix_filter=None,
            num_keys_sample=10,
            log_debug=True,
        )

        self.model = TransformerBridge(model=hf_model, adapter=adapter, tokenizer=tokenizer_handle)

        # Debug shim: log component mapping for troubleshooting checkpoint restoration
        if hasattr(adapter, "component_mapping"):
            component_mapping = adapter.component_mapping
            hf_prefixes = {getattr(c, "name", "?").split(".")[0] for c in component_mapping.values()}
            rank_zero_debug(f"TransformerBridge component_mapping keys: {list(component_mapping.keys())}")
            rank_zero_debug(f"TransformerBridge HF top-level prefixes: {hf_prefixes}")

        # Inspect TransformerBridge after instantiation
        inspect_state_dict_or_params(
            self.model,
            "self.model.named_parameters()",
            context_message="After TransformerBridge instantiation",
            prefix_filter="",
            num_keys_sample=10,
            log_debug=True,
        )
        inspect_state_dict_or_params(
            self.model,
            "self.model.tl_named_parameters()",
            context_message="After TransformerBridge instantiation",
            prefix_filter="",
            num_keys_sample=10,
            log_debug=True,
        )
        inspect_state_dict_or_params(
            self.model.state_dict(),
            "self.model.state_dict()",
            context_message="After TransformerBridge instantiation",
            prefix_filter="",
            num_keys_sample=10,
            log_debug=True,
        )

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

    def inspect_state_dict_or_params(
        source: Union[Dict[str, Any], Any],
        source_name: str = "state_dict",
        context_message: Optional[str] = None,
        prefix_filter: Optional[str] = "model.",
        num_keys_sample: int = 10,
        log_debug: bool = False,
        return_string: bool = False,
    ) -> Optional[str]:
        """Inspect and pretty-print state_dict or named_parameters for debugging checkpoint flows.

        This is a standalone debugging utility that can be used at any point in the checkpoint
        save/load pipeline to inspect the structure, types, shapes, and devices of model parameters.

        Args:
            source: Either a state_dict (dict) or an object with named_parameters() method
            source_name: Descriptive name for the source being inspected
            context_message: Optional context message to display before inspection output
            prefix_filter: Only include keys starting with this prefix (None = no filtering)
            num_keys_sample: Number of first/last keys to display in summary
            log_debug: If True, also log output at DEBUG level
            return_string: If True, return the formatted string instead of printing

        Returns:
            Formatted string if return_string=True, "data collection error" on failure, otherwise None

        Example:
            >>> # Inspect a model's state_dict
            >>> inspect_state_dict_or_params(
            ...     model.state_dict(),
            ...     "model.state_dict()",
            ...     context_message="Before checkpoint save"
            ... )
            >>>
            >>> # Inspect named_parameters iterator with requires_grad count
            >>> inspect_state_dict_or_params(
            ...     model,
            ...     "model.named_parameters()",
            ...     context_message="After TransformerBridge instantiation"
            ... )
        """
        from pprint import pformat

        try:
            # Convert source to dict format
            is_named_params = False
            requires_grad_count = None

            if isinstance(source, dict):
                state_dict = source
            elif hasattr(source, "named_parameters") or hasattr(source, "tl_named_parameters"):
                # Prefer tl_named_parameters when explicitly requested via source_name
                prefer_tl = "tl_named_parameters" in (source_name or "")
                prefer_named = "named_parameters" in (source_name or "")
                # Choose the appropriate iterator when available
                if hasattr(source, "tl_named_parameters") and (
                    prefer_tl or (not prefer_named and not hasattr(source, "named_parameters"))
                ):
                    is_named_params = True
                    params_list = list(source.tl_named_parameters())
                elif hasattr(source, "named_parameters"):
                    is_named_params = True
                    params_list = list(source.named_parameters())
                else:
                    # Fall back defensively
                    rank_zero_warn(
                        f"inspect_state_dict_or_params: Source has no named parameter iterators, got {type(source)}"
                    )
                    return "data collection error"
                state_dict = dict(params_list)
                requires_grad_count = sum(1 for _, p in params_list if hasattr(p, "requires_grad") and p.requires_grad)
            else:
                rank_zero_warn(
                    f"inspect_state_dict_or_params: Source must be a dict or have named_parameters() method, "
                    f"got {type(source)}"
                )
                return "data collection error"

            # Apply prefix filter if specified
            if prefix_filter is not None:
                filtered_dict = {k: v for k, v in state_dict.items() if k.startswith(prefix_filter)}
            else:
                filtered_dict = state_dict

            # Collect metadata
            num_keys = len(filtered_dict)
            keys_list = list(filtered_dict.keys())

            # Analyze each value
            value_metadata = []
            for key, value in filtered_dict.items():
                meta = {
                    "key": key,
                    "type": type(value).__name__,
                    "shape": tuple(value.shape) if hasattr(value, "shape") else None,
                    "device": str(value.device) if hasattr(value, "device") else None,
                    "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
                }
                value_metadata.append(meta)

            # Build output string
            lines = []
            lines.append("=" * 80)
            lines.append(f"STATE INSPECTION: {source_name}")
            if context_message:
                lines.append(f"Context: {context_message}")
            lines.append("=" * 80)
            lines.append(f"Prefix filter: {prefix_filter if prefix_filter else 'None (all keys)'}")
            lines.append(f"Total keys: {num_keys}")
            if is_named_params and requires_grad_count is not None:
                lines.append(f"Parameters with requires_grad=True: {requires_grad_count}")
            lines.append("")

            # Sample of first/last keys
            lines.append(f"First {min(num_keys_sample, num_keys)} keys:")
            for key in keys_list[:num_keys_sample]:
                lines.append(f"  {key}")
            lines.append("")

            if num_keys > num_keys_sample * 2:
                lines.append(f"Last {min(num_keys_sample, num_keys)} keys:")
                for key in keys_list[-num_keys_sample:]:
                    lines.append(f"  {key}")
                lines.append("")

            # Metadata summary
            lines.append("Value metadata summary (first 10):")
            for meta in value_metadata[:10]:
                lines.append(f"  {meta['key']}:")
                lines.append(f"    type: {meta['type']}")
                if meta["shape"] is not None:
                    lines.append(f"    shape: {meta['shape']}")
                if meta["device"] is not None:
                    lines.append(f"    device: {meta['device']}")
                if meta["dtype"] is not None:
                    lines.append(f"    dtype: {meta['dtype']}")
            lines.append("")

            # Type/shape distribution
            type_counts = {}
            shape_counts = {}
            device_counts = {}
            for meta in value_metadata:
                type_counts[meta["type"]] = type_counts.get(meta["type"], 0) + 1
                if meta["shape"] is not None:
                    shape_counts[meta["shape"]] = shape_counts.get(meta["shape"], 0) + 1
                if meta["device"] is not None:
                    device_counts[meta["device"]] = device_counts.get(meta["device"], 0) + 1

            lines.append("Type distribution:")
            lines.append(pformat(type_counts, indent=2))
            lines.append("")

            if shape_counts:
                lines.append("Shape distribution (top 10):")
                sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                lines.append(pformat(dict(sorted_shapes), indent=2))
                lines.append("")

            if device_counts:
                lines.append("Device distribution:")
                lines.append(pformat(device_counts, indent=2))
                lines.append("")

            lines.append("=" * 80)

            output = "\n".join(lines)

            # Output handling
            if log_debug:
                rank_zero_debug(f"\n{output}")

            if return_string:
                return output
            else:
                print(output)
                return None

        except Exception as e:
            error_msg = f"inspect_state_dict_or_params error for {source_name}: {e}"
            rank_zero_warn(error_msg)
            return "data collection error"

    # TODO: will need to gate this on Lightning being available as well once FTS can use raw torch
    # Strategy adapter for TransformerLens Bridge integration
    class TransformerBridgeStrategyAdapter(StrategyAdapter):
        """Strategy adapter to support TransformerLens Bridge naming translation.

        Handles checkpoint key translation between HF-style and TL-style formats:
        - Checkpoint keys use HF prefixes: model.transformer.h.N, model.transformer.wte, etc.
        - Runtime keys use TL prefixes: model.blocks.N, model.embed, etc.

        Uses the TransformerBridge adapter's component_mapping for architecture-agnostic conversion.
        """

        # Attach standalone debugging utility as static method
        inspect_state = staticmethod(inspect_state_dict_or_params)

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.exec_ft_phase = partial(StrategyAdapter.base_ft_phase, translation_func=self.logical_param_translation)

        def on_before_init_fts(self) -> None:
            # we patch our wrapped TransformerBridge module to use the TransformerBridge's state_dict and
            # load_state_dict methods to handle the specialized key translation logic
            setattr(self.pl_module, "state_dict", self.pl_module.model.state_dict)
            setattr(self.pl_module, "load_state_dict", self.pl_module.model.load_state_dict)

        def logical_param_translation(self, param_names: List[str]) -> List[str]:
            return param_names

        def fts_optim_transform(self, orig_pl: List[str], inspect_only: bool = False) -> List[str]:
            return orig_pl

        def lightning_module_state_dict(self) -> dict[str, Any]:
            """Override lightning_module_state_dict to use TransformerBridge state_dict to avoid dup keys."""
            assert (
                self.pl_module is not None
                and hasattr(self.pl_module, "model")
                and hasattr(self.pl_module.model, "state_dict")
            )
            return self.pl_module.model.state_dict()

        def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
            assert self.pl_module is not None
            self.pl_module.model.load_state_dict(checkpoint["state_dict"], strict=strict)

else:
    TransformerBridgeStrategyAdapter = object
