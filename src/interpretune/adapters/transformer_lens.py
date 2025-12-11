import os
from typing import Optional, Type, cast, Union, Dict, Any, List
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
        ### TMP DEBUG SHIM START ###
        rank_zero_debug("=== BRIDGE CONVERSION DEBUG ===")

        # 1. Capture HF model state dict
        hf_state_dict = hf_model.state_dict()
        hf_keys_sample = list(hf_state_dict.keys())[:10]
        rank_zero_debug(f"HF model state_dict keys (sample): {hf_keys_sample}")

        # 2. Create Bridge
        ### TMP DEBUG SHIM END ###
        self.model = TransformerBridge(model=hf_model, adapter=adapter, tokenizer=tokenizer_handle)

        # Debug shim: log component mapping for troubleshooting checkpoint restoration
        if hasattr(adapter, "component_mapping"):
            component_mapping = adapter.component_mapping
            hf_prefixes = {getattr(c, "name", "?").split(".")[0] for c in component_mapping.values()}
            rank_zero_debug(f"TransformerBridge component_mapping keys: {list(component_mapping.keys())}")
            rank_zero_debug(f"TransformerBridge HF top-level prefixes: {hf_prefixes}")

        ### TMP DEBUG SHIM START ###
        # 3. Capture Bridge model state dict (default - should return TL style keys)
        bridge_state_dict = self.model.state_dict()
        bridge_keys_sample = list(bridge_state_dict.keys())[:10]
        rank_zero_debug(f"Bridge model.state_dict() keys (sample): {bridge_keys_sample}")

        # 4. Capture Bridge TL-style state dict (if different)
        tl_params = {n: p.shape for n, p in self.model.tl_named_parameters()}
        tl_keys_sample = list(tl_params.keys())[:10]
        rank_zero_debug(f"Bridge tl_named_parameters() keys (sample): {tl_keys_sample}")

        # 5. Capture Bridge runtime parameters (named_parameters - what optimizer sees)
        runtime_params = {n: p.shape for n, p in self.model.named_parameters()}
        runtime_keys_sample = list(runtime_params.keys())[:10]
        rank_zero_debug(f"Bridge named_parameters() keys (sample): {runtime_keys_sample}")
        ### TMP DEBUG SHIM END ###
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

    # Strategy adapter for TransformerLens Bridge integration
    class TransformerBridgeStrategyAdapter(StrategyAdapter):
        """Strategy adapter to support TransformerLens Bridge naming translation.

        Handles checkpoint key translation between HF-style and TL-style formats:
        - Checkpoint keys use HF prefixes: model.transformer.h.N, model.transformer.wte, etc.
        - Runtime keys use TL prefixes: model.blocks.N, model.embed, etc.

        Uses the TransformerBridge adapter's component_mapping for architecture-agnostic conversion.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.exec_ft_phase = partial(StrategyAdapter.base_ft_phase, translation_func=self.logical_param_translation)
            self._component_mapping_cache: Optional[Dict[str, str]] = None
            self._hf_top_level_prefixes_cache: Optional[set] = None

        def logical_param_translation(self, param_names: List[str]) -> List[str]:
            return param_names

        def fts_optim_transform(self, orig_pl: List[str], inspect_only: bool = False) -> List[str]:
            return orig_pl

        def _get_bridge_adapter(self) -> Optional[Any]:
            """Get the TransformerBridge adapter from the pl_module if available."""
            try:
                model = getattr(self.pl_module, "model", None)
                if model is not None and hasattr(model, "adapter"):
                    return model.adapter
            except Exception:
                pass
            return None

        def _build_prefix_mapping(self, adapter: Any) -> Dict[str, str]:
            """Build HF->TL prefix mapping from the adapter's component_mapping.

            Returns a dict mapping HF prefixes to TL prefixes, e.g.: {     'transformer.wte': 'embed',
            'transformer.wpe': 'pos_embed',     'transformer.h': 'blocks',     'transformer.ln_f': 'ln_final',
            'lm_head': 'unembed', }

            Also populates self._hf_top_level_prefixes_cache with unique top-level HF prefixes (e.g., {'transformer',
            'lm_head'} for GPT-2, {'model', 'lm_head'} for Llama).
            """
            if self._component_mapping_cache is not None:
                return self._component_mapping_cache

            mapping: Dict[str, str] = {}
            hf_top_level_prefixes: set = set()
            component_mapping = getattr(adapter, "component_mapping", None)
            if component_mapping is None:
                self._hf_top_level_prefixes_cache = hf_top_level_prefixes
                return mapping

            for tl_name, component in component_mapping.items():
                hf_path = getattr(component, "name", None)
                if hf_path is not None:
                    mapping[hf_path] = tl_name
                    # Extract top-level prefix (first component of the path)
                    top_level = hf_path.split(".")[0]
                    hf_top_level_prefixes.add(top_level)

            self._component_mapping_cache = mapping
            self._hf_top_level_prefixes_cache = hf_top_level_prefixes
            return mapping

        def _convert_checkpoint_key_to_runtime(self, ckpt_key: str, prefix_mapping: Dict[str, str]) -> str:
            """Convert a checkpoint key (HF-style prefix) to runtime key (TL-style prefix).

            Checkpoint: model.transformer.h.0._original_component.attn.q._original_component.weight
            Runtime:    model.blocks.0._original_component.attn.q._original_component.weight

            Only converts the top-level prefix path. Everything after the first _original_component
            stays the same since those use TL-style submodule names already.
            """
            if not ckpt_key.startswith("model."):
                return ckpt_key

            key = ckpt_key[6:]  # Remove 'model.' prefix
            oc_marker = "._original_component"

            # Find the first _original_component - everything before it is the HF prefix to convert
            first_oc_idx = key.find(oc_marker)
            if first_oc_idx == -1:
                # No _original_component - shouldn't happen for bridge keys, return as-is
                return ckpt_key

            hf_prefix = key[:first_oc_idx]
            rest = key[first_oc_idx:]  # includes the _original_component

            # Convert HF prefix to TL prefix using mapping
            tl_prefix = hf_prefix  # default: keep as-is

            # Check for blocks pattern: e.g. transformer.h.N -> blocks.N
            blocks_hf_prefix = None
            for hf_path, tl_name in prefix_mapping.items():
                if tl_name == "blocks":
                    blocks_hf_prefix = hf_path
                    break

            if blocks_hf_prefix and hf_prefix.startswith(blocks_hf_prefix + "."):
                layer_part = hf_prefix[len(blocks_hf_prefix) + 1 :]
                tl_prefix = f"blocks.{layer_part}"
            else:
                # Check other top-level components (exact match)
                for hf_path, tl_name in prefix_mapping.items():
                    if hf_prefix == hf_path:
                        tl_prefix = tl_name
                        break

            return "model." + tl_prefix + rest

        def before_restore_model(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
            """Prepare checkpoint for TransformerBridge restoration.

            TransformerBridge creates multiple parameter aliases:
            1. Original HF keys in checkpoint: model.transformer.*
            2. TL-style aliases at runtime: model.embed.*, model.blocks.*, etc.
            3. Prefixed HF keys at runtime: model.original_model.transformer.*

            Checkpoints contain HF-style keys. We need to transform them to match
            the runtime module structure:
            - Add model.original_model.* prefix for HF keys
            - Add TL-style aliases using adapter component mapping

            Uses the bridge adapter's component_mapping for architecture-agnostic conversion.
            """
            state_dict = checkpoint.get("state_dict", {})
            if not state_dict:
                return checkpoint

            # Get the bridge adapter for component mapping
            adapter = self._get_bridge_adapter()
            if adapter is None:
                rank_zero_warn(
                    "TransformerBridgeStrategyAdapter: Could not access bridge adapter. "
                    "Skipping checkpoint key conversion."
                )
                return checkpoint

            prefix_mapping = self._build_prefix_mapping(adapter)
            if not prefix_mapping:
                rank_zero_warn(
                    "TransformerBridgeStrategyAdapter: No component mapping found. Skipping checkpoint key conversion."
                )
                return checkpoint

            rank_zero_info(
                "TransformerBridgeStrategyAdapter: Expanding checkpoint keys for TransformerBridge aliases..."
            )
            rank_zero_debug(f"TransformerBridgeStrategyAdapter: Prefix mapping: {prefix_mapping}")

            # Get HF top-level prefixes for architecture-agnostic original_model prefixing
            hf_top_level_prefixes = self._hf_top_level_prefixes_cache or set()
            rank_zero_debug(f"TransformerBridgeStrategyAdapter: HF top-level prefixes: {hf_top_level_prefixes}")

            expanded_state_dict = {}
            sample_conversions = []
            sample_original_model_keys = []

            for key, value in state_dict.items():
                if not key.startswith("model."):
                    # Non-model keys (optimizer states, etc.) pass through unchanged
                    expanded_state_dict[key] = value
                    continue

                # Extract the component after "model." prefix to check against HF prefixes
                key_after_model = key[6:]  # Remove "model." prefix
                matched_hf_prefix = None
                for hf_prefix in hf_top_level_prefixes:
                    if key_after_model.startswith(hf_prefix + ".") or key_after_model == hf_prefix:
                        matched_hf_prefix = hf_prefix
                        break

                if matched_hf_prefix:
                    # Add original_model prefix for HF keys (required by runtime module structure)
                    original_model_key = f"model.original_model.{key_after_model}"
                    expanded_state_dict[original_model_key] = value
                    if len(sample_original_model_keys) < 3:
                        sample_original_model_keys.append(f"{key} -> {original_model_key}")
                else:
                    # Non-HF model keys - keep as-is
                    expanded_state_dict[key] = value

                # Add TL-style alias key using component mapping
                tl_key = self._convert_checkpoint_key_to_runtime(key, prefix_mapping)
                if tl_key != key:
                    expanded_state_dict[tl_key] = value
                    if len(sample_conversions) < 5:
                        sample_conversions.append(f"{key} -> {tl_key}")

            checkpoint["state_dict"] = expanded_state_dict
            rank_zero_info(
                f"TransformerBridgeStrategyAdapter: Expanded {len(state_dict)} keys to {len(expanded_state_dict)} keys"
            )
            if sample_original_model_keys:
                rank_zero_debug(
                    f"TransformerBridgeStrategyAdapter: Sample original_model prefixing: {sample_original_model_keys}"
                )
            if sample_conversions:
                rank_zero_debug(f"TransformerBridgeStrategyAdapter: Sample TL conversions: {sample_conversions}")

            return checkpoint

else:
    TransformerBridgeStrategyAdapter = object
