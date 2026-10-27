import os
from typing import Type, cast, Any
import inspect
from functools import reduce
from copy import deepcopy

import torch
from transformers import PretrainedConfig as HFPretrainedConfig, PreTrainedModel
from transformer_lens.config import TransformerBridgeConfig
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
from interpretune.utils import (
    move_data_to_device,
    rank_zero_warn,
    rank_zero_info,
    rank_zero_debug,
    _resolve_env_auth_token,
)
from interpretune.utils.import_utils import _resolve_dtype
from interpretune.protocol import Adapter


################################################################################
# Mixins to support TransformerLens in different adapter contexts
################################################################################


_BRIDGE_PROCESSED_WEIGHT_PATCH_APPLIED = False


def _ensure_bridge_processed_weight_device_patch() -> None:
    """Patch GeneralizedComponent._apply so processed-weight caches follow device/dtype moves.

    TransformerBridge's set_processed_weights() stores processed tensors as plain attributes (e.g.
    GatedMLPBridge._processed_W_gate), which Module._apply()/.to() do not migrate — so gated-MLP architectures fail
    with a CPU/CUDA device mismatch whenever compatibility mode is enabled before the model moves to device (whether
    via the adapter's own move or Lightning's later one). Registering the tensors as non-persistent buffers is NOT an
    option: GeneralizedComponent.__getattr__ resolves neither _parameters nor _buffers, so register_buffer() makes the
    attribute unreachable. Instead, extend _apply so each component transforms its own cached tensors alongside its
    real parameters. Applies to every bridge regardless of which adapter constructed it (transformer_lens or
    sae_lens). Upstream TransformerLens defect, present in 3.5.1 and still in 4.0.0, whose GeneralizedComponent has
    no ``_apply`` override; gpt2-class MLP bridges are unaffected (no processed-tensor caching on the component).
    """
    global _BRIDGE_PROCESSED_WEIGHT_PATCH_APPLIED
    if _BRIDGE_PROCESSED_WEIGHT_PATCH_APPLIED:
        return
    from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent

    orig_apply = GeneralizedComponent._apply

    def _apply_with_processed_weights(self, fn, recurse: bool = True):
        out = orig_apply(self, fn, recurse)
        for attr, val in list(vars(self).items()):
            if attr.startswith("_processed_") and isinstance(val, torch.Tensor):
                setattr(self, attr, fn(val))
        return out

    GeneralizedComponent._apply = _apply_with_processed_weights
    _BRIDGE_PROCESSED_WEIGHT_PATCH_APPLIED = True


def _place_hf_model_before_wrap(hf_model: torch.nn.Module, pruned_cfg: dict) -> None:
    """Move the HF model to its target device before a ``TransformerBridge`` wraps it.

    The bridge's split q/k/v weights are views of the fused attention weight, which share its storage. Moving the bridge
    afterwards converts each parameter separately and gives every view its own copy (on gpt2, 36 extra 768x768
    storages, about 81 MiB), so the move has to happen while the HF model is the only owner of its storage.
    ``boot_transformers`` loads onto the target device for the same reason.
    """
    if pruned_cfg.get("move_to_device", True) and "device" in pruned_cfg:
        hf_model.to(pruned_cfg["device"])


class TLensAttributeMixin:
    """TransformerLens-aware attribute access: config and device resolution off the wrapped TL model."""

    @property
    def tl_cfg(self) -> TransformerBridgeConfig | None:
        """The wrapped model's TL config, or None (with a warning) before the model exists.

        Returns None rather than raising because this is read during setup and repr paths, where "not constructed yet"
        is an ordinary state rather than an error.
        """
        try:
            cfg = reduce(getattr, "model.cfg".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a TransformerLens config reference (has it been set yet?): {ae}")
            cfg = None
        return cfg  # type: ignore[return-value]

    # TODO: we aren't using IT's Property Composition feature for TLens yet, but might be worth enabling it
    @property
    def device(self) -> torch.device | None:
        """The wrapped TL model's device, or None when it cannot be determined yet."""
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
            device = get_best_available_device(self.tl_cfg)
        except (AttributeError, AssertionError) as ae:
            rank_zero_warn(f"Problem determining appropriate device from TransformerLens config. Received: {ae}")
            device = None
        return device

    @property
    def output_device(self) -> torch.device | None:
        """Device model outputs land on -- for TL, the same device as inputs."""
        return self.get_tl_device()  # provided by mixing class

    @property
    def input_device(self) -> torch.device | None:
        """Device batches must be moved to before a forward pass."""
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

    ``TransformerBridge`` wraps HF models without weight conversion, which is more memory efficient than
    the weight-converting path TransformerLens removed in 4.0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = None

    def build_tunable_view(self) -> Any | None:
        """The TransformerLens-names view over this module's bridge (#758)."""
        from interpretune.adapters.transformer_lens.tunable_view import TransformerBridgeView

        return TransformerBridgeView(self, roles=self.tunable_roles())

    def model_sig_keys(self, target_method: str) -> list:
        """The bridge method's parameter names, without ``labels``.

        Interpretune batches carry TASK labels under ``labels``, while TransformerLens 4.0's bridge ``forward`` accepts
        ``labels`` as language-model targets and passes them to the wrapped model, which then computes a causal-LM loss
        against classification targets and fails on the shape mismatch. Before 4.0 the bridge did not name
        ``labels``, so pruning dropped them; excluding them here keeps that behaviour.
        """
        return [key for key in super().model_sig_keys(target_method) if key != "labels"]

    def auto_model_init(self) -> None:
        """Can be overridden by subclasses to automatically initialize model from a configuration (e.g.
        hf_from_pretrained_cfg, tl_from_config etc.)."""
        if self.it_cfg.hf_from_pretrained_cfg:
            self.hf_pretrained_model_init()
        else:
            self.tl_config_model_init()

    def hf_pretrained_model_init(self) -> None:
        """Initialize from HF pretrained weights, then replace the model with its TL equivalent.

        Only a subset of the usual HF init flow runs, because the HF model is a means to an end here:
        it is converted to a ``TransformerBridge`` and the HF module itself is not what ends up on ``self``.
        """
        # for TL, only a subset of the HF pretrained init flow used since the model is replaced with a
        # TransformerBridge
        access_token = _resolve_env_auth_token(self.it_cfg.os_env_model_auth_key)
        quantization_config = super()._hf_configure_quantization()
        super()._update_hf_pretrained_cfg(quantization_config)
        cust_config, _ = super()._hf_gen_cust_config(access_token)
        self.model = self.hf_configured_model_init(cust_config, access_token)

        self._convert_hf_to_bridge()

    def hf_configured_model_init(
        self, cust_config: HFPretrainedConfig, access_token: str | None = None
    ) -> torch.nn.Module:
        """Instantiate the HF model that TL conversion will consume.

        Kept on CPU rather than the target device: both the HF weights and the TL model would otherwise
        be resident simultaneously, doubling peak memory for a model that is about to be replaced.
        """
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
        except Exception as load_error:
            # A gated-weight failure reaches here as a raw Hub error chain (401/403/404 soup);
            # interpret the three first-run cases by name, and let anything else propagate as before.
            from interpretune.utils.hf_auth_errors import (
                interpret_hf_model_load_error,
                model_id_from_pretrained_kwargs,
            )

            interpreted = interpret_hf_model_load_error(
                load_error,
                model_id=model_id_from_pretrained_kwargs(self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs),
                access_token=access_token,
                auth_env_key=self.it_cfg.os_env_model_auth_key,
            )
            if interpreted is not None:
                raise interpreted from load_error
            raise
        finally:
            if restore_async_load_env:
                del os.environ["HF_DEACTIVATE_ASYNC_LOAD"]
        # perhaps explore initializing on the meta device and then materializing as needed layer by layer during
        # loading/processing into hookedtransformer
        # with torch.device("meta"):
        #     model = self.it_cfg.model_class(config=cust_config)  # token=access_token)
        return model

    def tl_config_model_init(self) -> None:
        """Initialize a TL model from config alone, with no HF pretrained weights.

        ``boot_native`` builds a bridge around a randomly-initialized TL-native model, so config-only
        initialization needs neither an HF model nor a Hub call. ``cfg.init_mode`` and ``cfg.seed``
        control reproducibility.
        """
        # TODO: add note to documentation that we currently require tl_cfg to be not None (either from pretrained or
        #       custom config) based, so model_init will not be used. To fully customize TL behavior, override this
        #       method and init a config-based bridge as desired
        # TODO: suppress messages from tl about no tokenizer here, we're deferring the tokenizer attach until setup
        self.model = self._boot_native_bridge()

    def _boot_native_bridge(self) -> TransformerBridge:
        """Build a config-only bridge that honors the config the way ``HookedTransformer(cfg)`` did.

        Shared by every adapter with a config-only path, so they cannot drift. ``boot_native`` leaves three things
        to the caller that ``HookedTransformer`` did itself: the ``-1`` vocab sentinels, placement (it moves the
        model only when passed ``device``/``dtype``, ignoring ``cfg.device``, so a cuda config silently built a CPU
        model), and the hook-exposing flags.
        """
        tl_cfg = self.it_cfg.tl_cfg
        cfg = tl_cfg.cfg
        self._resolve_vocab_sentinels(cfg, self.it_cfg.tokenizer)
        # TransformerLens 4.0 attaches `boot_native` when `transformer_lens` imports its native source module, so it
        # exists at runtime but not on the class pyright reads.
        bridge = TransformerBridge.boot_native(  # pyright: ignore[reportAttributeAccessIssue]  # attached at runtime by the native source module
            cfg,
            tokenizer=self.it_cfg.tokenizer,
            device=getattr(cfg, "device", None) if getattr(tl_cfg, "move_to_device", True) else None,
            dtype=getattr(cfg, "dtype", None),
        )
        self._apply_bridge_hook_flags(bridge, cfg)
        return bridge

    # Config flags that change which hook points a bridge exposes. `boot_native` accepts them on the config but
    # does not propagate them to the attention/MLP components, so `use_attn_result=True` built a model with no
    # `hook_result` and nothing said so. Each has a bridge setter that propagates it, or raises naming the flag when
    # the architecture cannot provide it (the TL-native attention cannot provide per-head results, for instance).
    _BRIDGE_HOOK_FLAGS = ("use_attn_result", "use_split_qkv_input", "use_attn_in", "use_hook_mlp_in")

    @classmethod
    def _apply_bridge_hook_flags(cls, bridge, cfg) -> None:
        """Route hook-exposing config flags through the bridge's setters so each takes effect or is refused."""
        for flag in cls._BRIDGE_HOOK_FLAGS:
            if getattr(cfg, flag, False):
                getattr(bridge, f"set_{flag}")(True)

    @staticmethod
    def _resolve_vocab_sentinels(cfg, tokenizer) -> None:
        """Resolve TL's ``-1`` vocab sentinels from the tokenizer, which config-only init no longer does for us.

        ``HookedTransformer.__init__`` inferred these (``d_vocab = max(tokenizer.vocab.values()) + 1``, and
        ``d_vocab_out`` following it). ``boot_native`` does not: ``NativeModel`` passes ``cfg.d_vocab`` straight
        into ``nn.Embedding``, so a config that omitted ``d_vocab`` and relied on inference now fails with
        ``RuntimeError: Trying to create tensor with negative dimension -1``. The arithmetic below is upstream's,
        copied deliberately so a config that worked before produces the same embedding size.
        """
        if tokenizer is None or getattr(tokenizer, "vocab", None) is None:
            return
        if getattr(cfg, "d_vocab", None) == -1:
            cfg.d_vocab = max(tokenizer.vocab.values()) + 1
        if getattr(cfg, "d_vocab_out", None) == -1:
            cfg.d_vocab_out = cfg.d_vocab

    def _prune_tl_cfg_dict(self, normalize_device: bool = False, prune_list: list | None = None) -> dict:
        """Prunes the tl_cfg dictionary by removing IT-specific and HF-specific keys that shouldn't be passed to
        the ``TransformerBridge`` constructor.

        Returns:
            dict: The pruned dictionary
        """
        prune_list = prune_list or ["hf_model", "tokenizer"]
        pruned_dict = deepcopy(self.it_cfg.tl_cfg.__dict__)

        for key in prune_list:
            if key in pruned_dict:
                if pruned_dict[key] is not None:
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

    def _convert_hf_to_bridge(self) -> None:
        """Convert HF model to TransformerBridge (v3 architecture).

        TransformerBridge wraps the HF model without weight conversion, providing memory efficiency and better HF
        ecosystem compatibility.

        If the tl_cfg is an ITLensBridgeConfig, this method will:
        - Apply any transformer_bridge_config_overrides to the TransformerBridgeConfig
        - Call enable_compatibility_mode() with the specified kwargs if enable_compatibility_mode=True
        """
        from interpretune.adapters.transformer_lens.config import ITLensBridgeConfig

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
        _place_hf_model_before_wrap(hf_model, pruned_cfg)
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
            _ensure_bridge_processed_weight_device_patch()

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
        2. TL Model Config: Actual TransformerLens config (TransformerBridgeConfig)
        3. IT TL Config: Interpretune-specific settings (ITLensFromPretrainedConfig or ITLensCustomConfig)
        """
        # Override unsupported from pretrained options
        if self.hf_cfg:
            self.hf_cfg.lora_cfg = None  # type: ignore[assignment]  # config flexibility
            # TODO NEXT: enable bnb now that it's supported by TransformerLens
            self.hf_cfg.bitsandbytesconfig = None  # type: ignore[assignment]  # config flexibility
        # TODO: refactor the captured config here to only add tl_from_pretrained, other added in superclass
        # Serialize the actual TransformerLens config from the initialized model
        tl_model_cfg = self._make_config_serializable(self.model.cfg, ["device", "dtype"])

        self._it_state._init_hparams.update({"tl_model_cfg": tl_model_cfg})

        # Serialize IT-specific TransformerLens settings
        self._it_state._init_hparams.update({"it_tl_cfg": self.it_cfg.tl_cfg})

        # Call superclass to serialize HF PretrainedConfig (hf_preconversion_config)
        super()._capture_hyperparameters()

    def it_generate(self, batch: BatchEncoding | torch.Tensor, **kwargs) -> Any:
        """Generate from a batch, refusing tensor batches the bridge loop cannot start from.

        Defined here rather than on one composition because every TransformerLens-backed module (core,
        Lightning, and the SAE Lens and circuit-tracer TL backends) runs the same bridge generation loop. That
        loop samples each step from the last sequence position, so a row ending in padding continues from
        post-pad logits: its continuation is conditioned on the padding (or ends at once when end-of-text is
        predicted), never on the prompt. Batched string lists are safe (the bridge left-pads those itself);
        right-padded tensors are not, so they are refused by name instead of generating plausible output.
        """
        if isinstance(batch, torch.Tensor):
            tokenizer = getattr(getattr(self, "datamodule", None), "tokenizer", None)
            _reject_trailing_pad_tensor_batch(
                batch,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                padding_side=getattr(tokenizer, "padding_side", None),
            )
        return super().it_generate(batch, **kwargs)

    def set_input_require_grads(self) -> None:
        """Not supported for TL modules; logs and returns rather than raising."""
        # not currently supported by ITLensModule
        rank_zero_info("Setting input require grads not currently supported by ITLensModule.")


################################################################################
# TransformerLens Module Composition
################################################################################


class TransformerLensAdapter(TLensAttributeMixin):
    """Adapter composing interpretune modules with TransformerLens execution."""

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        """Register the TransformerLens datamodule and module compositions."""
        # Registered rather than discovered by import path, so a bundled adapter and a hub-delivered one
        # reach auto-composition through the same seam.
        from interpretune.adapters.transformer_lens.config import ITLensConfig

        adapter_ctx_registry.register_module_cfg_class(Adapter.transformer_lens, ITLensConfig)
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.transformer_lens),
            composition_classes=(ITDataModule,),
            description="TransformerLens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.transformer_lens),
            composition_classes=(ITDataModule, LightningDataModule),
            description="TransformerLens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.transformer_lens),
            composition_classes=(ITLensModule,),
            description="TransformerLens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(
            Adapter.transformer_lens,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.transformer_lens),
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
        """Move a batch to the TL input device when one is resolvable; a no-op otherwise."""
        device = self.input_device
        if device is not None:
            move_data_to_device(batch, device)
        return batch


class ITLensModule(TransformerLensAdapter, CoreHelperAttributes, BaseITLensModule):
    """The TransformerLens module composition."""


def _reject_trailing_pad_tensor_batch(
    input_ids: torch.Tensor, *, pad_token_id: int | None, padding_side: str | None
) -> None:
    """Refuse a tensor batch whose rows end in padding instead of generating from it.

    A trailing pad id means the bridge loop's first step reads post-pad logits for that row, so its
    continuation is conditioned on padding rather than on the prompt. Left-pad the
    batch (``padding_side="left"``) or pass strings so the bridge pads the batch itself.
    Batches with no trailing pads, and callers with no pad id to check against, pass through.
    """
    if pad_token_id is None or input_ids.dim() == 0:
        return
    if bool((input_ids[..., -1] == pad_token_id).any()):
        raise ValueError(
            "it_generate received a tensor batch with trailing pad token id "
            f"{pad_token_id} (tokenizer padding_side={padding_side!r}); the TransformerBridge "
            "generation loop samples from the last sequence position, so padded rows continue "
            "from post-pad logits rather than from their prompts. Left-pad "
            'the batch (padding_side="left") or pass a list of strings instead.'
        )
