"""Construct head-attribution measurements without dropping the selected RTE run's fitted enforcement."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Iterator

from it_examples.utils.rte_sparse_attribution import _check_selected_subject, _reference_inference
from it_examples.utils.sparse_attribution_checkpoint import VerifiedAttributionCheckpoint, tensor_state_sha256
from it_examples.utils.sparse_attribution_collect import _basis_layer_identity


def _source_parameter(module: Any) -> Any:
    import torch
    from circuit_tracer.utils.tl_nnsight_mapping import get_mapping

    source = module.model
    parameter = next(source.parameters())
    if any(p.dtype != torch.float32 or p.device != parameter.device for p in source.parameters()):
        raise ValueError("RTE head measurement currently requires a single-device float32 selected model")
    architectures = source.config.architectures
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("RTE head measurement requires one explicit HF model architecture")
    get_mapping(architectures[0])
    return parameter


def _full_basis(module: Any, specification: Any) -> Any:
    from circuit_tracer.transcoder import TranscoderSet
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    parameter = next(module.model.parameters())
    if isinstance(specification, str):
        basis, _ = load_transcoder_from_hub(
            specification, device=parameter.device, dtype=parameter.dtype, lazy_encoder=False, lazy_decoder=False
        )
    else:
        basis = deepcopy(specification)
    if type(basis) is not TranscoderSet:
        raise ValueError("RTE head measurement needs a materialized single-layer TranscoderSet")
    expected = set(range(module.model.config.get_text_config().num_hidden_layers))
    if len(basis.transcoders) != len(expected) or len(basis) != len(expected):
        raise ValueError(f"RTE head measurement needs a basis for every model layer: {sorted(expected)}")
    for layer, transcoder in enumerate(basis):
        _basis_layer_identity(transcoder, layer)
    for layer in module.sparse_cfg["enforce_layers"]:
        if _basis_layer_identity(basis[layer], layer) != _basis_layer_identity(module.sparse_tcs[str(layer)], layer):
            raise ValueError(f"measurement basis differs from the fitted enforcement transcoder at layer {layer}")
        if tensor_state_sha256(basis[layer].state_dict()) != tensor_state_sha256(
            module.sparse_tcs[str(layer)].state_dict()
        ):
            raise ValueError(f"measurement basis differs from the fitted enforcement transcoder at layer {layer}")
    return basis


def _clone_hf_model(module: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    source = module.model
    parameter = _source_parameter(module)
    config = deepcopy(source.config)
    config._attn_implementation = "eager"
    # Construction does not copy hooks: the sparse hooks below are deliberately installed in a known order.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        clone = AutoModelForCausalLM.from_config(config)
    clone = clone.to(device=parameter.device, dtype=parameter.dtype)
    clone.load_state_dict(source.state_dict(), strict=True)
    return clone.eval()


def _install_enforcement(module: Any, model: Any, stack: ExitStack) -> None:
    import torch

    def prepare(_model: Any, args: tuple, kwargs: dict) -> None:
        ids = kwargs.get("input_ids", args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError("RTE head measurement requires batched input_ids, including repeated attribution batches")
        mask = kwargs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        if mask.shape != ids.shape or not bool((mask == 1).all()):
            raise ValueError("RTE head measurement expects padding to be stripped before attribution")
        module._sparse_saved.clear()
        module._sparse_mask = mask
        if module.sparse_enforce:
            module.check_template(ids, mask)

    stack.callback(model.register_forward_pre_hook(prepare, with_kwargs=True).remove)
    cfg = module.sparse_cfg
    for layer in sorted(set(cfg["reg_layers"]) | set(cfg["enforce_layers"])):
        for key in ("x", "out"):
            site = model.get_submodule(cfg["sites"][key].format(layer=layer))
            stack.callback(site.register_forward_hook(module._sparse_keep(layer, key)).remove)
    for layer in cfg["enforce_layers"]:
        site = model.get_submodule(cfg["sites"]["out"].format(layer=layer))
        stack.callback(site.register_forward_hook(module._sparse_enforce(layer)).remove)


class RteSparseMeasurementFactory:
    """Yield an analysis component backed by an independent copy of the selected HF weights.

    Args:
        transcoder_set: A circuit-tracer Hub reference or a materialized ``TranscoderSet`` covering every layer.
            Its enforced layers must match the selected run's instrument transcoders exactly.
        zero_positions: Number of leading positions omitted from transcoder features, recorded in the basis id.

    The run's actual sparse hook factories and fitted buffers supply enforcement. Hooks are installed before NNsight
    wraps the model, so its captured MLP outputs include enforcement. Each repeated attribution forward gets its own
    attention mask and template check. Source caches, modes, weights and parameter gradients are retained.

    The selected float32 model and the measurement copy coexist on one device. Callers must budget for both models,
    optimizer state, full transcoders and attribution batches. GPU memory has not been calibrated by this factory.
    """

    def __init__(self, transcoder_set: Any, zero_positions: int = 1) -> None:
        if type(zero_positions) is not int or zero_positions < 0:
            raise ValueError("zero_positions must be a nonnegative integer")
        self.transcoder_set, self.zero_positions = transcoder_set, zero_positions

    @contextmanager
    def __call__(self, module: Any, checkpoint: VerifiedAttributionCheckpoint) -> Iterator[Any]:
        """Build a separate measurement model and release its enforcement hooks even if attribution raises."""
        from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
        from interpretune.adapters.circuit_tracer.backends import DEFAULT_CT_ANALYSIS_BACKEND

        _check_selected_subject(module, checkpoint)
        _source_parameter(module)
        basis = _full_basis(module, self.transcoder_set)
        clone = _clone_hf_model(module)
        with _reference_inference(module), ExitStack() as hooks:
            _install_enforcement(module, clone, hooks)
            replacement = NNSightReplacementModel(
                clone, tokenizer=module.datamodule.tokenizer, task="text-generation", dispatch=True
            )
            replacement.config = clone.config
            replacement._configure_replacement_model(basis)
            replacement.zero_positions = slice(0, self.zero_positions)
            yield SimpleNamespace(replacement_model=replacement, analysis_backend=DEFAULT_CT_ANALYSIS_BACKEND)
        _check_selected_subject(module, checkpoint)
