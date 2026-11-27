"""Measure a sparse RTE run's full validation set against a verified selected checkpoint.

The measurement factory must yield an Interpretune module exposing the public head-attribution technique. It must
preserve the selected model's behavior, including active fitted enforcement. The collector checks the reconstructed HF
weights and both answer logits before accepting any attribution row; exporting bare weights is not sufficient.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
import math
import re
from typing import Any, Callable, Iterator, Mapping

from it_examples.utils.sparse_attribution_checkpoint import (
    CheckpointMeasurements,
    VerifiedAttributionCheckpoint,
    tensor_state_sha256,
)
from it_examples.utils.sparse_attribution_collect import head_loading_row, transcoder_basis_id


def _check_selected_subject(module: Any, checkpoint: VerifiedAttributionCheckpoint) -> None:
    state = module.state_dict()
    if not set(checkpoint.state_keys).issubset(state):
        raise ValueError("RTE collection requires checkpoint keys matching the selected module's physical state")
    selected = {name: state[name] for name in checkpoint.state_keys}
    if tensor_state_sha256(selected) != checkpoint.state_sha256:
        raise ValueError("RTE module tensors do not match the verified selected checkpoint")


def _validation_items(module: Any) -> dict[str, Any]:
    import torch

    items = {}
    for item in module.datamodule.dataset["validation"]:
        missing = {"idx", "input_ids", "attention_mask"} - item.keys()
        if missing:
            raise ValueError(f"validation rows must retain required columns: missing {sorted(missing)}")
        identity = item["idx"]
        if type(identity) not in (int, str) or not str(identity) or str(identity) in items:
            raise ValueError(f"validation idx must be a distinct integer or string, got {identity!r}")
        ids, mask = torch.as_tensor(item["input_ids"]), torch.as_tensor(item["attention_mask"])
        if ids.ndim != 1 or ids.dtype != torch.int64 or mask.shape != ids.shape:
            raise ValueError(f"validation row {identity!r} needs one int64 token sequence and matching attention mask")
        if not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError(f"validation row {identity!r} needs a nonempty binary attention mask")
        positions = mask.bool().nonzero().flatten()
        if int(positions[-1] - positions[0]) + 1 != len(positions):
            raise ValueError(f"validation row {identity!r} has internal padding, which cannot be stripped safely")
        items[str(identity)] = ids[mask.bool()].detach().cpu()
    if not items:
        raise ValueError("sparse attribution requires a nonempty validation set")
    return items


@contextmanager
def _reference_inference(module: Any) -> Iterator[None]:
    """Retain every module's mode and the sparse forward's temporary caches, including on a failed measurement."""
    modes = [(child, child.training) for child in module.modules()]
    saved, mask = module._sparse_saved, module._sparse_mask
    module._sparse_saved = {}
    module.eval()
    try:
        yield
    finally:
        module._sparse_saved = saved
        module._sparse_mask = mask
        for child, training in modes:
            child.training = training


def _target_tokens(module: Any) -> tuple[int, int]:
    import torch

    ids = torch.as_tensor(module.it_cfg.entailment_mapping_indices)
    if ids.dtype not in (torch.int32, torch.int64) or ids.shape != (2,):
        raise ValueError("entailment_mapping_indices must contain two integer token ids")
    positive, negative = ids.tolist()
    if min(positive, negative) < 0 or positive == negative:
        raise ValueError("entailment_mapping_indices must contain distinct nonnegative token ids")
    return positive, negative


def _reference_logits(module: Any, items: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    device = next(module.model.parameters()).device
    reference = {}
    with _reference_inference(module), torch.no_grad():
        for row_id, ids in items.items():
            inputs = ids.to(device)[None]
            pair, _ = module.sparse_forward({"input_ids": inputs, "attention_mask": torch.ones_like(inputs)})
            if pair.shape != (1, 2) or not torch.isfinite(pair).all():
                raise ValueError(f"validation row {row_id!r} did not produce two finite answer logits")
            reference[row_id] = pair[0].detach().float().cpu()
            module._sparse_saved.clear()
    return reference


def _decision_target(model: Any, targets: tuple[int, int]) -> Any:
    from circuit_tracer.attribution.targets import CustomTarget

    config = model._module.config.get_text_config()
    if getattr(config, "final_logit_softcapping", None) is not None:
        raise ValueError("RTE logit-gap attribution does not yet support final_logit_softcapping")
    unembed = model._module.get_output_embeddings().weight.detach()
    if unembed.shape != (model.cfg.d_vocab, model.cfg.d_model) or max(targets) >= unembed.shape[0]:
        raise ValueError("RTE answer tokens require a vocabulary-by-model unembedding weight")
    return CustomTarget("logit_gap", 1.0, unembed[targets[0]] - unembed[targets[1]])


def _measurement_hf_state(model: Any) -> dict[str, Any]:
    """Separate the known registered transcoder child from HF weights; the basis is verified independently."""
    from nnsight import Envoy

    basis = model.transcoders
    if isinstance(basis, Envoy):
        basis = basis._module
    if model._module.get_submodule("transcoders") is not basis:
        raise ValueError("measurement model does not register its verified basis under the expected transcoders child")
    state = model._module.state_dict()
    artifact_keys = {f"transcoders.{name}" for name in basis.state_dict()}
    if not artifact_keys or not artifact_keys.issubset(state):
        raise ValueError("measurement model has an incomplete registered transcoder state")
    return {name: value for name, value in state.items() if name not in artifact_keys}


def _measurement_dtype(state: Mapping[str, Any]) -> str:
    """The one floating precision of the measured HF weights, which the attribution edges are computed in."""
    dtypes = {str(tensor.dtype).removeprefix("torch.") for tensor in state.values() if tensor.is_floating_point()}
    if len(dtypes) != 1:
        raise ValueError(f"measurement HF weights must share one floating precision, got {sorted(dtypes)}")
    return dtypes.pop()


def _model_provenance(module: Any) -> tuple[str, str]:
    config = module.it_cfg
    model_id = getattr(config, "model_name_or_path", None)
    loader = getattr(config, "hf_from_pretrained_cfg", None)
    arguments = getattr(loader, "pretrained_kwargs", None)
    revision = arguments.get("revision") if isinstance(arguments, Mapping) else None
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("RTE attribution requires the selected module's HF model_name_or_path")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("RTE attribution requires an immutable 40-character HF-loader revision pin")
    return model_id, revision


class RteSparseAttributionCollector:
    """Collect the selected sparse RTE model's head/source measurements over its complete validation set.

    Args:
        measurement_factory: A callable or ``class_path`` mapping. Called with the training module and verified
            checkpoint, it returns a context manager yielding a module for ``it.head_loadings``. The yielded
            module's replacement model must preserve the selected HF weights and any fitted enforcement. No
            default factory silently replaces an enforced run with its bare weights.
        logit_atol: Absolute tolerance when comparing both reconstructed answer logits with the selected run.
        logit_rtol: Relative tolerance for that comparison. The default is for float32 CPU/GPU measurement.

    Model identity comes from the selected module's HF load configuration, which must pin ``revision`` to a full
    commit hash. It is not inferred from a moving Hub branch or a separate caller-supplied label. The context manager
    owns construction, device placement and cleanup. This collector never switches enforcement
    off, samples the validation set, changes the target, or treats matching version strings as weight evidence.
    """

    def __init__(
        self,
        measurement_factory: Callable[[Any, VerifiedAttributionCheckpoint], AbstractContextManager[Any]]
        | Mapping[str, Any],
        logit_atol: float = 1e-5,
        logit_rtol: float = 1e-5,
    ) -> None:
        if isinstance(measurement_factory, Mapping):
            from interpretune.hub.entrypoints import instantiate_hub_aware_class

            measurement_factory = instantiate_hub_aware_class(dict(measurement_factory))
        if not callable(measurement_factory):
            raise ValueError("measurement_factory must return a context manager for the selected model")
        if any(not math.isfinite(value) or value < 0 for value in (logit_atol, logit_rtol)):
            raise ValueError("logit tolerances must be finite and nonnegative")
        self.measurement_factory = measurement_factory
        self.logit_atol, self.logit_rtol = logit_atol, logit_rtol

    def __call__(self, module: Any, checkpoint: VerifiedAttributionCheckpoint) -> CheckpointMeasurements:
        """Verify the reconstruction before accepting its attribution rows, then recheck its basis and weights."""
        import torch
        from transformers import BatchEncoding
        import interpretune as it
        import interpretune.analysis
        from interpretune.analysis.ops.base import AnalysisBatch

        _check_selected_subject(module, checkpoint)
        model_id, revision = _model_provenance(module)
        items = _validation_items(module)
        targets = _target_tokens(module)
        enforced = module.sparse_enforce
        if type(enforced) is not bool:
            raise ValueError("the selected module must state its boolean sparse_enforce mode")
        weights_sha = tensor_state_sha256(module.model.state_dict())
        reference = _reference_logits(module, items)
        rows = []
        max_logit_error = 0.0
        with self.measurement_factory(module, checkpoint) as measurement:
            model = measurement.replacement_model
            backend = getattr(model, "backend", None)
            if backend != "nnsight":
                raise ValueError(f"RTE checkpoint collection requires the nnsight backend, got {backend!r}")
            basis = transcoder_basis_id(model)
            measured_state = _measurement_hf_state(model)
            if tensor_state_sha256(measured_state) != weights_sha:
                raise ValueError("measurement HF weights do not match the selected training model")
            measurement_dtype = _measurement_dtype(measured_state)
            target = _decision_target(model, targets)
            for row_id, ids in items.items():
                with torch.no_grad():
                    logits, _ = model.get_activations(ids.to(model.device))
                pair = logits[0, -1, list(targets)].detach().float().cpu()
                if not torch.allclose(pair, reference[row_id], atol=self.logit_atol, rtol=self.logit_rtol):
                    raise ValueError(
                        f"validation row {row_id!r}: measurement answer logits {pair.tolist()} do not match selected "
                        f"logits {reference[row_id].tolist()}; preserve fitted enforcement and model configuration"
                    )
                max_logit_error = max(max_logit_error, float((pair - reference[row_id]).abs().max()))
                raw = it.head_loadings(measurement, AnalysisBatch(), BatchEncoding(), 0, prompt=ids, head_target=target)
                rows.append(
                    head_loading_row(
                        raw,
                        row_id=row_id,
                        checkpoint_sha256=checkpoint.sha256,
                        basis_id=basis,
                        decision=float(reference[row_id][0] - reference[row_id][1]),
                        answer_position=len(ids) - 1,
                    )
                )
            if transcoder_basis_id(model) != basis or tensor_state_sha256(_measurement_hf_state(model)) != weights_sha:
                raise ValueError("measurement basis or HF weights changed during attribution")
        if module.sparse_enforce != enforced:
            raise ValueError("the measurement factory changed the selected sparse enforcement mode")
        if _model_provenance(module) != (model_id, revision):
            raise ValueError("the selected model provenance changed during attribution")
        _check_selected_subject(module, checkpoint)
        return CheckpointMeasurements(
            rows,
            list(items),
            basis,
            model_id,
            revision,
            targets,
            measurement_dtype,
            checks={
                "selected_hf_state_sha256": weights_sha,
                "validation_inputs_sha256": tensor_state_sha256(items),
                "max_answer_logit_abs_error": max_logit_error,
                "logit_atol": self.logit_atol,
                "logit_rtol": self.logit_rtol,
                "sparse_enforcement_enabled": enforced,
            },
        )
