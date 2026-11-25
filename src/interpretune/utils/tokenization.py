from __future__ import annotations

import functools
import inspect
from typing import Any, Mapping

from transformers.tokenization_utils_base import BatchEncoding


DEFAULT_DECODE_KWARGS = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}


def sanitize_input_name(model_input_names: list[str], features: BatchEncoding) -> BatchEncoding:
    """Rename the ``input_ids`` key to the tokenizer's configured primary input name, in place.

    Some HuggingFace code paths hardcode ``input_ids`` regardless of ``model_input_names``, so a
    tokenizer configured for a different primary name (TransformerLens conventionally uses ``input``)
    still emits ``input_ids``. This re-keys the encoding so downstream code can rely on the configured
    name. A no-op when the primary name already is ``input_ids``.
    """
    # HF hardcodes the example input name in some contexts:  https://bit.ly/hf_input_ids_hardcode
    if (primary_input := model_input_names[0]) != "input_ids":
        features[primary_input] = features["input_ids"]
        del features["input_ids"]
    return features


# Names a model forward takes its token input under. A keyword call whose input-like keys include none the forward names
# cannot supply the model's primary input.
_INPUT_LIKE = frozenset({"input_ids", "input", "inputs_embeds", "tokens"})


@functools.lru_cache(maxsize=None)
def _named_forward_params(cls: type) -> frozenset[str] | None:
    forward = getattr(cls, "forward", None)
    if forward is None:
        return None
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return None
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return frozenset(name for name, p in params.items() if p.kind in kinds and name != "self") or None


def _derives_from(model: object, package: str) -> bool:
    """Whether a class of ``model``'s MRO is defined in ``package``, without importing it."""
    return any(cls.__module__.partition(".")[0] == package for cls in type(model).__mro__)


def verifiable_forward_inputs(model: object) -> frozenset[str] | None:
    """The named input parameters of ``model``'s forward, or None when they cannot be read reliably.

    Only forwards whose call semantics are known are read: a HuggingFace ``PreTrainedModel`` (directly or under a PEFT
    wrapper), whose forward also accepts ``**kwargs`` so an unknown key is swallowed and fails later as a missing
    ``input_ids``; a TransformerLens bridge or a subclass of one; and an nnsight wrapper (circuit-tracer's nnsight
    replacement model is one), which outside a trace calls the module it wraps with the keywords it was given, so the
    wrapped module's forward is the contract. Any other wrapper's signature is not evidence, and None is returned rather
    than a guess.
    """
    from transformers import PreTrainedModel

    if _derives_from(model, "nnsight"):
        # a subclass with a `__call__` of its own may rename inputs, so only nnsight's own call is vouched for
        passes_through = type(model).__call__.__module__.partition(".")[0] == "nnsight"  # type: ignore[operator]
        wrapped = vars(model).get("_module")
        return verifiable_forward_inputs(wrapped) if passes_through and wrapped is not None else None
    if _derives_from(model, "peft"):
        # `base_model` is the tuner that holds the wrapped model, not the model itself
        base = getattr(model, "get_base_model", None)
        return verifiable_forward_inputs(base()) if base is not None else None
    known = isinstance(model, PreTrainedModel) or _derives_from(model, "transformer_lens")
    return _named_forward_params(type(model)) if known else None


def forward_call_mismatch(model: object, kwargs: Mapping[str, Any]) -> str | None:
    """Why a keyword-only forward call on ``model`` cannot supply its primary input; None when it can or cannot be
    told.

    The analysis ops resolve a batch's input key through aliases, so a config whose tokenizer names the wrong key passes
    every analysis path and fails only on a direct forward, which is what every Trainer loop runs, with the model's own
    message about a missing input.
    """
    given = sorted(k for k in kwargs if k in _INPUT_LIKE)
    if not given:
        return None
    accepted = verifiable_forward_inputs(model)
    if accepted is None or any(k in accepted for k in given):
        return None
    takes = sorted(_INPUT_LIKE & accepted) or sorted(accepted)[:4]
    return (
        f"the batch supplies its input as {given}, but {type(model).__name__}.forward takes {takes}. A batch's input "
        f"key comes from the tokenizer's model_input_names, so declare the name this forward takes, e.g. "
        f"tokenizer_kwargs.model_input_names: [{takes[0]!r}, 'attention_mask']."
    )
