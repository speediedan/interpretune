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


def verifiable_forward_inputs(model: object) -> frozenset[str] | None:
    """The named input parameters of ``model``'s forward, or None when they cannot be read reliably.

    Only forwards whose call semantics are known are read: a HuggingFace ``PreTrainedModel`` (directly or under a PEFT
    wrapper), whose forward also accepts ``**kwargs`` so an unknown key is swallowed and fails later as a missing
    ``input_ids``, and a TransformerLens bridge. A wrapper whose ``__call__`` dispatches through its own machinery
    (nnsight, circuit-tracer's replacement models) may expose the wrapped model's ``forward`` while accepting other
    keys, so its signature is not evidence and None is returned rather than a guess.
    """
    from transformers import PreTrainedModel

    known = isinstance(model, PreTrainedModel)
    known = known or isinstance(getattr(model, "base_model", None), PreTrainedModel)  # a PEFT wrapper
    known = known or type(model).__module__.startswith("transformer_lens.")
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
