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


# Names a model forward commonly takes its token input under; used only to keep the refusal message short.
_INPUT_LIKE = ("input_ids", "input", "inputs_embeds", "tokens")


def verifiable_forward_inputs(model: object) -> set[str] | None:
    """The named input parameters of ``model``'s forward, or None when they cannot be read reliably.

    Only forwards whose call semantics are known are read: a HuggingFace ``PreTrainedModel`` (directly or under a PEFT
    wrapper), whose forward also accepts ``**kwargs`` so an unknown key is swallowed and fails later as a missing
    ``input_ids``, and a TransformerLens bridge. A wrapper whose ``__call__`` dispatches through its own machinery
    (nnsight, circuit-tracer's replacement models) may expose the wrapped model's ``forward`` while accepting other
    keys, so its signature is not evidence and None is returned rather than a guess.
    """
    import inspect

    from transformers import PreTrainedModel

    known = isinstance(model, PreTrainedModel)
    known = known or isinstance(getattr(model, "base_model", None), PreTrainedModel)  # a PEFT wrapper
    known = known or type(model).__module__.startswith("transformer_lens.")
    if not known:
        return None
    forward = getattr(type(model), "forward", None)
    if forward is None:
        return None
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return None
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {name for name, p in params.items() if p.kind in kinds and name != "self"} or None


def forward_input_mismatch(model: object, model_input_names: list[str]) -> str | None:
    """Why ``model`` cannot take batches keyed by ``model_input_names``; None when it can or cannot be checked.

    The analysis ops resolve a batch's input key through aliases, so a mismatch passes every analysis path and fails
    only on a direct forward, which is what every Trainer loop runs: a config can ship broken for training with all of
    its analysis tests green.
    """
    accepted = verifiable_forward_inputs(model)
    if accepted is None or not model_input_names or model_input_names[0] in accepted:
        return None
    takes = sorted(n for n in accepted if n in _INPUT_LIKE) or sorted(accepted)[:4]
    return (
        f"the tokenizer declares {model_input_names[0]!r} as the primary model input "
        f"(tokenizer_kwargs.model_input_names={list(model_input_names)}), but {type(model).__name__}.forward takes "
        f"{takes}; a direct forward (any Trainer loop) would fail. Declare the name this model's forward takes, e.g. "
        f"model_input_names: [{takes[0]!r}, 'attention_mask']."
    )
