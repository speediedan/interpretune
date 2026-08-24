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
