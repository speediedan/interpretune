from transformers.tokenization_utils_base import BatchEncoding


DEFAULT_DECODE_KWARGS = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}

def _sanitize_input_name(model_input_names: list[str], features: BatchEncoding) -> None:
    # HF hardcodes the example input name in some contexts:  https://bit.ly/hf_input_ids_hardcode
    if (primary_input := model_input_names[0]) != "input_ids":
        features[primary_input] = features["input_ids"]
        del features["input_ids"]
    return features
