# Prompt Parity Migration Guide

## Scope

This note documents the current migration path for Gemma instruction-tuned prompt handling in the concept-direction parity harness.

The goals are:

1. Keep a Hugging Face `apply_chat_template` path as the canonical reference.
2. Keep a dataclass-based path for cases where we need explicit string construction.
3. Verify that both paths are token-for-token identical before using them interchangeably.
4. Separate prompt-template parity from task-framing effects at the assistant response boundary.

## Code Changes

The shared Gemma prompt config in [src/it_examples/example_prompt_configs.py](/home/speediedan/repos/interpretune/src/it_examples/example_prompt_configs.py) now does two things:

1. Builds the manual Gemma chat string with the correct BOS token and turn separators.
2. Exposes `apply_chat_template_fn(...)` so the harness can compare the manual path against the tokenizer-native path directly.

The concept-direction harness utilities in [tests/concept_direction_approach_parity/concept_direction_experiment_utils.py](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/concept_direction_experiment_utils.py) now:

1. Use explicit `apply_chat_template` instead of the older `tokenizer` mode name.
2. Allow `TARGET_TOKENS` to override `TARGET_TOKEN_IDS`.
3. Emit prompt strings, token IDs, and token strings for each render mode.
4. Record pre-intervention and post-intervention top-token snapshots for easier debugging.

The source notebook [tests/concept_direction_approach_parity/concept_direction_experiment_harness.ipynb](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/concept_direction_experiment_harness.ipynb) was also cleaned up to remove the whitespace-heavy cell formatting introduced by earlier edits.

## Token-By-Token Validation

Validated locally with:

```bash
cd /home/speediedan/repos/interpretune
/mnt/cache/speediedan/.venvs/it_latest/bin/python - <<'PY'
from transformers import AutoTokenizer
from src.it_examples.example_prompt_configs import GemmaPromptConfig

prompt = "Fact: the capital of the state containing Dallas is"
cfg = GemmaPromptConfig()

for model_name in ["google/gemma-2-2b-it", "google/gemma-3-1b-it"]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    apply_str = cfg.apply_chat_template_fn(tokenizer, prompt, tokenize=False, add_generation_prompt=True)
    dataclass_str = cfg.model_chat_template_fn(prompt, tokenization_pattern="gemma-chat")
    apply_ids = tokenizer(apply_str, add_special_tokens=False)["input_ids"]
    dataclass_ids = tokenizer(dataclass_str, add_special_tokens=False)["input_ids"]
    print(model_name, apply_str == dataclass_str, apply_ids == dataclass_ids)
PY
```

Observed results:

1. Gemma 2 IT: string parity `True`, token-ID parity `True`
2. Gemma 3 IT: string parity `True`, token-ID parity `True`

The normalized render for both instruction-tuned Gemma models is:

```text
<bos><start_of_turn>user
Fact: the capital of the state containing Dallas is<end_of_turn>
<start_of_turn>model
```

## What Was Wrong Before

The earlier manual dataclass rendering diverged from Hugging Face `apply_chat_template` in three important ways:

1. It omitted `<bos>`.
2. It inserted an extra space before `<end_of_turn>`.
3. It omitted the newline between `<end_of_turn>` and `<start_of_turn>model`.

That mismatch required a notebook-local BOS workaround. The workaround is no longer the right fix once the shared prompt config is correct.

## Important Finding: Template Parity Was Not The Main IT Failure

After template parity was fixed, the Gemma IT models still did not behave like the base continuation setup for the raw multihop prompt.

Validated next-token probe for Gemma 2 IT:

1. Plain continuation prompt `Fact: the capital of the state containing Dallas is` keeps ` Austin` on top.
2. Chat-formatted prompt with a fresh assistant turn makes `The` the top next token.

Observed Gemma 2 IT top-token behavior:

1. Plain prompt top token: ` Austin`
2. Chat prompt top token: `The`

Observed Gemma 3 IT top-token behavior:

1. Chat prompt top token: `The`
2. `Austin` remains near the top, but is not the highest-probability first token without extra task framing.

This means the remaining issue is task framing, not template corruption.

## Effective IT Prompt Framing

For Gemma 2 IT and Gemma 3 IT, explicitly constraining the assistant response changes the top token back to `Austin`.

Example that works much better:

```text
Answer with only the missing city name. Fact: the capital of the state containing Dallas is
```

Observed results:

1. Gemma 2 IT top token becomes `Austin`
2. Gemma 3 IT top token becomes `Austin`

## Migration Guidance

Use this rule set going forward:

1. Use `apply_chat_template` as the reference behavior for chat models.
2. Only use manual dataclass rendering if we also validate string parity and token-ID parity against `apply_chat_template`.
3. Do not use a notebook-local BOS-prefix patch for Gemma chat prompts.
4. Treat assistant-boundary task framing as a separate variable from chat-template correctness.
5. For instruction-tuned multihop completion experiments, prefer prompts that constrain the assistant to emit the answer token directly if the evaluation depends on the first generated token being the city name.

The harness now follows that rule for the `capitals_states` concept pair:

1. plain runs keep `Fact: the capital of the state containing Dallas is`
2. chat-rendered runs default to `Answer with only the missing city name. Fact: the capital of the state containing Dallas is`

## Immediate Follow-Up

With the default prompt split now in place, the remaining follow-up is to rerun the IT notebook/config flows and confirm the intervention behavior under the corrected first-token framing.
