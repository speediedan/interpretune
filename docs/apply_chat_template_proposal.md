# Proposal: Use HuggingFace `apply_chat_template` for Prompt Construction

## Status

**Proposed** — Low priority. Current manual prompt config approach works well and provides full control.

## Background

Interpretune currently uses manual prompt config classes (e.g., `Gemma2PromptConfig`, `Gemma3PromptConfig`,
`Llama3PromptConfig`) defined in `src/it_examples/example_prompt_configs.py` to construct chat-formatted prompts
for instruction-tuned models.

These classes define model-specific tokens (`<start_of_turn>`, `<|begin_of_text|>`, etc.) and implement
`model_chat_template_fn()` to wrap task prompts in the appropriate chat template format.

HuggingFace's `transformers` library provides a built-in mechanism for this via
[`tokenizer.apply_chat_template()`](https://huggingface.co/docs/transformers/main/en/chat_templating), which uses
Jinja2 templates embedded in each tokenizer's configuration.

## Current Approach

```python
# Manual prompt config (current)
@dataclass(kw_only=True)
class GemmaPromptConfig:
    B_TURN: str = "<start_of_turn>"
    E_TURN: str = "<end_of_turn>"
    USER_ROLE: str = "user"
    ASSISTANT_ROLE: str = "model"

    def model_chat_template_fn(self, task_prompt, tokenization_pattern=None):
        if tokenization_pattern == "gemma-chat":
            sequence = self.USER_ROLE_START + f"{task_prompt.strip()} {self.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence
```

## Proposed Enhancement

Add an optional `apply_chat_template` mode that delegates to the HuggingFace tokenizer:

```python
# Proposed: delegate to HF tokenizer
def model_chat_template_fn(self, task_prompt, tokenization_pattern=None, tokenizer=None):
    if tokenization_pattern and tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": task_prompt.strip()}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fall back to manual template
    return self._manual_template(task_prompt, tokenization_pattern)
```

## Advantages

1. **Reduced maintenance**: No need to manually track chat template changes across model versions
2. **Broader model support**: Any model with a Jinja2 chat template would work automatically
3. **Consistency**: Uses the same template the model was trained with (embedded in tokenizer config)
4. **System prompts**: HF's API natively supports system prompts, multi-turn conversations, tool use

## Disadvantages / Considerations

1. **Less control**: Manual templates allow fine-grained customization of prompt structure
2. **Tokenizer dependency**: Requires tokenizer to be available at prompt construction time (may not
   always be the case in all code paths)
3. **Template variability**: Different HF model revisions may have different templates, making
   reproducibility harder
4. **Testing overhead**: Need to validate that HF templates produce equivalent outputs to our manual ones
5. **Non-chat models**: Pre-trained (non-IT) models don't have chat templates, so the manual fallback
   is still needed (e.g., `google/gemma-3-1b-pt` has no chat template but `google/gemma-3-1b-it` does)

## Recommendation

Keep manual prompt configs as the default for maximum control and reproducibility. Add `apply_chat_template`
as an opt-in convenience for users who prefer it, particularly for new model families where writing a manual
template is tedious.

Implementation priority is low since:
- We currently support only a handful of model families (GPT-2, Gemma2, Gemma3, Llama3)
- Manual templates are simple and well-tested
- The prompt config pattern is well-established in the codebase

## Related

- `src/it_examples/example_prompt_configs.py` — Current prompt config implementations
- HuggingFace docs: [Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
