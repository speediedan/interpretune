"""Living prompt-provenance guard (hub design §11.5, umbrella ruling 2026-08-11).

The chat-template-first re-point is gated per model family on BYTE equivalence between the manual
token-spelling definition and the tokenizer's native ``apply_chat_template`` output. This module is
that gate — and it SURVIVES the re-point as a living guard: upstream chat-template drift in a model
revision must fail here, named, rather than silently altering published prompt bytes (#202's concern
carried into the new mechanism).

Measured 2026-08-11:
- gemma-2-2b-it / gemma-3-1b-it: byte-IDENTICAL → re-point eligible.
- Llama-3.2 Instruct: STRUCTURALLY divergent — the native template injects a date-stamped system
  block (``Cutting Knowledge Date / Today Date``), making its output NON-REPRODUCIBLE day-to-day.
  Escalated to the USER per the divergence rule; the manual definition remains the provenance-safe
  path for Llama3, and this guard pins the divergence (both its existence and its shape) so a future
  template change re-opens the decision loudly.
"""

from __future__ import annotations

import os

import pytest


def _tokenizer(model_id: str):
    transformers = pytest.importorskip("transformers")
    token = os.environ.get("HF_GATED_PUBLIC_REPO_AUTH_KEY") or os.environ.get("HF_TOKEN")
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id, token=token)
    except Exception as e:  # gated/offline environments skip rather than fail
        pytest.skip(f"tokenizer for {model_id} unavailable here: {type(e).__name__}")


TASK_PROMPT = "Does the previous passage imply that X? Answer with only one word, either Yes or No."


class TestChatTemplateEquivalence:
    @pytest.mark.parametrize("model_id", ["google/gemma-2-2b-it", "google/gemma-3-1b-it"])
    def test_gemma_manual_spelling_matches_native_template(self, model_id):
        """Byte equivalence that authorized the gemma re-point; drift here = upstream template change."""
        from it_examples.examples.prompt_configs.prompt_configs import GemmaPromptConfig

        tok = _tokenizer(model_id)
        manual = GemmaPromptConfig().model_chat_template_fn(TASK_PROMPT, "gemma-chat")
        templated = tok.apply_chat_template(
            [{"role": "user", "content": TASK_PROMPT}], tokenize=False, add_generation_prompt=True
        )
        assert manual == templated

    def test_llama3_divergence_is_pinned(self):
        """Llama3's native template is date-stamped (non-reproducible) — the divergence itself is the pin.

        If this starts PASSING equivalence (or the divergence changes shape), the Llama3 re-point decision re-opens;
        until then the manual definition is the provenance-safe path.
        """
        from it_examples.examples.prompt_configs.prompt_configs import Llama3PromptConfig

        tok = _tokenizer("meta-llama/Llama-3.2-3B-Instruct")
        manual = Llama3PromptConfig().model_chat_template_fn(TASK_PROMPT, "llama3-chat")
        templated = tok.apply_chat_template(
            [{"role": "user", "content": TASK_PROMPT}], tokenize=False, add_generation_prompt=True
        )
        assert manual != templated, "Llama3 native template now matches manual spelling — re-open the re-point decision"
        assert "Today Date:" in templated, (
            "Llama3 native template no longer date-stamps its system block — re-open the re-point decision"
        )
