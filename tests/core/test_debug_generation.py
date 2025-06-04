from unittest.mock import patch
import re

import pytest
from torch.testing import assert_close

from tests.runif import RunIf
from interpretune.extensions import DebugGeneration

IT_TEST_TEXT = {
    "text": [("Interpretune is a flexible ML experimentation framework that makes code adhering to a simple"
             " protocol compatible with a wide variety of frameworks and research packages. Most of the adapters"
             " currently available are for ML interpretability research packages but the defined protocol should"
             " in principle accommodate adapters written for a vast range of frameworks, applications and research"
             " domains.")]}

class TestClassDebugGen:

    TEST_DEBUG_SEQS = ["Hello, I'm a large language,", "The day after Tuesday"]

    @pytest.mark.usefixtures("make_deterministic")
    def test_debug_session_top1(self, get_it_session__tl_gpt2_debug__setup):
        fixture = get_it_session__tl_gpt2_debug__setup
        debug_module = fixture.it_session.module.debug_lm
        EXPECTED_DEBUG_BASIC = (0.25, 16)
        acc, correct_tokens = debug_module.top1_token_accuracy_on_sample(IT_TEST_TEXT["text"][0])
        assert_close(actual=acc.cpu().item(), expected=EXPECTED_DEBUG_BASIC[0], rtol=0.10, atol=0)
        assert_close(actual=len(correct_tokens), expected=EXPECTED_DEBUG_BASIC[1], rtol=0.10, atol=0)

    # TODO: extract model-specific debug pattern logic from DebugGeneration to a pattern dispatcher fn/class
    @pytest.mark.parametrize(
        "session_fixture, format, pad_token, gen_kwargs, batch_mode, expected",
        [
            pytest.param("get_it_session__l_gemma2_debug__setup", "gemma2-chat", "<pad>",
                         {"gen_config_override": {"max_new_tokens": 4, "pad_token_id": 0},
                          "decode_cfg_override": {"skip_special_tokens": False}}, True, (2, True),
                          marks=RunIf(standalone=True, lightning=True, bf16_cuda=True)),
            pytest.param("get_it_session__l_llama3_debug__setup", None, "<|finetune_right_pad_id|>",
                         {"gen_config_override": {"max_new_tokens": 4, "pad_token_id": 128004},
                          "decode_cfg_override": {"skip_special_tokens": False}}, True, (2, True),
                          marks=RunIf(standalone=True, lightning=True, bf16_cuda=True)),
            pytest.param("get_it_session__l_gemma2_debug__setup", None, "<pad>",
                         {"gen_config_override": {"max_new_tokens": 4, "pad_token_id": 0}}, False, (2, False),
                         marks=RunIf(optional=True, lightning=True, bf16_cuda=True)),
            pytest.param("get_it_session__l_llama3_debug__setup", "llama3-chat", "<|finetune_right_pad_id|>",
                         {"gen_config_override": {"max_new_tokens": 4, "pad_token_id": 128004}}, False, (2, False),
                         marks=RunIf(optional=True, lightning=True, bf16_cuda=True)),
        ],
        ids=["gemma2_decode_override_no_skip_special_batch", "llama3_decode_override_no_skip_special_batch",
             "gemma2_default_serial", "llama3_default_serial",],
    )
    def test_debug_session_chat(self, request, session_fixture, format, pad_token, gen_kwargs, batch_mode, expected):
        it_session_fixture = request.getfixturevalue(session_fixture)
        it_module = it_session_fixture.it_session.module
        debug_module = it_module.debug_lm
        test_seqs = debug_module.chat_debug_sequences(format=format, sequences=self.TEST_DEBUG_SEQS)
        debug_gen_fn = debug_module.debug_generate_batch if batch_mode else debug_module.debug_generate_serial
        answers, full_outputs = debug_gen_fn(test_seqs, **gen_kwargs)
        pad_token_id = gen_kwargs.get("gen_config_override", {}).get("pad_token_id", 0)
        if batch_mode:
            inspect_tokens = full_outputs.sequences[1]
            assert inspect_tokens[2].item() == pad_token_id
        else:
            inspect_tokens = full_outputs[1].sequences
            inspect_tokens = inspect_tokens.squeeze()
            assert inspect_tokens[2].item() != pad_token_id
        padpat = re.compile(r".*" + pad_token + ".*")
        pad_included = True if padpat.match(answers[1]) else False
        assert len(answers) == expected[0]
        assert pad_included == expected[1]
        m_prefix = "composing TestITModule with: \n  - LightningAdapter\n  - BaseITModule"
        assert m_prefix in repr(it_module)

    def test_debug_generation_chat_invalid_format(self):
        with pytest.warns(UserWarning, match=r"Failed to generate chat.*"):
            debug_gen = DebugGeneration()
            stripped_seqs = debug_gen.chat_debug_sequences(format="invalid_format", sequences=["test A ", " test B"])
            assert stripped_seqs == ["test A", "test B"]

    @pytest.mark.parametrize(
        "gen_kwargs, batch_mode, expected",
        [
            pytest.param({"gen_kwargs_override": {"max_new_tokens": 4}}, True, (2, False)),
            pytest.param({"gen_kwargs_override": {"max_new_tokens": 4}, "gen_output_attr": "tokens"}, True, (2, False)),
            pytest.param({"gen_kwargs_override": {"max_new_tokens": 4},
                          "decode_cfg_override": {"skip_special_tokens": False}}, True, (2, True)),
            pytest.param({"gen_kwargs_override": {"max_new_tokens": 4}}, False, (2, False)),
            pytest.param({"gen_kwargs_override": {"max_new_tokens": 4},
                          "gen_output_attr": "tokens"}, False, (2, False)),
            pytest.param({"gen_kwargs_override": {"max_new_tokens": 4},
                          "decode_cfg_override": {"skip_special_tokens": False}}, False,
                         (2, False)),
        ],
        ids=["default_batch", "cust_gen_output_attr_batch", "decode_override_no_skip_special_batch",
             "default_serial", "cust_gen_output_attr_serial", "decode_override_no_skip_special_serial"],
    )
    def test_debug_session_debug_gen(self, get_it_session__tl_gpt2_debug__setup, gen_kwargs, batch_mode, expected):
        fixture = get_it_session__tl_gpt2_debug__setup
        debug_module = fixture.it_session.module.debug_lm
        test_seqs = debug_module.debug_sequences(self.TEST_DEBUG_SEQS)
        debug_gen_fn = debug_module.debug_generate_batch if batch_mode else debug_module.debug_generate_serial
        answers, full_outputs = debug_gen_fn(test_seqs, **gen_kwargs)
        if batch_mode:
            inspect_tokens = full_outputs.tokens[1]
            assert inspect_tokens[3].item() == 50256
        else:
            inspect_tokens = full_outputs[1].tokens
            inspect_tokens = inspect_tokens.squeeze()
            assert inspect_tokens[3].item() != 50256
        padpat = re.compile(r".*\|endoftext\|.*")
        pad_included = True if padpat.match(answers[1]) else False
        assert len(answers) == expected[0]
        assert pad_included == expected[1]

    @pytest.mark.parametrize(
        "gen_kwargs, expected",
        [
        pytest.param({"gen_config_override": {"max_new_tokens": 4}, "gen_output_attr": "tokens"}, (1, False)),
        ],
        ids=["cust_gen_output_attr_serial_config_str"],
    )
    def test_debug_session_debug_gen_str(self, get_it_session__tl_gpt2_debug__setup, gen_kwargs, expected):
        fixture = get_it_session__tl_gpt2_debug__setup
        debug_module = fixture.it_session.module.debug_lm
        test_sequence = self.TEST_DEBUG_SEQS[1]
        test_seqs = debug_module.debug_sequences(test_sequence)
        answers, full_outputs = debug_module.debug_generate_serial(test_seqs, **gen_kwargs)
        inspect_tokens = full_outputs[0].tokens
        inspect_tokens = inspect_tokens.squeeze()
        assert inspect_tokens[4].item() != 50256
        padpat = re.compile(r".*\|endoftext\|.*")
        pad_included = True if padpat.match(answers[0]) else False
        assert len(answers) == expected[0]
        assert pad_included == expected[1]

    @pytest.mark.parametrize(
        "gen_kwargs, gen_error",
        [pytest.param({"gen_config_override": {"max_new_tokens": 4}}, "No compatible default"),],
        ids=["gen_output_attr_error"],
    )
    def test_debug_session_exceptions(self, get_it_session__tl_gpt2_debug__setup, gen_kwargs, gen_error):
        fixture = get_it_session__tl_gpt2_debug__setup
        debug_module = fixture.it_session.module.debug_lm
        test_seqs = debug_module.debug_sequences(["Hello, I'm a large language,", "The day after Tuesday"])
        if gen_error:
            from interpretune.extensions import DebugGeneration
            with patch.object(DebugGeneration, "DEFAULT_OUTPUT_ATTRS", ("fake",)):
                with pytest.raises(ValueError, match=gen_error):
                    _ = debug_module.debug_generate_batch(test_seqs, **gen_kwargs)

    @pytest.mark.usefixtures("make_deterministic")
    @pytest.mark.parametrize("default, limit_chars, stride, expected", [(True, 8192, 512, 27.3), (False, 900, 8, 74.2)],
                             ids=["wikitext", "custom"])
    def test_debug_session_perplexity(self, get_it_session__tl_gpt2_debug__setup, default, limit_chars, stride,
                                      expected):
        fixture = get_it_session__tl_gpt2_debug__setup
        debug_module = fixture.it_session.module.debug_lm
        corpus = IT_TEST_TEXT if not default else None
        ppl = debug_module.perplexity_on_sample(corpus, limit_chars=limit_chars, stride=stride)
        assert_close(actual=ppl.cpu().item(), expected=expected, rtol=0.03, atol=0)
