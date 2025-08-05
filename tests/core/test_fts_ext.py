# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: move these ruff exceptions for jaxtyping to module level config instead of file level
# ruff: noqa: F722, F821


import torch
from torch import Tensor
from torch.testing import assert_close
import einops
from jaxtyping import Int, Float

from tests.runif import RunIf

class TestClassFTSExtension:


    # TODO: note, this is just a placeholder test for now, it is a copy of a TL test fn that uses attn_only attribution
    # instead of the MLP-inclusive type that we'll actually test with FTS
    @staticmethod
    def logit_attribution(
        embed: Float[Tensor, "seq d_model"],
        l1_results: Float[Tensor, "seq nheads d_model"],
        l2_results: Float[Tensor, "seq nheads d_model"],
        W_U: Float[Tensor, "d_model d_vocab"],
        tokens: Int[Tensor, "seq"]
    ) -> Float[Tensor, "seq-1 n_components"]:
        '''
        Inputs:
            embed: the embeddings of the tokens (i.e. token + position embeddings)
            l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
            l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
            W_U: the unembedding matrix
            tokens: the token ids of the sequence

        Returns:
            Tensor of shape (seq_len-1, n_components)
            represents the concatenation (along dim=-1) of logit attributions from:
                the direct path (seq-1,1)
                layer 0 logits (seq-1, n_heads)
                layer 1 logits (seq-1, n_heads)
            so n_components = 1 + 2*n_heads
        '''
        W_U_correct_tokens = W_U[:, tokens[1:]]
        direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
        l1_attributions = einops.einsum(W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead")
        l2_attributions = einops.einsum(W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead")
        return torch.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)

    # TODO: re-enable this test on windows once a windows-specific OSError ([22]) is resolved.
    @RunIf(skip_windows=True)
    def test_tl_direct_attr(self, get_it_session__tl_cust_mi__setup, tmp_path):
        fixture = get_it_session__tl_cust_mi__setup
        it_session = fixture.it_session
        curr_model = it_session.module.model
        dataloader = it_session.datamodule.test_dataloader()
        # Assert that tokenizer padding_side is 'right' as expected by the test and TLMechInterpCfg
        tokenizer = getattr(it_session.datamodule, 'tokenizer', None)
        assert tokenizer is not None, "Tokenizer not found on datamodule."
        assert getattr(tokenizer, 'padding_side', None) == 'right', (
            f"Expected tokenizer.padding_side == 'right', got '{getattr(tokenizer, 'padding_side', None)}'. "
            "This test expects right-padding to match TLMechInterpCfg. "
        )
        # TODO: update this example to properly test logit attribution in the presence of MLP etc components
        tokens = torch.tensor(dataloader.dataset[0]['input']).unsqueeze(0)
        with torch.inference_mode():
            logits, cache = curr_model.run_with_cache(tokens, remove_batch_dim=True)
            embed = cache["embed"]
            l1_results = cache["result", 0]
            l2_results = cache["result", 1]
            TestClassFTSExtension.logit_attribution(embed, l1_results, l2_results, curr_model.W_U, tokens[0])
            logits[0, torch.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        correct_logits = logits[0, 17:19, 18:20][:,1]  # predicted logits for "lung"/"cancer" respectively at pos 18, 19
        expected_correct_logits = torch.tensor([0.071049094200, 0.685997366905])
        assert_close(expected_correct_logits, correct_logits, atol=1e-3, rtol=0)
