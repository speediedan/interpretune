"""#708's block-sparse featurizers as the #866 example loads them: the encoder's block gate, the decoder, the
binding."""

import pytest
import torch

from it_examples.utils.bsf_featurizers import BlockSparseFeaturizer, _Tiny, block_topk


def test_block_topk_keeps_each_rows_largest_blocks():
    norms = torch.tensor([[0.1, 3.0, 2.0, 0.5], [4.0, 0.0, 1.0, 5.0]])
    torch.testing.assert_close(block_topk(norms, 2), torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]))


def test_encode_gates_to_l0_blocks_and_decodes_at_the_input_scale():
    tiny = _Tiny()
    state = tiny.state()
    bsf = BlockSparseFeaturizer(state)
    x = torch.randn(5, tiny.d)
    codes = bsf.encode(x)
    assert codes.shape == (5, tiny.n_groups, tiny.group_size)
    assert ((codes.norm(dim=-1) > 0).sum(-1) == tiny.l0).all()
    # the reference computation, written out from the checkpoint
    w = state["state_dict"]
    a = ((x - state["mean"]) / state["scale"]) @ w["W_enc"] + w["b_enc"]
    a = a.view(5, tiny.n_groups, tiny.group_size)
    a = a * block_topk(a.norm(dim=-1), tiny.l0).unsqueeze(-1)
    expected = (a.flatten(-2) @ w["W_dec"]) * state["scale"] + state["mean"]
    torch.testing.assert_close(bsf(x), expected)
    torch.testing.assert_close(bsf.block(1), w["W_dec"][3:6])


def test_a_mismatched_checkpoint_is_refused():
    state = _Tiny().state()
    state["n_groups"] = 5
    with pytest.raises(ValueError, match="latent coordinates"):
        BlockSparseFeaturizer(state)


def test_the_binding_addresses_one_block_per_group():
    pytest.importorskip("venngram")
    bsf = BlockSparseFeaturizer(_Tiny().state())
    binding = bsf.binding("bsf")
    assert binding.n_blocks == 6 and binding.block_size == 3 and binding.layer == 3
