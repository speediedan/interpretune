"""#708's block-sparse featurizers as the #866 example loads them: the encoder's block gate, the decoder, the
binding."""

import pytest
import torch

from it_examples.utils.bsf_featurizers import BlockSparseFeaturizer, KeptError, _Tiny, block_topk


def test_block_topk_keeps_each_rows_largest_blocks():
    norms = torch.tensor([[0.1, 3.0, 2.0, 0.5], [4.0, 0.0, 1.0, 5.0]])
    torch.testing.assert_close(block_topk(norms, 2), torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]))


def test_encode_gates_to_l0_blocks_and_decodes_at_the_input_scale():
    tiny = _Tiny()
    state = tiny.state()
    bsf = BlockSparseFeaturizer(state)
    x = torch.randn(5, tiny.d)
    codes = bsf.block_codes(x)
    assert codes.shape == (5, tiny.n_groups, tiny.group_size)
    assert ((codes.norm(dim=-1) > 0).sum(-1) == tiny.l0).all()
    torch.testing.assert_close(bsf.encode(x), codes.flatten(-2))  # the latents a backend carries are flat
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


def test_the_decision_book_names_one_block_per_set_and_loads():
    pytest.importorskip("venngram")
    from venngram.book import AddressBook

    from it_examples.utils.bsf_featurizers import decision_book

    book = decision_book(model_revision="0" * 40)
    assert book["sets"]["resid_yes"] == {"features": "bsf_resid", "layers": [17], "units": [327]}
    assert book["sets"]["jspace_no"]["space"] == "jlens"
    loaded = AddressBook.from_mapping(book)
    assert sorted(loaded.sets()) == ["jspace_no", "jspace_yes", "resid_no", "resid_yes"]


def test_an_attached_featurizer_declares_its_hook_and_writes_the_models_dtype():
    bsf = BlockSparseFeaturizer(_Tiny().state(), io_dtype=torch.bfloat16)
    assert bsf.cfg.metadata.hook_name == "blocks.3.hook_resid_post" and bsf.use_error_term
    x = torch.randn(2, 4, _Tiny.d)
    out = bsf.decode(bsf.encode(x))
    assert out.shape == x.shape and out.dtype == torch.bfloat16


def _spliced_kept_error():
    pytest.importorskip("venngram.tuning")
    from torch import nn

    from interpretune.analysis.venngram import LatentCatalog
    from interpretune.tuning.latents import LatentSite, attach_latent_model

    tiny = _Tiny()
    root = nn.Module()
    root.model = nn.ModuleList([nn.Linear(tiny.d, tiny.d) for _ in range(4)])
    kept = KeptError(BlockSparseFeaturizer(tiny.state()))
    attach_latent_model(root, "bsf", kept, LatentSite("model.3", op="replace"), frozen=True)
    return root, kept, LatentCatalog([kept.binding("bsf")])


def test_kept_error_passes_the_activation_through_until_its_adapter_trains():
    from interpretune.tuning.weights import attach_feature_adapter

    root, kept, catalog = _spliced_kept_error()
    f, x = kept.featurizer, torch.randn(3, _Tiny.d)
    h = root.model[3].forward(x)  # the module's own output, without the splice's hooks
    torch.testing.assert_close(root.model[3](x), h)  # spliced, zero adapters: the activation is unchanged
    block = int((f.block_codes(h).norm(dim=-1) > 0).sum(0).argmax())  # a block open on these inputs
    target = f"features:bsf & layers:3 & units:[{block}]"
    handles = attach_feature_adapter(root, catalog, target, name="blk", n_layers=4)
    assert {h.target for h in handles} == {f"latent_models.bsf.featurizer.{t}" for t in ("W_enc", "b_enc", "W_dec")}
    torch.testing.assert_close(root.model[3](x), h)  # the adapters start at zero
    with torch.no_grad():
        for handle in handles:
            handle.edit.delta.add_(0.5)
    # the activation moves by the change in the reconstruction: the adapted featurizer's less the loaded one's
    loaded = (KeptError(BlockSparseFeaturizer(_Tiny().state())).featurizer.encode(h) @ kept.W_dec_0) * f.scale
    expected = h + (f.encode(h) @ f.W_dec) * f.scale - loaded
    moved = root.model[3](x)
    assert (moved - h).abs().max() > 0
    torch.testing.assert_close(moved, expected)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(root.model[3](x), h)
