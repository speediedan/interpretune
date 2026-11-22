"""Neuron-finder settings as a checkpoint book, and the statistic their doses are in (#869)."""

import pytest
import torch

from it_examples.utils.neuron_finder import (
    Setting,
    activation_std,
    answer_margin,
    book_payload,
    book_settings,
    cached_revision,
    candidate_settings,
    dose_unit,
    edit_sign,
    found,
    parse_settings,
    residual_medians,
    select_candidates,
    screen,
    settings_book,
    steered_margins,
)

SETTINGS = ["9:2684:1.5", "7:2147:0.8", "9:484:-2.5"]
REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def test_settings_parse_and_print_in_neuron_finders_form():
    parsed = parse_settings(SETTINGS)
    assert parsed[1] == Setting(7, 2147, 0.8)
    assert [str(s) for s in parsed] == SETTINGS


@pytest.mark.parametrize(
    "bad, match",
    [
        (["7:2147"], "layer:neuron:dose"),
        (["7:x:0.8"], "integer"),
        (["-1:3:0.8"], "nonnegative"),
        (["7:3:nan"], "finite"),
        (["7:3:0.8", "7:3:1.5"], "set twice"),
        ([], "No settings"),
    ],
)
def test_malformed_settings_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        parse_settings(bad)


def test_the_book_pins_the_checkpoint_and_round_trips_the_settings():
    book = settings_book(SETTINGS, model_id="gpt2", revision=REV, architecture="GPT2LMHeadModel")
    assert book["portability"] == "checkpoint"
    assert book["applies_to"]["models"] == [{"id": "gpt2", "revision": REV, "basis": "unprocessed"}]
    assert book["sets"]["found"] == {"point": "mlp.out.hook_in", "units": {7: [2147], 9: [484, 2684]}}
    assert sorted(book_settings(book)) == sorted(parse_settings(SETTINGS))


def test_the_book_loads_and_steers_by_dose_times_the_statistic():
    pytest.importorskip("venngram")
    from interpretune.analysis.venngram import load_recipe, recipe_interventions

    recipe = load_recipe(settings_book(SETTINGS, model_id="gpt2", revision=REV)).recipe
    stats = {(7, 2147): 10.0, (9, 484): 20.0, (9, 2684): 30.0}
    payload = recipe_interventions(recipe, site_widths={"mlp.out.hook_in": 3072}, statistics={"dose_unit": stats})
    assert set(payload) == {"blocks.7.mlp.out.hook_in", "blocks.9.mlp.out.hook_in"}
    (l9,) = payload["blocks.9.mlp.out.hook_in"]
    assert l9.op == "add"
    expected = torch.zeros(3072)
    expected[484], expected[2684] = -2.5 * 20.0, 1.5 * 30.0
    torch.testing.assert_close(l9.intervention_tensor.float(), expected)


def test_a_checkpoint_book_refuses_another_revision():
    pytest.importorskip("venngram")
    from venngram.errors import AddressError
    from venngram.book import AddressBook

    book = AddressBook.from_mapping(settings_book(SETTINGS, model_id="gpt2", revision=REV))
    book.check_applies(model_id="gpt2", revision=REV)
    with pytest.raises(AddressError, match="applies only to"):
        book.check_applies(model_id="gpt2", revision="main")


@pytest.mark.parametrize("unit_dim", [0, 1])
def test_the_dose_unit_is_the_median_over_the_column_norm(unit_dim):
    weight = torch.randn(6, 4) if unit_dim == 0 else torch.randn(4, 6)
    got = dose_unit({3: (weight, unit_dim)}, {3: 12.0}, [(3, 1), (3, 5)])
    for unit in (1, 5):
        column = weight[unit] if unit_dim == 0 else weight[:, unit]
        assert got[(3, unit)] == pytest.approx(12.0 / float(column.norm()), rel=1e-5)


def test_residual_medians_skip_the_first_token_and_padding():
    # one layer after the embeddings; two texts, the second left-padded by one token
    norms = torch.tensor([[100.0, 1.0, 2.0, 3.0], [9.0, 100.0, 4.0, 6.0]])
    states = (torch.zeros(2, 4, 1), norms.unsqueeze(-1))
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    # text 1: tokens 1..3 -> median 2; text 2: real tokens [100, 4, 6], from the second on -> [4, 6], median 4
    assert residual_medians([states], [mask]) == [pytest.approx(3.0)]


def test_select_candidates_takes_each_target_strongest_first():
    score = torch.tensor([[0.5, -2.0, 0.1], [3.0, -0.2, -1.0]])
    picked = select_candidates(score, 2)
    assert picked[1] == [(1, 0), (0, 0)]
    assert picked[-1] == [(0, 1), (1, 2)]


def test_edit_sign_follows_the_target_times_the_gradient():
    write = torch.tensor([[0.5, -2.0]])
    assert edit_sign(write, 0, 0, 1) == 1
    assert edit_sign(write, 0, 1, 1) == -1
    # a neuron whose gradient points down is steered down by raising its activation
    assert edit_sign(write, 0, 1, -1) == 1
    with pytest.raises(ValueError, match="target is"):
        edit_sign(write, 0, 0, 0)


def test_candidate_settings_sign_each_dose_by_its_edit():
    write = torch.tensor([[0.5, -2.0, 0.1]])
    settings = candidate_settings(write, {1: [(0, 0), (0, 1)], -1: [(0, 2)]}, 4.0)
    assert settings == [Setting(0, 0, 4.0), Setting(0, 1, -4.0), Setting(0, 2, -4.0)]


def test_activation_std_is_per_neuron_and_floored():
    acts = torch.zeros(4, 1, 2)
    acts[:, 0, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    std = activation_std(acts)
    assert std.shape == (1, 2)
    assert std[0, 0] > 1.0 and std[0, 1] == pytest.approx(1e-6)


def test_found_compares_with_the_controls_quantile():
    controls = [0.0] * 19 + [1.0]
    verdict = found({"a": 2.0, "b": 0.01, "c": -3.0}, controls)
    assert verdict == {"a": True, "b": False, "c": False}
    with pytest.raises(ValueError, match="No control"):
        found({"a": 1.0}, [])


def test_answer_margin_reads_the_last_position():
    logits = torch.zeros(2, 3, 5)
    logits[:, -1, 1], logits[:, -1, 4] = torch.tensor([3.0, -1.0]), torch.tensor([1.0, 2.0])
    logits[:, 0, 1] = 100.0  # an earlier position is not read
    torch.testing.assert_close(answer_margin(logits, (1, 4)), torch.tensor([2.0, -3.0]))


def _tiny_qwen3():
    tr = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = tr.Qwen3Config(
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
    )
    return tr.Qwen3ForCausalLM(cfg).eval()


def test_the_write_screen_is_the_margins_gradient_at_the_answer_position():
    model = _tiny_qwen3()
    projections = {i: layer.mlp.down_proj for i, layer in enumerate(model.model.layers)}
    batch = {
        "input_ids": torch.randint(0, 64, (3, 5)),
        "attention_mask": torch.ones(3, 5, dtype=torch.long),
        "labels": torch.tensor([0, 1, 0]),
    }
    result = screen(model, projections, [batch], (7, 9))
    assert result.activations.shape == (3, 2, 32) and result.write.shape == (2, 32)
    assert all(p.requires_grad for p in model.parameters()), "the parameters' flags are restored"
    assert result.read().shape == (2, 32)

    # the oracle shares no code with the screen: a finite difference of the mean margin in one neuron
    layer, unit, eps = 1, 5, 1e-3

    def margin_with(delta):
        def pre(_module, args):
            x = args[0].clone()
            x[:, -1, unit] += delta
            return (x,)

        handle = projections[layer].register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        finally:
            handle.remove()
        return answer_margin(logits, (7, 9)).mean()

    numeric = (margin_with(eps) - margin_with(-eps)) / (2 * eps)
    torch.testing.assert_close(result.write[layer, unit], numeric, atol=1e-3, rtol=1e-2)


def test_book_payload_scales_the_doses_by_the_books_statistic_at_the_last_token():
    pytest.importorskip("venngram")
    from interpretune.analysis.backends.capabilities import PositionScope

    book = settings_book(["3:2:2.0"], model_id="gpt2", revision=REV, statistic="activation_std", tokens="last")
    payload = book_payload(book, 8, {(3, 2): 0.5})
    ((spec,),) = payload.values()
    assert list(payload) == ["blocks.3.mlp.out.hook_in"]
    assert spec.position_scope == PositionScope.LAST_TOKEN
    expected = torch.zeros(8)
    expected[2] = 1.0
    torch.testing.assert_close(spec.intervention_tensor.float(), expected)


def test_steered_margins_read_both_passes_of_the_backend():
    class Backend:
        def fwd_w_intervention(self, model, inputs, payload):
            assert set(inputs) == {"input", "attention_mask"} and payload == {"p": 1}
            clean = torch.zeros(2, 4, 3)
            edited = clean.clone()
            edited[:, -1, 0] = 2.0
            return clean, edited

    batch = {"input_ids": torch.ones(2, 4), "attention_mask": torch.ones(2, 4), "labels": torch.zeros(2)}
    clean, edited = steered_margins(Backend(), None, [batch, batch], {"p": 1}, (0, 1), input_key="input")
    torch.testing.assert_close(clean, torch.zeros(4))
    torch.testing.assert_close(edited, torch.full((4,), 2.0))


def test_cached_revision_reads_the_snapshot_commit(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = tmp_path / "snapshots" / "c1899de2" / "config.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: str(snapshot))
    assert cached_revision("org/model") == "c1899de2"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: None)
    with pytest.raises(ValueError, match="not in the local Hugging Face cache"):
        cached_revision("org/model")
