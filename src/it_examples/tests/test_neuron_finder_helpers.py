"""Neuron-finder settings as a checkpoint book, and the statistic their doses are in (#869)."""

import pytest
import torch

from it_examples.utils.neuron_finder import (
    Setting,
    book_settings,
    dose_unit,
    parse_settings,
    residual_medians,
    settings_book,
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
