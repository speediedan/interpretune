"""CPU tests for the attention-aware attribution notebook's helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from it_examples.utils import attention_attribution_helpers as aah


def _graph(selected, n_other=5, logit_ids=(7, 9)):
    """A stand-in attribution graph: 6 active features, ``selected`` of them as nodes, then ``n_other`` others."""
    active = torch.tensor(
        [[layer, pos, 100 + i] for i, (layer, pos) in enumerate([(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)])]
    )
    n_nodes = len(selected) + n_other
    adjacency = torch.arange(n_nodes * n_nodes, dtype=torch.float32).reshape(n_nodes, n_nodes)
    return SimpleNamespace(
        active_features=active,
        selected_features=torch.tensor(selected),
        adjacency_matrix=adjacency,
        logit_target_ids=torch.tensor(logit_ids),
    )


def test_align_regime_nodes_matches_features_by_identity():
    first, second = _graph([0, 2, 3, 5]), _graph([3, 0, 4, 2])
    aligned = aah.align_regime_nodes(first, second)
    assert aligned.n_common == 3 and aligned.n_first_only == 1 and aligned.n_second_only == 1
    # every aligned feature position names the same active feature in both graphs
    for j in range(aligned.n_common):
        a = int(first.selected_features[aligned.first[j]])
        b = int(second.selected_features[aligned.second[j]])
        assert a == b and torch.equal(aligned.common_features[j], first.active_features[a])
    # the non-feature nodes follow in order in both
    assert aligned.first[aligned.n_common :].tolist() == list(range(4, 9))
    assert aligned.second[aligned.n_common :].tolist() == list(range(4, 9))


def test_index_alignment_would_compare_different_features():
    # the positive control for the helper's reason to exist: same shape, same index, different features
    first, second = _graph([0, 2, 3, 5]), _graph([3, 0, 4, 2])
    assert first.adjacency_matrix.shape == second.adjacency_matrix.shape
    assert not torch.equal(first.selected_features, second.selected_features)


@pytest.mark.parametrize(
    "second, match",
    [
        (SimpleNamespace(**{**vars(_graph([0, 1])), "logit_target_ids": torch.tensor([7, 8])}), "logit targets"),
        (_graph([0, 1], n_other=6), "error, token and logit"),
    ],
)
def test_align_regime_nodes_refuses_incomparable_graphs(second, match):
    with pytest.raises(ValueError, match=match):
        aah.align_regime_nodes(_graph([0, 1]), second)


def test_align_regime_nodes_refuses_other_prompts():
    other = _graph([0, 1])
    other.active_features = other.active_features + 1
    with pytest.raises(ValueError, match="active features"):
        aah.align_regime_nodes(_graph([0, 1]), other)


def test_top_k_overlap_uses_identities():
    a, b = torch.tensor([3.0, 2.0, 1.0]), torch.tensor([1.0, 2.0, 3.0])
    # by index the top-1 sets differ; by identity they are the same node
    assert aah.top_k_overlap(a, b, ks=(1,), ids_a=["x", "y", "z"], ids_b=["z", "y", "x"]) == {1: 1.0}
    assert aah.top_k_overlap(a, b, ks=(1,), ids_a=["x", "y", "z"], ids_b=["x", "y", "z"]) == {1: 0.0}
    with pytest.raises(ValueError, match="identity"):
        aah.top_k_overlap(a, b, ks=(1,), ids_a=["x"], ids_b=["x", "y", "z"])


class _Tokenizer:
    vocab = {"▁Austin": 2, "▁Dallas": 3, "▁the": 1}

    def encode(self, word, add_special_tokens=False):
        return [self.vocab[word]] if word in self.vocab else [5, 6]

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return inverse.get(ids[0], "?").replace("▁", " ")


class _Model:
    def __init__(self, logits):
        self.logits = logits

    def get_activations(self, prompt):
        return self.logits.view(1, 1, -1), None


def test_check_next_token_reads_rank_margin_and_copy():
    logits = torch.tensor([0.0, 1.0, 4.0, 2.5, 0.0, 0.0, 0.0])
    check = aah.check_next_token(_Model(logits), _Tokenizer(), "Dallas", "p", "▁Austin", copy_token="▁Dallas")
    assert check.rank == 1 and check.margin == pytest.approx(1.5)
    assert check.prob == pytest.approx(float(logits.softmax(-1)[2]))
    assert check.copy_prob == pytest.approx(float(logits.softmax(-1)[3]))
    assert check.top[0][0] == " Austin"


def test_check_next_token_refuses_multi_token_answers():
    with pytest.raises(ValueError, match="single-token"):
        aah.check_next_token(_Model(torch.zeros(7)), _Tokenizer(), "x", "p", "▁Lansing")


def test_neumann_term_mass_and_figures():
    trace = [torch.tensor([1.0, -1.0]), torch.tensor([0.5, 0.0]), torch.tensor([0.0, 0.0])]
    np.testing.assert_allclose(aah.neumann_term_mass(trace), [2.0, 0.5, 0.0])
    assert len(aah.plot_neumann_convergence({"frozen": trace}).to_dict()["data"]) == 1
    ridge = aah.plot_ridgeline([("a", np.array([1e-3, 1e-2, 0.0])), ("b", np.array([0.0]))], "t", "x")
    assert len(ridge.to_dict()["data"]) == 1  # a series with fewer than two values above the floor draws no curve
    kinds = ["feature", "feature", "error", "token"]
    fig = aah.plot_regime_edges(
        torch.tensor([1.0, 0.0, 0.5, -0.2]),
        torch.tensor([0.5, 0.0, 0.5, 0.3]),
        kinds,
        ["f0", "f1", "e0", "t0"],
        "Austin",
        annotate=1,
    )
    assert {trace["name"].split(" ")[0] for trace in fig.to_dict()["data"]} == {"feature", "error", "token", "no"}


def test_head_split_parts_add_to_the_bypass_plus_heads():
    rows = torch.tensor([[1.5], [2.0], [1.0], [2.5]])  # head 0 copy, head 1 copy, bypass, every head live
    fig = aah.plot_head_split({"rows": rows, "n_heads": 2, "layer": 3}, 0, "src", "Austin")
    assert list(fig.to_dict()["data"][0]["y"]) == [1.0, 0.5, 1.0, 0.0]


def test_running_credit_and_conductance_path():
    running = {
        0: [(torch.tensor([0.1, 0.0]), torch.tensor([0.2, 0.0]), 0.0)] * 2,
        1: [(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), 0.0)] * 2,
    }
    assert aah.running_head_credit(running).shape == (2, 2, 2)
    assert len(aah.plot_running_credit(running, n_steps=2, top=2).to_dict()["data"]) == 2
    points = [
        {"alphas": torch.tensor([0.0, 0.5]), "t": torch.tensor([1.0, 0.5]), "dt_dalpha": torch.tensor([-1.0, -1.0])},
        {"alphas": torch.tensor([1.0]), "t": torch.tensor([0.0]), "dt_dalpha": torch.tensor([-1.0])},
    ]
    alphas, t, _ = aah.conductance_path(points)
    assert alphas.tolist() == [0.0, 0.5, 1.0] and t.tolist() == [1.0, 0.5, 0.0]
    assert len(aah.plot_conductance_path(points, "T").to_dict()["data"]) == 3


def test_display_plotly_figure_embeds_html(monkeypatch):
    shown = []
    monkeypatch.setattr("it_examples.utils.nb_ui_utils.display", shown.append)
    aah.display_plotly_figure(aah.plot_residual_convergence([8, 16], [0.02, 0.01]), "it-test")
    assert len(shown) == 1 and "cdn.plot.ly" in shown[0].data and 'id="it-test-' in shown[0].data


def test_notebook_table_escapes_cells_and_renders_only_http_links():
    from it_examples.utils.nb_ui_utils import NotebookLink, build_notebook_table_html

    html_out = build_notebook_table_html(
        [["<script>x</script>", NotebookLink("L20 #15589", "https://example.org/a?b=1&c=2")]],
        ["text", "link"],
        title="T <b>",
        caption="denominator & units",
    )
    assert "<script>" not in html_out and "&lt;script&gt;x&lt;/script&gt;" in html_out
    assert '<a href="https://example.org/a?b=1&amp;c=2">L20 #15589</a>' in html_out
    assert "T &lt;b&gt;" in html_out and "denominator &amp; units" in html_out
    with pytest.raises(ValueError, match="HTTP"):
        build_notebook_table_html([[NotebookLink("x", "javascript:alert(1)")]], ["link"], title="t")
    with pytest.raises(ValueError, match="column count"):
        build_notebook_table_html([[1, 2]], ["only one"], title="t")


def test_feature_context_links_quotes_tokens_and_marks_missing_explanations():
    ctx = aah.FeatureContext(
        tokens=["<bos>", " is"],
        explanations={(20, 15589): "Texas"},
        model_id="gemma-2-2b",
        source_set="gemmascope-transcoder-16k",
    )
    link = ctx.link(20, 15589)
    assert link.url == "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-transcoder-16k/15589"
    assert ctx.describe(20, 1, 15589) == "L20 #15589 on ' is' (pos 1), \"Texas\""
    assert ctx.explanation(3, 7) == "-" and ctx.describe(3, 1, 7) == "L3 #7 on ' is' (pos 1)"


def test_resolve_feature_context_requests_each_feature_once(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {(20, 15589): "Texas"}

    monkeypatch.setattr(aah, "resolve_feature_explanations", fake)
    ctx = aah.resolve_feature_context(
        [(20, 10, 15589), (20, 9, 15589), (21, 10, 5943)], ["a"], model_id="m", source_set="s"
    )
    assert seen["feature_tuples"] == [(20, 15589), (21, 5943)]
    assert seen["preferred_type_name"] == "oai_token-act-pair"
    assert ctx.explanation(20, 15589) == "Texas"
