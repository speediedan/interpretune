"""CPU tests for the attention-aware attribution notebook's helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
    neumann = aah.plot_neumann_convergence({"frozen": trace}).to_dict()["data"]
    assert len(neumann) == 1
    # the terminating all-zero term is not plotted: on a log axis it would drag the line off the plot
    fig_trace: Any = aah.plot_neumann_convergence({"frozen": trace}).data[0]
    assert list(fig_trace.x) == [1, 2] and list(fig_trace.y) == [2.0, 0.5]
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
    assert fig.layout.title.text.startswith("Edges into 'Austin'")


def test_stacked_label_offsets_never_share_a_row():
    # a tight cluster in the upper right plus one point on the left, as the regime scatter produces
    xs, ys = [7.08, -5.39, 6.08, 5.11, 7.16, 5.02], [6.27, 5.4, 6.73, 6.29, 6.83, 6.06]
    span, plot_px = 8.8, 340.0
    offsets = aah._stacked_label_offsets(xs, ys, span, plot_px, gap_px=20.0)
    rows = sorted((span - y) * plot_px / (2 * span) + ay for y, (_, ay) in zip(ys, offsets))
    assert all(b - a >= 20.0 - 1e-9 for a, b in zip(rows, rows[1:]))
    assert all(r <= plot_px + 1e-9 for r in rows)
    # labels point toward the emptier half: left of a right-half point, right of a left-half one
    assert [ax < 0 for ax, _ in offsets] == [x >= 0 for x in xs]


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


def test_notebook_table_colours_signed_cells_aligns_numbers_and_shows_a_legend():
    from it_examples.utils.nb_ui_utils import (
        NEGATIVE_COLOUR,
        POSITIVE_COLOUR,
        SIGN_LEGEND,
        Signed,
        build_notebook_table_html,
    )

    html_out = build_notebook_table_html(
        [["up", Signed(1.5), 3], ["down", Signed(-0.25, "+.3f"), 2.5], ["flat", Signed(0.0), "n/a"]],
        ["row", "shift", "count"],
        title="t",
        legend=SIGN_LEGEND,
        emphasize=[1],
    )
    assert f'style="color:{POSITIVE_COLOUR};font-weight:600">+1.50</td>' in html_out
    assert f'style="color:{NEGATIVE_COLOUR};font-weight:600">-0.250</td>' in html_out
    assert 'style="color:inherit;font-weight:600">+0.00</td>' in html_out
    assert '<td class="num">3</td>' in html_out and "<td>n/a</td>" in html_out
    assert html_out.count('<tr class="em">') == 1 and '<tr class="em"><td>down</td>' in html_out
    assert 'class="legend"' in html_out and "positive" in html_out and "negative" in html_out
    assert 'class="legend"' not in build_notebook_table_html([["a"]], ["c"], title="t")
    with pytest.raises(ValueError, match="outside the table"):
        build_notebook_table_html([["a"]], ["c"], title="t", emphasize=[1])


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


def test_node_positions_and_position_change_mass_separate_within_and_cross_position_edges():
    features = torch.tensor([[0, 0, 5], [1, 1, 6]])  # two features, at positions 0 and 1
    n_layers, n_pos, n_logits = 1, 2, 1
    positions = aah.node_positions(features, n_layers, n_pos, n_logits)
    assert positions.tolist() == [0, 1, 0, 1, 0, 1, -1]
    kinds = aah.node_kinds(2, n_layers, n_pos, n_logits)
    delta = torch.zeros(7, 7)
    delta[1, 0] = -3.0  # feature at pos 1 <- feature at pos 0: crosses positions
    delta[0, 0] = 2.0  # within position 0
    delta[6, 1] = 100.0  # a logit row is not a feature-to-feature edge
    mass = aah.position_change_mass(delta, positions, kinds, n_pos)
    assert mass.tolist() == [[2.0, 0.0], [3.0, 0.0]]


def test_influence_by_kind_reports_a_share_that_does_not_scale_with_the_logit_weights():
    kinds = aah.node_kinds(2, 1, 2, 1)
    influence = torch.tensor([0.4, 0.2, 0.1, 0.1, 0.2, 0.0, 0.0])
    one = aah.influence_by_kind(influence, kinds)
    scaled = aah.influence_by_kind(influence * 0.01, kinds)
    assert one["error"] == pytest.approx(0.2) and one["total"] == pytest.approx(1.0)
    assert scaled["error_share"] == pytest.approx(one["error_share"]) == pytest.approx(0.2)


def test_residual_partition_reports_what_its_parts_leave_unexplained():
    part = aah.ResidualPartition(
        t_from=8.0, t_to=-6.0, parts={"embed": 0.0, "attn3": -10.0, "mlp3": -3.5}, recon_error=0.0
    )
    assert part.change == -14.0
    assert part.total("attn") == -10.0 and part.total("mlp") == -3.5
    assert part.unexplained == pytest.approx(-0.5)


class _TinyLM(torch.nn.Module):
    """Just enough of a Hugging Face causal LM for the logit lens: a final norm, an unembedding and a soft cap."""

    def __init__(self, cap: float | None = None):
        super().__init__()
        self.norm = torch.nn.RMSNorm(2, elementwise_affine=False)
        self.head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        self.config = type("Cfg", (), {"final_logit_softcapping": cap})()

    def get_decoder(self):
        return self

    def get_output_embeddings(self):
        return self.head


def test_logit_lens_contrast_reads_through_norm_and_cap():
    module = type("M", (), {"model": _TinyLM()})()
    residuals = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    # RMSNorm maps (3, 0) to (sqrt 2, 0) and (0, 2) to (0, sqrt 2); token 0 minus token 1 reads +sqrt 2 and -sqrt 2
    read = aah.logit_lens_contrast(module, residuals, 0, 1)
    torch.testing.assert_close(read, torch.tensor([2.0**0.5, -(2.0**0.5)]))
    read, direction = aah.logit_lens_contrast(module, residuals, 0, 1, with_direction=True)
    assert direction.shape == residuals.shape
    # a step along the returned direction raises the read; a norm is scale-invariant, so a pure rescale does not
    step = aah.logit_lens_contrast(module, residuals + 1e-3 * direction, 0, 1)
    assert bool((step > read).all())
    torch.testing.assert_close(aah.logit_lens_contrast(module, 5 * residuals, 0, 1), read)
    capped = aah.logit_lens_contrast(type("M", (), {"model": _TinyLM(cap=1.0)})(), residuals, 0, 1)
    assert bool((capped.abs() < 1.0).all())  # tanh(sqrt 2) - tanh(0)


def test_plot_state_read_one_panel_per_position():
    reads = {"Dallas": torch.tensor([[0.0, 1.0], [0.5, 2.0]]), "Detroit": torch.tensor([[0.0, -1.0], [0.0, -2.0]])}
    traces: list[Any] = list(aah.plot_state_read(reads, {0: "first", 1: "second"}, "t", "y").data)
    assert len(traces) == 4 and [t.name for t in traces if t.showlegend] == ["Dallas", "Detroit"]
    assert list(traces[2].y) == [1.0, 2.0]  # panels in position order, prompts within: Dallas at position 1


def test_token_positions_mark_each_labelled_position_once_in_the_legend():
    from it_examples.utils.nb_ui_utils import build_token_positions_html

    tokens = ["<bos>", "\u2581apple", "\u2581:", "\n", "Fruit"]
    marks = {1: ("probe word", "#111111"), 3: ("context state", "#222222"), 4: ("answer state", "#333333")}
    html_out = build_token_positions_html(tokens, marks, title="Where <states> are read")
    assert "Where &lt;states&gt; are read" in html_out and "&lt;bos&gt;" in html_out
    assert " apple" in html_out and "\u23ce" in html_out  # the space marker and the newline as a reader sees them
    assert html_out.count('class="lab"') == 3 and 'title="position 4: answer state"' in html_out
    assert html_out.count('class="sw"') == 3
    with pytest.raises(ValueError, match=r"marked positions \[5\] are outside the prompt's 5 tokens"):
        build_token_positions_html(tokens, {5: ("x", "#000")}, title="t")


@pytest.mark.parametrize("n_layers", [26, 16, 12])
def test_intervention_arms_resolve_on_any_depth(n_layers):
    half = n_layers // 2
    sites = aah.resolve_intervention_arms(n_layers)
    assert set(sites) == set(aah.INTERVENTION_ARMS)
    assert sites["attention, all layers"] == tuple(f"blocks.{layer}.hook_attn_out" for layer in range(n_layers))
    assert sites["MLP, late half"] == tuple(f"blocks.{layer}.hook_mlp_out" for layer in range(half, n_layers))
    early, late = set(sites["attention, early half"]), set(sites["attention, late half"])
    assert not early & late and early | late == set(sites["attention, all layers"])
    both = set(sites["attention, all layers"]) | set(sites["MLP, all layers"])
    assert set(sites["both, all layers"]) == both and len(sites["both, all layers"]) == 2 * n_layers


def test_intervention_arms_take_another_book():
    sites = aah.resolve_intervention_arms(
        4, {"attention, last layer": "@attn & layers:-1"}, aah.intervention_arm_sets(4)
    )
    assert sites == {"attention, last layer": ("blocks.3.hook_attn_out",)}


def test_module_tokenizer_prefers_the_replacement_models_then_the_configs():
    replacement, configured = object(), object()
    with_both = SimpleNamespace(
        replacement_model=SimpleNamespace(tokenizer=replacement), it_cfg=SimpleNamespace(tokenizer=configured)
    )
    assert aah.module_tokenizer(with_both) is replacement
    assert aah.module_tokenizer(SimpleNamespace(it_cfg=SimpleNamespace(tokenizer=configured))) is configured
    with pytest.raises(ValueError, match="no tokenizer"):
        aah.module_tokenizer(SimpleNamespace(it_cfg=SimpleNamespace(tokenizer=None)))
