"""The shared notebook display helpers in :mod:`interpretune.harness.notebook_ui`.

The J-space readout comparison is a pure markup builder with a summary a notebook can assert on. The other helpers
hand an HTML string to ``display``; the tests capture that string by replacing the module's ``display`` and read back
which values land in which cells, which rows are highlighted or marked, and what the helper returns. Everything is
synthetic (tiny tokenizers, small tensors, a stubbed feature API), so the tests check the rendering contract and
nothing about a model or the network.
"""

from __future__ import annotations

import html

import re
import sys
from types import SimpleNamespace

import pytest
import torch

from interpretune.harness import notebook_ui
from interpretune.harness.display import format_score
from interpretune.harness.notebook_inputs import FeatureIOProfile
from interpretune.harness.notebook_ui import (
    best_variant_token_ids,
    build_jlens_readout_comparison_html,
    display_concept_positions,
    display_feature_decoupling_table,
    display_steering_results,
    display_target_gap,
    display_top_features_comparison,
    format_prob,
    plot_decoder_projection_map,
    resolve_feature_explanations,
)

_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_ROW = re.compile(r'<tr class="(?:even|odd)-row">(.*?)</tr>', re.S)


def test_pre_post_scores_land_in_their_columns_with_rank_moves():
    pre = [("France", 12.5), ("Paris", 11.0), ("Rome", 9.25)]
    post = [("Paris", 13.75), ("France", 12.0), ("Madrid", 8.5)]
    markup, summary = build_jlens_readout_comparison_html(pre, post, layer=25, basis="jlens_folded")
    assert summary.pre_labels == ("France", "Paris", "Rome")
    assert summary.post_labels == ("Paris", "France", "Madrid")
    assert summary.pre_scores == (12.5, 11.0, 9.25)
    assert summary.post_scores == (13.75, 12.0, 8.5)
    # Paris 2 -> 1 is +1, France 1 -> 2 is -1, Madrid was outside pre top-k.
    assert summary.rank_moves == (1, -1, None)
    assert summary.layer == 25 and summary.basis == "jlens_folded"
    for text in ("+12.50", "+11.00", "+13.75", "+12.00", "+1", "-1", "new", "layer 25"):
        assert text in markup, text


def test_tokens_are_escaped_and_uneven_sides_render():
    pre = [("<b>FR</b>", 1.0)]
    post = [("<b>FR</b>", 2.0), ("&amp;", -0.5)]
    markup, summary = build_jlens_readout_comparison_html(pre, post, layer=3, basis="embed")
    assert "<b>FR</b>" not in markup and "&lt;b&gt;FR&lt;/b&gt;" in markup
    assert summary.rank_moves == (0, None)


@pytest.fixture
def shown(monkeypatch) -> list[str]:
    """Collect the HTML string of every ``display(HTML(...))`` call the helpers make."""
    captured: list[str] = []
    monkeypatch.setattr(notebook_ui, "display", lambda obj: captured.append(obj.data))
    return captured


class _VocabTokenizer:
    """Tiny two-way vocabulary.

    ``encode`` prepends a BOS id unless special tokens are disabled, so a caller that forgets
    ``add_special_tokens=False`` resolves every token to the BOS id.
    """

    BOS = 0

    def __init__(self) -> None:
        self.vocab = {"<bos>": 0, "the": 1, "<b>": 2, "Fruit": 3, "Color": 4, " Fruit": 7, " Color": 9}
        self.inverse = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.vocab[text]]
        return [self.BOS, *ids] if add_special_tokens else ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.inverse[int(t)] for t in token_ids)


class TestFormatProb:
    def test_percent_at_and_above_threshold(self) -> None:
        assert format_prob(0.5) == "50.0000%"
        assert format_prob(1e-3) == "0.1000%"
        assert format_prob(0.25, precision=1) == "25.0%"

    def test_scientific_below_threshold_with_capped_precision(self) -> None:
        assert format_prob(9.5e-4) == "9.50e-04"
        assert format_prob(1e-5, precision=1) == "1.0e-05"
        # precision above 2 is capped in scientific notation
        assert format_prob(1e-5, precision=6) == "1.00e-05"


class TestAnswerLogits:
    """The displays read one prompt's answer row: every accepted shape gives the last position, others are
    refused."""

    def test_each_accepted_shape_reads_the_last_position(self) -> None:
        from interpretune.harness.notebook_ui import _ensure_1d_logits

        seq = torch.arange(12.0).reshape(3, 4)
        for logits in (seq[-1], seq, seq.unsqueeze(0), seq[-1:]):
            assert torch.equal(_ensure_1d_logits(logits), seq[-1])

    def test_a_batch_of_prompts_is_refused_by_shape(self) -> None:
        from interpretune.harness.notebook_ui import _ensure_1d_logits

        with pytest.raises(ValueError, match=r"got \(2, 3, 4\)"):
            _ensure_1d_logits(torch.zeros(2, 3, 4))


class TestBestVariantTokenIds:
    def test_picks_the_variant_with_the_higher_reference_logit(self) -> None:
        logits = torch.zeros(10)
        logits[7], logits[3] = 5.0, 1.0  # " Fruit" beats "Fruit"
        logits[4], logits[9] = 2.0, 0.0  # "Color" beats " Color"
        assert best_variant_token_ids(_VocabTokenizer(), ["Fruit", "Color"], logits) == [7, 4]

    def test_bare_spelling_when_best_variant_disabled(self) -> None:
        logits = torch.zeros(10)
        logits[7] = logits[9] = 5.0
        assert best_variant_token_ids(_VocabTokenizer(), ["Fruit", "Color"], logits, use_best_variant=False) == [3, 4]

    def test_reads_the_last_position_of_batched_sequence_logits(self) -> None:
        logits = torch.zeros(1, 2, 10)
        logits[0, 0, 7] = 9.0  # an earlier position prefers " Fruit"
        logits[0, -1, 3] = 9.0  # the last position prefers "Fruit"
        assert best_variant_token_ids(_VocabTokenizer(), ["Fruit"], logits) == [3]


class TestDisplayTargetGap:
    @staticmethod
    def _logits() -> tuple[torch.Tensor, torch.Tensor]:
        pre = torch.zeros(1, 2, 6)
        post = torch.zeros(1, 2, 6)
        # A first position that would give a very different gap if it were read instead of the last.
        pre[0, 0, 2], pre[0, 0, 4] = -5.0, 5.0
        pre[0, 1, 2], pre[0, 1, 4] = 2.0, 1.0
        post[0, 1, 2], post[0, 1, 4] = 4.0, 0.5
        return pre, post

    def test_returns_last_position_gaps_and_renders_them(self, shown) -> None:
        pre, post = self._logits()
        pre_gap, post_gap = display_target_gap(pre, post, ("Fruit", 2), ("<Veg>", 4), title="Gap & check")
        assert pre_gap == pytest.approx(1.0)
        assert post_gap == pytest.approx(3.5)
        (markup,) = shown
        rows = [_TD.findall(r) for r in _ROW.findall(markup)]
        pre_p = torch.softmax(pre[0, -1], -1)
        post_p = torch.softmax(post[0, -1], -1)
        assert rows[0] == [
            "Fruit",
            format_prob(float(pre_p[2]), precision=3),
            format_prob(float(post_p[2]), precision=3),
            "2.0000",
            "4.0000",
            "+2.0000",
        ]
        assert rows[1][0] == "&lt;Veg&gt;" and rows[1][3:] == ["1.0000", "0.5000", "-0.5000"]
        (gap_row,) = re.findall(r'<tr style="background:[^"]*">(.*?)</tr>', markup, re.S)
        assert _TD.findall(gap_row) == ["Gap (Fruit − &lt;Veg&gt;)", "", "", "+1.0000", "+3.5000", "+2.5000"]
        assert "Gap &amp; check" in markup

    def test_empty_title_omits_the_title_banner(self, shown) -> None:
        pre, post = self._logits()
        display_target_gap(pre, post, ("a", 2), ("b", 4), title="")
        assert "background:#555" not in shown[0]


def test_display_concept_positions_highlights_only_concept_positions(shown) -> None:
    display_concept_positions(_VocabTokenizer(), [1, 3, 2, 4], {1, 3})
    (markup,) = shown
    chips = re.findall(r'<span title="pos (\d+)" style="([^"]*)"[^>]*>(.*?)</span>', markup)
    assert [(int(p), text) for p, _, text in chips] == [(0, "the"), (1, "Fruit"), (2, "&lt;b&gt;"), (3, "Color")]
    highlighted = {int(p) for p, style, _ in chips if "#2471A3" in style}
    assert highlighted == {1, 3}
    assert "<b>[1, 3]</b>" in markup


class TestDisplayFeatureDecouplingTable:
    @staticmethod
    def _profiles(with_signature: bool) -> list[FeatureIOProfile]:
        sig = (("Fruit", 3.0), ("Color", 2.0), ("tok", 1.0), ("hidden", 0.5)) if with_signature else None
        return [
            FeatureIOProfile(10, 7, 0.0, 3.0, 0.05, jlens_signature=sig, jlens_concept_mass=0.75 if sig else None),
            FeatureIOProfile(12, 9, 0.5, 2.0, -0.20),
            FeatureIOProfile(14, 3, 0.1, 1.0, 0.01),
        ]

    def test_rows_sort_by_abs_projection_and_carry_markers(self, shown) -> None:
        long_text = "a<b " + "x" * 100
        display_feature_decoupling_table(self._profiles(False), {(12, 9): long_text}, target_label="P−Q")
        (markup,) = shown
        rows = [_TD.findall(r) for r in _ROW.findall(markup)]
        assert [r[0] for r in rows] == ["L12/9", "L10/7", "L14/3"]
        assert rows[0][:5] == ["L12/9", "0.500", "2.00", "-0.2000", "suppressor-motif"]
        # truncated before escaping, so an entity is never cut
        assert rows[0][5] == html.escape(long_text[:80])
        assert rows[1][:5] == ["L10/7", "0.000", "3.00", "+0.0500", "decoupled"]
        assert rows[2][4:] == ["", ""]
        assert "Output proj (P−Q)" in markup

    def test_signature_cell_shows_top_three_tokens_and_mass(self, shown) -> None:
        display_feature_decoupling_table(self._profiles(True))
        rows = {r[0]: r for r in (_TD.findall(r) for r in _ROW.findall(shown[0]))}
        assert rows["L10/7"][4] == "Fruit, Color, tok (0.75)"
        assert rows["L12/9"][4] == " (—)"
        assert rows["L10/7"][5] == "decoupled"

    def test_header_matches_row_width_with_and_without_signatures(self, shown) -> None:
        for with_signature in (False, True):
            display_feature_decoupling_table(self._profiles(with_signature))
        for markup, width in zip(shown, (6, 7)):
            assert len(re.findall(r"<th ", markup)) == width
            assert {len(_TD.findall(r)) for r in _ROW.findall(markup)} == {width}

    def test_signature_tokens_are_escaped(self, shown) -> None:
        sig = (("<bos>", 1.0),)
        profile = FeatureIOProfile(10, 7, 0.0, 3.0, 0.05, jlens_signature=sig, jlens_concept_mass=0.5)
        display_feature_decoupling_table([profile])
        assert "&lt;bos&gt; (0.50)" in shown[0] and "<bos>" not in shown[0]


class TestDisplayTopFeaturesComparison:
    _FEATURES = {"cfg <A>": [(10, 2, 7), (12, 3, 9)], "cfg B": [(14, 0, 3)]}
    _SCORES = {"cfg <A>": [0.5, -1.234e-4], "cfg B": [0.0]}

    @staticmethod
    def _columns(markup: str) -> list[list[list[str]]]:
        return [
            [_TD.findall(r) for r in re.findall(r"<tr>(<td>.*?)</tr>", col, re.S)]
            for col in markup.split('<div class="col">')[1:]
        ]

    def test_columns_follow_labels_without_scores(self, shown) -> None:
        display_top_features_comparison(self._FEATURES)
        (markup,) = shown
        headers = re.findall(r'class="col-header" style="background-color: (#\w+);">(.*?)</div>', markup)
        assert headers == [("#2471A3", "cfg &lt;A&gt;"), ("#27AE60", "cfg B")]
        assert "Score</th>" not in markup and "Explanation</th>" not in markup
        cols = self._columns(markup)
        assert len(cols[0]) == 2 and len(cols[1]) == 1
        assert "(10,&#8239;2,&#8239;7)" in markup and "(14,&#8239;0,&#8239;3)" in markup

    def test_plain_scores_use_format_score(self, shown) -> None:
        display_top_features_comparison(self._FEATURES, self._SCORES)
        assert "<th>Score</th>" in shown[0]
        cols = self._columns(shown[0])
        assert [r[-1] for r in cols[0]] == [format_score(0.5), format_score(-1.234e-4)]

    def test_signed_scores_split_sign_and_magnitude(self, shown) -> None:
        display_top_features_comparison(self._FEATURES, self._SCORES, show_score_sign=True)
        assert "<th>Sign</th><th>|Score|</th>" in shown[0]
        cols = self._columns(shown[0])
        assert [r[2:4] for r in cols[0]] == [["+", format_score(0.5)], ["−", format_score(1.234e-4)]]
        assert cols[1][0][2:4] == ["0", format_score(0.0)]

    def test_links_and_explanations_key_on_layer_and_feature(self, shown) -> None:
        display_top_features_comparison(
            self._FEATURES,
            neuronpedia_model="gemma-2-2b",
            neuronpedia_base_url="https://np.example/",
            feature_explanations={(10, 7): "fruit & veg", (10, 2): "keyed by position, never shown"},
        )
        markup = shown[0]
        assert 'href="https://np.example/gemma-2-2b/10-gemmascope-transcoder-16k/7"' in markup
        assert 'href="https://np.example/gemma-2-2b/14-gemmascope-transcoder-16k/3"' in markup
        cols = self._columns(markup)
        assert [r[-1] for r in cols[0]] == ["fruit &amp; veg", ""]
        assert "never shown" not in markup


class TestDisplaySteeringResults:
    @staticmethod
    def _results() -> SimpleNamespace:
        pre = torch.zeros(1, 2, 10)
        post = torch.zeros(1, 2, 10)
        pre[0, -1, 7], pre[0, -1, 3] = 2.0, 1.0  # " Fruit" favored pre-intervention
        pre[0, -1, 4], pre[0, -1, 9] = 1.5, 0.0  # "Color" favored
        post[0, -1, 7], post[0, -1, 4] = 6.0, 0.5
        return SimpleNamespace(
            top_feature_ids=[torch.tensor([10, 2, 7]), torch.tensor([12, 3, 9])],
            top_feature_scores=torch.tensor([0.5, -0.25]),
            concept_direction=torch.tensor([[0.0, 1.0, 0.0]]),
            pre_intervention_logits=pre,
            post_intervention_logits=post,
        )

    def test_summary_and_both_tables(self, shown) -> None:
        results = self._results()
        summary = display_steering_results(
            results, _VocabTokenizer(), ["Fruit", "Color"], gap_title="Steer", min_layer=10
        )
        assert summary.steered_features == [(10, 2, 7), (12, 3, 9)]
        assert summary.target_ids == (7, 4)
        assert summary.pre_gap == pytest.approx(0.5)
        assert summary.post_gap == pytest.approx(5.5)
        assert summary.direction.shape == (3,) and summary.pre_logits.shape == (10,)
        assert torch.equal(summary.post_logits, results.post_intervention_logits[0, -1])
        features_html, gap_html = shown
        assert "Steered Features (signed influence)" in features_html and "<th>Sign</th>" in features_html
        assert "Steer" in gap_html and "+5.5000" in gap_html

    def test_bare_variant_when_best_variant_disabled(self, shown) -> None:
        summary = display_steering_results(
            self._results(), _VocabTokenizer(), ["Fruit", "Color"], use_best_variant=False
        )
        assert summary.target_ids == (3, 4)

    def test_min_layer_violation_is_refused(self, shown) -> None:
        with pytest.raises(ValueError, match="layer_slice constraint violated"):
            display_steering_results(self._results(), _VocabTokenizer(), ["Fruit", "Color"], min_layer=11)
        assert shown == []


class TestResolveFeatureExplanations:
    @pytest.fixture
    def fetched(self, monkeypatch) -> list:
        """Stub the feature API so no test reaches the network; payloads are keyed by the requested index."""
        from interpretune.utils import neuronpedia_explanations

        payloads = {
            "1": {"explanations": [{"description": "  fruit\n  words  ", "typeName": "auto"}]},
            "2": {
                "explanations": [
                    {"description": "generic", "typeName": "auto"},
                    {"description": "preferred one", "typeName": "np_max-act"},
                ]
            },
            "3": {"explanations": [{"description": "   "}, "not a mapping", {"description": "second row"}]},
            "4": {"explanations": None},
            "5": RuntimeError("feature API down"),
        }
        calls: list = []

        def fake_fetch(ref, timeout_seconds=30):
            calls.append((ref, timeout_seconds))
            payload = payloads[ref.index]
            if isinstance(payload, Exception):
                raise payload
            return payload

        monkeypatch.setattr(neuronpedia_explanations, "fetch_feature_payload", fake_fetch)
        return calls

    def test_best_effort_mapping(self, fetched) -> None:
        tuples = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (10, 1)]
        out = resolve_feature_explanations(
            model_id="gemma-2-2b", source_set="clt-hp", feature_tuples=tuples, timeout_seconds=3
        )
        # whitespace collapsed; blank and non-mapping rows skipped; no-list and failing fetches yield no entry
        assert out == {(10, 1): "fruit words", (11, 2): "generic", (12, 3): "second row"}
        # duplicates fetched once, in order, each ref built from its own tuple
        assert [(r.layer, r.index) for r, _ in fetched] == [
            ("10-clt-hp", "1"),
            ("11-clt-hp", "2"),
            ("12-clt-hp", "3"),
            ("13-clt-hp", "4"),
            ("14-clt-hp", "5"),
        ]
        assert {t for _, t in fetched} == {3}

    def test_preferred_type_wins_and_falls_back_when_absent(self, fetched) -> None:
        out = resolve_feature_explanations(
            model_id="m", source_set="s", feature_tuples=[(11, 2), (10, 1)], preferred_type_name="np_max-act"
        )
        assert out == {(11, 2): "preferred one", (10, 1): "fruit words"}

    def test_base_url_reaches_the_ref(self, fetched) -> None:
        resolve_feature_explanations(
            model_id="m", source_set="s", feature_tuples=[(10, 1)], base_url="http://localhost:3000"
        )
        assert fetched[0][0].api_url == "http://localhost:3000/api/feature/m/10-s/1"

    def test_no_tuples_makes_no_request(self, fetched) -> None:
        assert resolve_feature_explanations(model_id="m", source_set="s", feature_tuples=[]) == {}
        assert fetched == []


class TestPlotDecoderProjectionMap:
    """The UMAP step is pinned to PCA so the figure is deterministic and needs no ``umap-learn``."""

    @pytest.fixture(autouse=True)
    def _pca_projection(self, monkeypatch) -> None:
        from interpretune.utils import latent_state_projection

        real = latent_state_projection.project_embeddings
        monkeypatch.setattr(
            latent_state_projection,
            "project_embeddings",
            lambda values, **kwargs: real(values, method="pca", n_components=2),
        )

    @pytest.fixture
    def shown_figures(self, monkeypatch):
        import matplotlib.pyplot as plt

        figures: list = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: figures.append(plt.gcf()))
        yield figures
        for fig in figures:
            plt.close(fig)

    @staticmethod
    def _inputs():
        torch.manual_seed(3)
        profiles = [
            FeatureIOProfile(10, 7, 0.0, 3.0, 0.05, jlens_concept_mass=0.9),
            FeatureIOProfile(12, 9, 0.5, 2.0, -0.20),
            FeatureIOProfile(14, 3, 0.1, 1.0, 0.01, jlens_concept_mass=0.1),
        ]
        return profiles, torch.randn(3, 8), torch.randn(12, 8)

    def test_plotly_figure_is_embedded_with_labels_and_hover(self, shown, shown_figures) -> None:
        pytest.importorskip("plotly")
        profiles, vecs, bg = self._inputs()
        plot_decoder_projection_map(profiles, vecs, bg, feature_explanations={(10, 7): "fires on fruit"}, title="Map")
        (markup,) = shown
        assert 'id="it-decoder-map-' in markup
        markup = markup.replace("\\u002f", "/")  # plotly's JSON escapes "/" in trace text
        for label in ("L10/7", "L12/9", "L14/3", "fires on fruit", "PCA (torch_pca)", "output proj"):
            assert label in markup, label
        assert shown_figures == []

    def test_concept_mass_coloring_titles_the_colorbar(self, shown, shown_figures) -> None:
        pytest.importorskip("plotly")
        profiles, vecs, bg = self._inputs()
        plot_decoder_projection_map(profiles, vecs, bg, color_by="jlens_concept_mass")
        assert "J-space concept mass" in shown[0]

    def test_unknown_coloring_is_refused_before_rendering(self, shown, shown_figures) -> None:
        profiles, vecs, bg = self._inputs()
        with pytest.raises(ValueError, match="not a coloring"):
            plot_decoder_projection_map(profiles, vecs, bg, color_by="bogus")
        assert shown == [] and shown_figures == []

    def test_matplotlib_fallback_without_plotly(self, shown, shown_figures, monkeypatch) -> None:
        # A None entry in sys.modules makes the import fail, which is the plotly-unavailable path.
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
        profiles, vecs, bg = self._inputs()
        plot_decoder_projection_map(profiles, vecs, bg, title="Map")
        assert shown == []
        (fig,) = shown_figures
        ax = fig.axes[0]
        assert {t.get_text() for t in ax.texts} == {"L10/7", "L12/9", "L14/3"}
        assert ax.get_title().startswith("Map") and "static companion" not in ax.get_title()

    def test_static_companion_renders_both(self, shown, shown_figures) -> None:
        pytest.importorskip("plotly")
        profiles, vecs, bg = self._inputs()
        plot_decoder_projection_map(profiles, vecs, bg, static_companion=True)
        assert len(shown) == 1
        (fig,) = shown_figures
        assert fig.axes[0].get_title().endswith("(static companion)")


class TestNotebookTable:
    def test_escapes_cells_and_renders_only_http_links(self) -> None:
        html_out = notebook_ui.build_notebook_table_html(
            [["<script>x</script>", notebook_ui.NotebookLink("L20 #15589", "https://example.org/a?b=1&c=2")]],
            ["text", "link"],
            title="T <b>",
            caption="denominator & units",
        )
        assert "<script>" not in html_out and "&lt;script&gt;x&lt;/script&gt;" in html_out
        assert '<a href="https://example.org/a?b=1&amp;c=2">L20 #15589</a>' in html_out
        assert "T &lt;b&gt;" in html_out and "denominator &amp; units" in html_out
        with pytest.raises(ValueError, match="HTTP"):
            notebook_ui.build_notebook_table_html(
                [[notebook_ui.NotebookLink("x", "javascript:alert(1)")]], ["link"], title="t"
            )
        with pytest.raises(ValueError, match="column count"):
            notebook_ui.build_notebook_table_html([[1, 2]], ["only one"], title="t")
        with pytest.raises(ValueError, match="at least one column"):
            notebook_ui.build_notebook_table_html([], [], title="t")

    def test_colours_signed_cells_aligns_numbers_and_shows_a_legend(self) -> None:
        Signed = notebook_ui.Signed
        html_out = notebook_ui.build_notebook_table_html(
            [["up", Signed(1.5), 3], ["down", Signed(-0.25, "+.3f"), 2.5], ["flat", Signed(0.0), "n/a"]],
            ["row", "shift", "count"],
            title="t",
            legend=notebook_ui.SIGN_LEGEND,
            emphasize=[1],
        )
        assert f'style="color:{notebook_ui.POSITIVE_COLOUR};font-weight:600">+1.50</td>' in html_out
        assert f'style="color:{notebook_ui.NEGATIVE_COLOUR};font-weight:600">-0.250</td>' in html_out
        assert 'style="color:inherit;font-weight:600">+0.00</td>' in html_out
        assert '<td class="num">3</td>' in html_out and "<td>n/a</td>" in html_out
        assert html_out.count('<tr class="em">') == 1 and '<tr class="em"><td>down</td>' in html_out
        assert 'class="legend"' in html_out and "positive" in html_out and "negative" in html_out
        assert 'class="legend"' not in notebook_ui.build_notebook_table_html([["a"]], ["c"], title="t")
        with pytest.raises(ValueError, match="outside the table"):
            notebook_ui.build_notebook_table_html([["a"]], ["c"], title="t", emphasize=[1])

    def test_a_boolean_is_text_not_a_number(self) -> None:
        html_out = notebook_ui.build_notebook_table_html([[True, 2]], ["ok", "n"], title="t")
        assert "<td>True</td>" in html_out and '<td class="num">2</td>' in html_out

    def test_display_renders_the_built_table(self, shown) -> None:
        rows, columns = [["a", notebook_ui.Signed(-1.0)]], ["name", "shift"]
        notebook_ui.display_notebook_table(rows, columns, title="T", caption="c", emphasize=[0])
        assert shown == [notebook_ui.build_notebook_table_html(rows, columns, title="T", caption="c", emphasize=[0])]


def test_display_plotly_figure_embeds_html_the_docs_build_keeps(shown) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    notebook_ui.display_plotly_figure(go.Figure(go.Scatter(x=[8, 16], y=[0.02, 0.01])), "it-test")
    (markup,) = shown
    assert "cdn.plot.ly" in markup and 'id="it-test-' in markup


class _PieceTokenizer:
    """Token pieces by id, through ``convert_ids_to_tokens`` as a Hugging Face tokenizer gives them."""

    pieces = ["<bos>", "Paris", "Rome", "<i>", "Madrid", "Oslo"]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.pieces[token_id]


class TestDisplayTopkTokenPredictions:
    @staticmethod
    def _logits() -> tuple[torch.Tensor, torch.Tensor]:
        original = torch.full((1, 2, 6), -10.0)
        new = torch.full((1, 2, 6), -10.0)
        original[0, 0, 5] = 20.0  # an earlier position that must not be read
        original[0, 1, 1], original[0, 1, 2], original[0, 1, 3] = 3.0, 2.0, 1.0
        new[0, 1, 2], new[0, 1, 3], new[0, 1, 4] = 3.0, 2.5, 1.0
        return original, new

    @staticmethod
    def _rows(markup: str) -> list[list[str]]:
        return [_TD.findall(r) for r in _ROW.findall(markup)]

    def test_both_top_k_tables_read_the_last_position_in_order(self, shown) -> None:
        original, new = self._logits()
        notebook_ui.display_topk_token_predictions("The <capital>", original, new, _PieceTokenizer(), k=2)
        (markup,) = shown
        rows = self._rows(markup)
        assert [r[0] for r in rows] == ["Paris", "Rome", "Rome", "&lt;i&gt;"]
        p_paris = torch.softmax(original[0, -1], -1)[1].item()
        assert rows[0][1] == format_prob(p_paris, precision=3)
        assert "The &lt;capital&gt;" in markup and "Oslo" not in markup
        assert "Key Tokens" not in markup

    def test_key_tokens_sort_by_change_with_signed_bars(self, shown) -> None:
        original, new = self._logits()
        notebook_ui.display_topk_token_predictions(
            "s", original, new, _PieceTokenizer(), k=1, key_tokens=[("Paris", 1), ("Madrid", 4), ("Rome", 2)]
        )
        (markup,) = shown
        key_rows = self._rows(markup)[2:]
        assert [r[0] for r in key_rows] == ["Rome", "Madrid", "Paris"]
        p0, p1 = torch.softmax(original[0, -1], -1), torch.softmax(new[0, -1], -1)
        assert key_rows[2][1] == format_prob(p0[1].item()) and key_rows[2][2] == format_prob(p1[1].item())
        assert "-" + format_score(abs((p1[1] - p0[1]).item())) in key_rows[2][3]
        assert "#C0392B" in key_rows[2][3] and "#27AE60" in key_rows[0][3]

    def test_a_batch_of_prompts_is_refused(self, shown) -> None:
        with pytest.raises(ValueError, match="one prompt"):
            notebook_ui.display_topk_token_predictions(
                "s", torch.zeros(2, 1, 6), torch.zeros(2, 1, 6), _PieceTokenizer(), k=1
            )
        assert shown == []
