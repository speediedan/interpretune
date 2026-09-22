"""The attribution-comparison display: a pure markup builder with a summary a notebook can assert on.

Built from a synthetic ``subspace_attribution_scores`` result rather than a model, so the tests check the
rendering contract (which numbers land where, what the fractions mean) and nothing about gemma.
"""

from __future__ import annotations

import pytest

from it_examples.utils.nb_ui_utils import build_attribution_comparison_html


def _attribution() -> dict:
    return {
        "token_ids": [11, 22],
        "attribution_shares": [1.5, -0.5],
        "delta_coords": [3.0, 2.0],
        "readouts": [0.5, -0.25],
        "attribution_total": 1.0,
        "predicted_delta": 1.2,
        "unexplained_remainder": 0.2,
        "basis": "jlens_norm_aware",
    }


FEATURES = [(24, 7, 100), (25, 7, 200), (24, 7, 300)]
SCORES = [0.5, -1.0, 0.25]


def test_direction_rows_carry_both_factors_and_fractions_of_explained_mass():
    markup, summary = build_attribution_comparison_html(_attribution(), ["Fruit-pole", "Color-pole"])
    assert summary.direction_labels == ("Fruit-pole", "Color-pole")
    assert summary.direction_shares == (1.5, -0.5)
    # |1.5| / (|1.5| + |-0.5|) and |-0.5| / 2.0: fractions are of the explained MASS, so a negative share
    # still counts as explaining, in the other direction.
    assert summary.direction_fractions == pytest.approx((0.75, 0.25))
    assert summary.attribution_total == 1.0 and summary.predicted_delta == 1.2
    assert summary.unexplained_remainder == pytest.approx(0.2)
    for text in ("+3.0000", "+0.5000", "+1.5000", "75.0%", "-0.5000", "25.0%", "Remainder", "+0.2000"):
        assert text in markup, text
    assert "Measured &#916;gap" not in markup and "dGap/ds" not in markup


def test_optional_columns_render_only_when_given():
    markup, _ = build_attribution_comparison_html(
        _attribution(), ["a", "b"], finite_difference_slopes=[0.55, -0.3], measured_delta=5.7
    )
    assert "dGap/ds" in markup and "+0.5500" in markup and "-0.3000" in markup
    assert "Measured &#916;gap" in markup and "+5.7000" in markup


def test_features_rank_by_magnitude_with_links_signs_and_explanations():
    markup, summary = build_attribution_comparison_html(
        _attribution(),
        ["a", "b"],
        features=FEATURES,
        feature_scores=SCORES,
        feature_explanations={(25, 200): "citrus fruit words"},
        neuronpedia_model="gemma-2-2b",
        neuronpedia_set="gemmascope-transcoder-16k",
        neuronpedia_base_url="https://www.neuronpedia.org/",
        top_n=2,
    )
    # top_n=2 keeps the two largest |score|: -1.0 (L25 f200) then 0.5 (L24 f100); 0.25 is dropped.
    assert summary.feature_fractions == pytest.approx((1.0 / 1.5, 0.5 / 1.5))
    assert markup.index("f200") < markup.index("f100") if "f200" in markup else True
    assert 'href="https://www.neuronpedia.org/gemma-2-2b/25-gemmascope-transcoder-16k/200"' in markup
    assert "citrus fruit words" in markup
    assert "300" not in markup.split("Explanation")[-1].split("</tbody>")[0]
    # The negative feature's sign cell is the red minus, the positive one the green plus.
    assert 'color:#d1242f;font-weight:600">−' in markup and 'color:#1a7f37;font-weight:600">+' in markup


def test_label_and_slope_count_mismatches_are_refused_by_name():
    with pytest.raises(ValueError, match="3 direction labels for 2 attribution shares"):
        build_attribution_comparison_html(_attribution(), ["a", "b", "c"])
    with pytest.raises(ValueError, match="1 finite-difference slopes for 2"):
        build_attribution_comparison_html(_attribution(), ["a", "b"], finite_difference_slopes=[0.1])


def test_missing_factors_render_as_not_available_rather_than_guessed():
    attribution = {k: v for k, v in _attribution().items() if k not in ("delta_coords", "readouts")}
    markup, _ = build_attribution_comparison_html(attribution, ["a", "b"])
    assert markup.count("n/a") == 4 and "+1.5000" in markup


def test_the_columns_are_sized_to_their_content_and_the_pair_scrolls_rather_than_colliding():
    """The layout contract, pinned because breaking it is invisible in a wide notebook and ugly in the docs.

    Measured on the docs theme: with shrinkable flex items (``flex: 1``, ``flex-basis: 0``) the two columns
    settle at half the content width, each table overflows its own box, and the two render on top of one
    another. Sizing each column to its content and letting the PAIR scroll is what fixes it; the scroll lives
    inside the widget, so the page never scrolls sideways.
    """
    markup, _ = build_attribution_comparison_html(_attribution(), ["a", "b"], features=FEATURES, feature_scores=SCORES)
    style = markup[markup.index("<style>") : markup.index("</style>")]
    assert "flex-wrap: nowrap" in style and "overflow-x: auto" in style, "the pair must scroll, not wrap or collide"
    assert "flex: 0 0 auto" in style and "width: max-content" in style, "a column must not shrink below its table"
    assert "min-width: 300px" not in style, "a fixed min-width is what let the tables overflow their columns"


def test_footer_values_span_one_cell_beside_their_label_and_never_a_numeric_column():
    """A total in share units under a slope header reads as a slope, so it must not sit in a numeric column; one
    left-aligned spanning cell keeps it beside its label without the empty gaps that padding it into the Share
    column used to leave."""
    with_slopes, _ = build_attribution_comparison_html(
        _attribution(), ["a", "b"], finite_difference_slopes=[0.55, -0.3]
    )
    # One cell covering every column bar the label: 5 with the slope column present, 4 without.
    assert '<td class="lbl" colspan="5">+1.0000</td>' in with_slopes
    without, _ = build_attribution_comparison_html(_attribution(), ["a", "b"])
    assert '<td class="lbl" colspan="4">+1.0000</td>' in without
    # The gapped form is what the change removed; an empty padding cell must not come back.
    for markup in (with_slopes, without):
        assert "<td colspan=" not in markup, "an empty padding cell is the layout this replaced"


def test_footer_cell_count_matches_the_header_width():
    """A span that does not total the header width silently skews every footer row, and the browser hides it by
    stretching the table rather than erroring."""
    import re

    for slopes, expected in (([0.55, -0.3], 6), (None, 5)):
        markup, _ = build_attribution_comparison_html(_attribution(), ["a", "b"], finite_difference_slopes=slopes)
        left = markup.split('<div class="col-header"')[1]
        header_cols = len(re.findall(r"<th\b", left.split("</thead>")[0]))
        assert header_cols == expected, f"{header_cols} header columns, expected {expected}"
        footer = re.search(r'<tr class="total">.*?</tr>', left).group(0)
        # Read colspan out of each tag rather than one combined pattern: an optional group after a lazy
        # prefix silently never captures, which counts every cell as width 1 and passes only by accident.
        width = 0
        for tag in re.findall(r"<td\b[^>]*>", footer):
            span = re.search(r'colspan="(\d+)"', tag)
            width += int(span.group(1)) if span else 1
        assert width == header_cols, f"footer spans {width} columns, header has {header_cols}"
