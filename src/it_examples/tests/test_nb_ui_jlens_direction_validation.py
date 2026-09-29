"""The J-lens direction validation display: a pure markup builder with a summary a notebook can assert on.

Built from a synthetic ``subspace_attribution_scores`` result rather than a model, so the tests check the
rendering contract (which numbers land where, what the fractions mean) and nothing about gemma.
"""

from __future__ import annotations

import pytest

from it_examples.utils.nb_ui_utils import build_jlens_direction_validation_html


def _attribution() -> dict:
    return {
        "token_ids": [11, 22],
        "attribution_shares": [1.5, -0.5],
        "delta_coords": [3.0, 2.0],
        "readouts": [0.5, -0.25],
        "attribution_total": 1.0,
        "predicted_delta": 1.2,
        "unexplained_remainder": 0.2,
        "basis": "jlens_folded",
    }


def test_direction_rows_carry_both_factors_and_fractions_of_explained_mass():
    markup, summary = build_jlens_direction_validation_html(_attribution(), ["Fruit-pole", "Color-pole"])
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
    markup, _ = build_jlens_direction_validation_html(
        _attribution(), ["a", "b"], finite_difference_slopes=[0.55, -0.3], measured_delta=5.7
    )
    assert "dGap/ds" in markup and "+0.5500" in markup and "-0.3000" in markup
    assert "Measured &#916;gap" in markup and "+5.7000" in markup


def test_label_and_slope_count_mismatches_are_refused_by_name():
    with pytest.raises(ValueError, match="3 direction labels for 2 attribution shares"):
        build_jlens_direction_validation_html(_attribution(), ["a", "b", "c"])
    with pytest.raises(ValueError, match="1 finite-difference slopes for 2"):
        build_jlens_direction_validation_html(_attribution(), ["a", "b"], finite_difference_slopes=[0.1])


def test_missing_factors_render_as_not_available_rather_than_guessed():
    attribution = {k: v for k, v in _attribution().items() if k not in ("delta_coords", "readouts")}
    markup, _ = build_jlens_direction_validation_html(attribution, ["a", "b"])
    assert markup.count("n/a") == 4 and "+1.5000" in markup


def test_no_feature_column_implies_a_comparison_the_machinery_cannot_make():
    """SAE-feature influence and J-lens gap shares are different vocabularies; the validation table must not set
    them side by side, and it must be titled as what it is."""
    markup, summary = build_jlens_direction_validation_html(_attribution(), ["a", "b"])
    assert "SAE" not in markup and "Node" not in markup and "comparison" not in markup.lower()
    assert "J-lens direction validation" in markup
    assert not hasattr(summary, "feature_fractions")


def test_the_table_is_sized_to_its_content_and_scrolls_inside_the_widget():
    """Measured on the docs theme: a shrinkable table overflows its box; sizing to content and scrolling inside
    the widget keeps the page from scrolling sideways."""
    markup, _ = build_jlens_direction_validation_html(_attribution(), ["a", "b"])
    style = markup[markup.index("<style>") : markup.index("</style>")]
    assert "overflow-x: auto" in style and "width: max-content" in style


def test_footer_values_span_one_cell_beside_their_label_and_never_a_numeric_column():
    """A total in share units under a slope header reads as a slope, so it must not sit in a numeric column; one
    left-aligned spanning cell keeps it beside its label without the empty gaps that padding it into the Share
    column used to leave."""
    with_slopes, _ = build_jlens_direction_validation_html(
        _attribution(), ["a", "b"], finite_difference_slopes=[0.55, -0.3]
    )
    # One cell covering every column bar the label: 5 with the slope column present, 4 without.
    assert '<td class="lbl" colspan="5">+1.0000</td>' in with_slopes
    without, _ = build_jlens_direction_validation_html(_attribution(), ["a", "b"])
    assert '<td class="lbl" colspan="4">+1.0000</td>' in without
    # The gapped form is what the change removed; an empty padding cell must not come back.
    for markup in (with_slopes, without):
        assert "<td colspan=" not in markup, "an empty padding cell is the layout this replaced"


def test_footer_cell_count_matches_the_header_width():
    """A span that does not total the header width silently skews every footer row, and the browser hides it by
    stretching the table rather than erroring."""
    import re

    for slopes, expected in (([0.55, -0.3], 6), (None, 5)):
        markup, _ = build_jlens_direction_validation_html(_attribution(), ["a", "b"], finite_difference_slopes=slopes)
        left = markup.split('<div class="scroll">')[1]
        header_cols = len(re.findall(r"<th\b", left.split("</thead>")[0]))
        assert header_cols == expected, f"{header_cols} header columns, expected {expected}"
        footer_row = re.search(r'<tr class="total">.*?</tr>', left)
        # Not just for the type checker: if no footer row rendered at all, say so, rather than dying on
        # an attribute of None and reporting a crash where the real finding is a missing row.
        assert footer_row is not None, "no footer row rendered in the directions table"
        footer = footer_row.group(0)
        # Read colspan out of each tag rather than one combined pattern: an optional group after a lazy
        # prefix silently never captures, which counts every cell as width 1 and passes only by accident.
        width = 0
        for tag in re.findall(r"<td\b[^>]*>", footer):
            span = re.search(r'colspan="(\d+)"', tag)
            width += int(span.group(1)) if span else 1
        assert width == header_cols, f"footer spans {width} columns, header has {header_cols}"
