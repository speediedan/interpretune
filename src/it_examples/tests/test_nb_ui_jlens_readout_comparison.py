"""The J-space readout comparison display: a pure markup builder with a summary a notebook can assert on.

Built from synthetic pre/post top-*k* pairs rather than a model, so the tests check the rendering
contract (which numbers land where, what the movement column means) and nothing about gemma.
"""

from __future__ import annotations

from it_examples.utils.nb_ui_utils import build_jlens_readout_comparison_html


def test_pre_post_scores_land_in_their_columns_with_rank_moves():
    pre = [("France", 12.5), ("Paris", 11.0), ("Rome", 9.25)]
    post = [("Paris", 13.75), ("France", 12.0), ("Madrid", 8.5)]
    markup, summary = build_jlens_readout_comparison_html(pre, post, layer=25, basis="jlens_norm_aware")
    assert summary.pre_labels == ("France", "Paris", "Rome")
    assert summary.post_labels == ("Paris", "France", "Madrid")
    assert summary.pre_scores == (12.5, 11.0, 9.25)
    assert summary.post_scores == (13.75, 12.0, 8.5)
    # Paris 2 -> 1 is +1, France 1 -> 2 is -1, Madrid was outside pre top-k.
    assert summary.rank_moves == (1, -1, None)
    assert summary.layer == 25 and summary.basis == "jlens_norm_aware"
    for text in ("+12.50", "+11.00", "+13.75", "+12.00", "+1", "-1", "new", "layer 25"):
        assert text in markup, text


def test_tokens_are_escaped_and_uneven_sides_render():
    pre = [("<b>FR</b>", 1.0)]
    post = [("<b>FR</b>", 2.0), ("&amp;", -0.5)]
    markup, summary = build_jlens_readout_comparison_html(pre, post, layer=3, basis="embed")
    assert "<b>FR</b>" not in markup and "&lt;b&gt;FR&lt;/b&gt;" in markup
    assert summary.rank_moves == (0, None)
