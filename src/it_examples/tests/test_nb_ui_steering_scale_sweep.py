"""The steering scale-sweep table: signed, colour-coded shifts and a flip column, checked without a kernel."""

from __future__ import annotations

import pytest

from it_examples.utils.nb_ui_utils import build_steering_scale_sweep_html
from it_examples.utils.steering_demo_helpers import SteeringScalePoint


def _point(arm: str, scale: float, pre: tuple[float, float], post: tuple[float, float]) -> SteeringScalePoint:
    return SteeringScalePoint(arm, scale, pre, post, (0.10, 0.60), (0.30, 0.20))


def test_shifts_are_signed_and_coloured_by_convention_and_flips_are_reported():
    points = [
        _point("J-space patch", 1.0, (25.0, 27.0), (27.5, 25.0)),  # gap -2 -> +2.5: flipped
        _point("direct-hook add", 1.0, (25.0, 27.0), (25.5, 27.0)),  # gap -2 -> -1.5: not flipped
    ]
    markup = build_steering_scale_sweep_html(points, ["Fruit", "Color"])
    assert "&#916; gap (Fruit &#8722; Color)" in markup
    # J-space row: +2.500 Fruit logit (green), -2.000 Color logit (red), +4.500 gap, +20.00 pp / -40.00 pp.
    for cell in (
        'color:#1a7f37;font-weight:600">+2.500',
        'color:#d1242f;font-weight:600">-2.000',
        '">+4.500',
        '">+20.00',
        'color:#d1242f;font-weight:600">-40.00',
    ):
        assert cell in markup, cell
    rows = markup.split("<tbody>")[1].split("</tr>")
    assert rows[0].endswith("<td>yes</td>") and rows[1].endswith("<td>no</td>")
    # A zero shift carries no colour claim.
    assert 'color:inherit;font-weight:600">+0.000' in markup


def test_a_gap_that_lands_on_zero_is_a_tie_not_a_flip():
    """Measured in the gemma-2-2b render: a J-space run moved the gap from -1.875 by exactly +1.875."""
    markup = build_steering_scale_sweep_html([_point("J-space patch", 20, (25.0, 26.875), (28.0, 28.0))], ["a", "b"])
    assert markup.split("<tbody>")[1].split("</tr>")[0].endswith("<td>no</td>")


def test_labels_are_escaped_and_the_label_count_is_enforced():
    markup = build_steering_scale_sweep_html([_point("<arm>", 5, (0.0, 1.0), (1.0, 0.0))], ["<a>", "b"])
    assert "&lt;arm&gt;" in markup and "&lt;a&gt;" in markup and "<arm>" not in markup
    with pytest.raises(ValueError, match="expected two target token labels, got 3"):
        build_steering_scale_sweep_html([], ["a", "b", "c"])
