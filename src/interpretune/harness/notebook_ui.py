"""Notebook display helpers that several example tutorials share.

Supported notebook-support API: each helper renders HTML (or an interactive figure) for a result the analysis ops
produce, so tutorials present target gaps, top features, steering results and lens readouts the same way. Adapted
from ``circuit_tracer.utils.demo_utils``. Helpers that only one tutorial uses stay with that tutorial.
"""

from __future__ import annotations

import html
from typing import Any, Mapping, NamedTuple, Sequence

import torch
from IPython.display import HTML, display

from interpretune.harness.display import format_score


# Counter for unique plotly div ids (stable within a run, so artifacts stay diffable).
_decoder_map_html_counter = [0]


def format_prob(p: float, *, precision: int = 4) -> str:
    """Format a probability with automatic scientific notation for small values.

    Values >= 1e-3 are shown as percentages; smaller values use scientific notation.
    """
    if p >= 1e-3:
        return f"{p * 100:.{precision}f}%"
    return f"{p:.{min(precision, 2)}e}"


def _ensure_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    """The vocab logits at the last sequence position, from ``(V,)``, ``(seq, V)`` or ``(1, seq, V)``.

    Any other shape is refused: a batch of several prompts has no single answer row to read.
    """
    if logits.dim() == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.dim() == 2:
        logits = logits[-1]
    if logits.dim() != 1:
        raise ValueError(
            f"expected logits of shape (V,), (seq, V) or (1, seq, V), got {tuple(logits.shape)}: "
            "pass one prompt's logits"
        )
    return logits.float()


def resolve_feature_explanations(
    *,
    model_id: str,
    source_set: str,
    feature_tuples: Sequence[tuple[int, int]],
    base_url: str = "https://www.neuronpedia.org",
    preferred_type_name: str | None = None,
    timeout_seconds: int = 15,
) -> dict[tuple[int, int], str]:
    """Best-effort explanation text for ``(layer, feature_index)`` tuples via the feature API.

    Works against both public neuronpedia.org and a local Neuronpedia dev webapp ``base_url``.
    Fetch failures and features without explanations simply yield no entry, so callers can
    always render (an empty cell) without guarding.
    """
    from interpretune.utils.neuronpedia_explanations import feature_tuples_to_feature_refs, fetch_feature_payload

    unique_tuples = list(dict.fromkeys((int(layer), int(feat)) for layer, feat in feature_tuples))
    if not unique_tuples:
        return {}
    feature_refs = feature_tuples_to_feature_refs(
        model_id=model_id,
        source_set=source_set,
        feature_tuples=unique_tuples,
        base_url=base_url,
    )
    explanations: dict[tuple[int, int], str] = {}
    for feature_tuple, feature_ref in zip(unique_tuples, feature_refs):
        try:
            payload = fetch_feature_payload(feature_ref, timeout_seconds=timeout_seconds)
        except Exception:
            continue
        rows = payload.get("explanations")
        if not isinstance(rows, list):
            continue
        candidate_rows = [row for row in rows if isinstance(row, Mapping)]
        if preferred_type_name:
            preferred_rows = [
                row for row in candidate_rows if str(row.get("typeName") or "").strip() == preferred_type_name
            ]
            candidate_rows = preferred_rows or candidate_rows
        for row in candidate_rows:
            description = row.get("description")
            if isinstance(description, str) and description.strip():
                explanations[feature_tuple] = " ".join(description.split())
                break
    return explanations


def display_top_features_comparison(
    feature_sets: dict[str, list[tuple[int, int, int]]],
    scores_sets: dict[str, list[float]] | None = None,
    neuronpedia_model: str | None = None,
    neuronpedia_set: str = "gemmascope-transcoder-16k",
    neuronpedia_base_url: str = "https://www.neuronpedia.org",
    show_score_sign: bool = False,
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
) -> None:
    """Display top features from multiple attribution configurations side by side.

    Args:
        feature_sets: Mapping from config label to list of ``(layer, pos, feat_idx)`` tuples.
        scores_sets: Optional mapping from config label to list of attribution scores.
        neuronpedia_model: Neuronpedia model slug (e.g. ``"gemma-2-2b"``).
            When provided, feature indices become clickable links.
        neuronpedia_set: Neuronpedia set name.
        neuronpedia_base_url: Base URL for Neuronpedia feature links.
        show_score_sign: Render directional scores as separate Sign / ``|Score|`` columns
            (use with signed score sources such as ``signed_influence``).
        feature_explanations: Optional ``(layer, feature_index) -> explanation`` mapping
            (e.g. from :func:`resolve_feature_explanations`); adds an Explanation column
            with empty cells for unmapped features.
    """
    labels = list(feature_sets.keys())
    colors = ["#2471A3", "#27AE60", "#8E44AD", "#E67E22", "#C0392B", "#16A085"]
    resolved_base_url = neuronpedia_base_url.rstrip("/")

    style = """
    <style>
    .features-cmp {
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }
    .features-cmp .col {
        flex: 1;
        min-width: 220px;
    }
    .features-cmp .col-header {
        font-weight: bold;
        font-size: 14px;
        padding: 4px 8px;
        border-radius: 3px;
        color: white;
        margin-bottom: 6px;
    }
    .features-cmp table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .features-cmp th, .features-cmp td {
        text-align: left;
        padding: 3px 6px;
        border: 1px solid rgba(150,150,150,0.5);
    }
    .features-cmp th {
        background-color: rgba(200,200,200,0.3);
        font-weight: bold;
    }
    .features-cmp .monospace { font-family: monospace; }
    .features-cmp a.np-link {
        color: inherit;
        text-decoration: none;
        border-bottom: 1px dashed rgba(150,150,150,0.6);
    }
    .features-cmp a.np-link:hover {
        color: #2980B9;
        border-bottom-style: solid;
    }
    </style>
    """

    body = '<div class="features-cmp">'
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]
        features = feature_sets[label]
        scores = scores_sets.get(label) if scores_sets else None
        body += '<div class="col">'
        body += f'<div class="col-header" style="background-color: {color};">{html.escape(label)}</div>'
        body += "<table><thead><tr><th>#</th><th>Node</th>"
        if scores is not None:
            if show_score_sign:
                body += "<th>Sign</th><th>|Score|</th>"
            else:
                body += "<th>Score</th>"
        if feature_explanations is not None:
            body += "<th>Explanation</th>"
        body += "</tr></thead><tbody>"
        for j, (layer, pos, feat_idx) in enumerate(features):
            score_cell = ""
            if scores is not None:
                score_value = float(scores[j])
                if show_score_sign:
                    # Colour the sign: the signed-influence selection can steer with negative-signed
                    # features, so direction is the column a reader scans for first. Inline styles
                    # (not a class) because notebook HTML output is frequently rendered without the
                    # surrounding stylesheet; these hues stay legible on light and dark backgrounds.
                    if score_value > 0:
                        score_sign, sign_colour = "+", "#1a7f37"
                    elif score_value < 0:
                        score_sign, sign_colour = "−", "#d1242f"
                    else:
                        score_sign, sign_colour = "0", "inherit"
                    score_cell = (
                        f'<td style="color:{sign_colour};font-weight:600">{score_sign}</td>'
                        f"<td>{format_score(abs(score_value))}</td>"
                    )
                else:
                    score_cell = f"<td>{format_score(score_value)}</td>"
            if neuronpedia_model is not None:
                np_url = (
                    f"{html.escape(resolved_base_url)}/{html.escape(neuronpedia_model)}/"
                    f"{layer}-{html.escape(neuronpedia_set)}/{feat_idx}"
                )
                feat_link = (
                    f'<a class="np-link" href="{np_url}" target="_blank" title="View on Neuronpedia">{feat_idx}</a>'
                )
            else:
                feat_link = str(feat_idx)
            node_cell = f'<td class="monospace">({layer},&#8239;{pos},&#8239;{feat_link})</td>'
            explanation_cell = ""
            if feature_explanations is not None:
                explanation = feature_explanations.get((int(layer), int(feat_idx)), "")
                explanation_cell = f"<td>{html.escape(explanation)}</td>"
            body += f"<tr><td>{j + 1}</td>{node_cell}{score_cell}{explanation_cell}</tr>"
        body += "</tbody></table></div>"
    body += "</div>"

    display(HTML(style + body))


class JlensReadoutComparisonSummary(NamedTuple):
    """What the pre/post J-space readout table asserts, in notebook-assertable form."""

    pre_labels: tuple[str, ...]
    post_labels: tuple[str, ...]
    pre_scores: tuple[float, ...]
    post_scores: tuple[float, ...]
    rank_moves: tuple[int | None, ...]
    layer: int
    basis: str


def build_jlens_readout_comparison_html(
    pre_top: Sequence[tuple[str, float]],
    post_top: Sequence[tuple[str, float]],
    *,
    layer: int,
    basis: str,
) -> tuple[str, JlensReadoutComparisonSummary]:
    """Build a pre/post J-space readout comparison table.

    Each side is a ranked ``(token_label, readout_score)`` sequence as produced by the readout used in
    the steering demos (folded-norm path, top-*k* by logit). Ranks are 1-based in the movement column:
    ``=`` kept its rank, ``+n``/``-n`` moved up/down *n* places, ``new`` was outside the pre top-*k*.

    Returns the markup plus a summary a notebook can assert on.
    """
    pre_labels = tuple(t for t, _ in pre_top)
    post_labels = tuple(t for t, _ in post_top)
    pre_scores = tuple(float(s) for _, s in pre_top)
    post_scores = tuple(float(s) for _, s in post_top)
    pre_rank = {t: i + 1 for i, t in enumerate(pre_labels)}
    rank_moves: tuple[int | None, ...] = tuple(
        None if t not in pre_rank else pre_rank[t] - (i + 1) for i, t in enumerate(post_labels)
    )

    def _move_text(move: int | None) -> str:
        if move is None:
            return "new"
        if move == 0:
            return "="
        return f"+{move}" if move > 0 else f"{move}"

    rows = ""
    for i in range(max(len(pre_top), len(post_top))):
        pre_t = pre_labels[i] if i < len(pre_labels) else ""
        pre_s = f"{pre_scores[i]:+.2f}" if i < len(pre_scores) else ""
        post_t = post_labels[i] if i < len(post_labels) else ""
        post_s = f"{post_scores[i]:+.2f}" if i < len(post_scores) else ""
        move = _move_text(rank_moves[i]) if i < len(rank_moves) else ""
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        rows += (
            f'<tr class="{row_class}">'
            f'<td style="text-align:right;">{i + 1}</td>'
            f'<td class="monospace" title="{html.escape(pre_t)}">{html.escape(pre_t)}</td>'
            f'<td style="text-align:right;">{pre_s}</td>'
            f'<td class="monospace" title="{html.escape(post_t)}">{html.escape(post_t)}</td>'
            f'<td style="text-align:right;">{post_s}</td>'
            f'<td style="text-align:center;">{move}</td>'
            f"</tr>\n"
        )

    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:640px;margin-bottom:10px;font-size:13px;">
        <div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;
            border-radius:3px;background:#555;color:white;display:inline-block;">
            {html.escape(f"J-space readout @ answer position (layer {layer}, basis {basis})")}</div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:right;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">#</th>
                    <th style="text-align:left;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Pre token</th>
                    <th style="text-align:right;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Pre</th>
                    <th style="text-align:left;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Post token</th>
                    <th style="text-align:right;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Post</th>
                    <th style="text-align:center;padding:3px 6px;
                        border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Move</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    summary = JlensReadoutComparisonSummary(
        pre_labels=pre_labels,
        post_labels=post_labels,
        pre_scores=pre_scores,
        post_scores=post_scores,
        rank_moves=rank_moves,
        layer=int(layer),
        basis=str(basis),
    )
    return markup, summary


def display_jlens_readout_comparison(
    pre_top: Sequence[tuple[str, float]],
    post_top: Sequence[tuple[str, float]],
    *,
    layer: int,
    basis: str,
) -> JlensReadoutComparisonSummary:
    """Render :func:`build_jlens_readout_comparison_html` and return its summary (see that function for the
    columns)."""
    markup, summary = build_jlens_readout_comparison_html(pre_top, post_top, layer=layer, basis=basis)
    display(HTML(markup))
    return summary


def best_variant_token_ids(
    tokenizer: Any,
    tokens: Sequence[str],
    reference_logits: torch.Tensor,
    *,
    use_best_variant: bool = True,
) -> list[int]:
    """Resolve tokens to the plain-vs-leading-space variant the model favors under reference logits.

    SentencePiece/BPE vocabularies carry distinct ids for ``"Fruit"`` vs ``" Fruit"`` (``▁Fruit``);
    which variant a model concentrates mass on depends on the prompt position (post-newline chat
    answer positions favor the bare variant, mid-sentence positions the space-prefixed one). By
    default each token resolves to whichever variant carries the higher reference (typically
    pre-intervention) logit; pass ``use_best_variant=False`` to always take the bare spelling.
    """
    reference_1d = _ensure_1d_logits(reference_logits)
    resolved: list[int] = []
    for token in tokens:
        variant_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in (token, " " + token)]
        if use_best_variant:
            resolved.append(max(variant_ids, key=lambda i: float(reference_1d[i])))
        else:
            resolved.append(variant_ids[0])
    return resolved


def display_target_gap(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_a: tuple[str, int],
    target_b: tuple[str, int],
    title: str = "Target gap",
) -> tuple[float, float]:
    """Render a consolidated pre/post table for a two-token target contrast and return the gaps.

    One reusable display for any pre/post logit (or activation-derived logit) gap — feature-mediated
    interventions, direct hook-tensor steering, ablations, etc. Shows each target token's pre/post
    probability and logit with its per-token delta, plus a highlighted ``A − B`` gap row. Returns
    ``(pre_gap, post_gap)`` so callers can assert on steering outcomes without recomputing.
    """
    pre_1d = _ensure_1d_logits(pre_logits)
    post_1d = _ensure_1d_logits(post_logits)
    pre_probs = torch.softmax(pre_1d, dim=-1)
    post_probs = torch.softmax(post_1d, dim=-1)

    (label_a, id_a), (label_b, id_b) = target_a, target_b
    pre_gap = float(pre_1d[id_a] - pre_1d[id_b])
    post_gap = float(post_1d[id_a] - post_1d[id_b])

    def _cell(value: str, *, align: str = "right", bold: bool = False) -> str:
        weight = "font-weight:bold;" if bold else ""
        cell_style = f"text-align:{align};padding:3px 6px;border:1px solid rgba(150,150,150,0.5);{weight}"
        return f'<td style="{cell_style}">{value}</td>'

    rows = ""
    for i, (label, tid) in enumerate(((label_a, id_a), (label_b, id_b))):
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        delta = float(post_1d[tid] - pre_1d[tid])
        rows += (
            f'<tr class="{row_class}">'
            + _cell(html.escape(label), align="left")
            + _cell(format_prob(float(pre_probs[tid]), precision=3))
            + _cell(format_prob(float(post_probs[tid]), precision=3))
            + _cell(f"{float(pre_1d[tid]):.4f}")
            + _cell(f"{float(post_1d[tid]):.4f}")
            + _cell(f"{delta:+.4f}")
            + "</tr>\n"
        )
    rows += (
        '<tr style="background:rgba(120,180,240,0.15);">'
        + _cell(f"Gap ({html.escape(label_a)} − {html.escape(label_b)})", align="left", bold=True)
        + _cell("")
        + _cell("")
        + _cell(f"{pre_gap:+.4f}", bold=True)
        + _cell(f"{post_gap:+.4f}", bold=True)
        + _cell(f"{post_gap - pre_gap:+.4f}", bold=True)
        + "</tr>\n"
    )

    title_html = (
        f'<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;'
        f'border-radius:3px;background:#555;color:white;display:inline-block;">'
        f"{html.escape(title)}</div>"
        if title
        else ""
    )
    header_cell = (
        'style="text-align:right;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);'
        'background:rgba(200,200,200,0.3);"'
    )
    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:640px;margin-bottom:10px;font-size:13px;">
        {title_html}
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left;padding:3px 6px;border:1px solid rgba(150,150,150,0.5);
                        background:rgba(200,200,200,0.3);">Token</th>
                    <th {header_cell}>Pre prob</th>
                    <th {header_cell}>Post prob</th>
                    <th {header_cell}>Pre logit</th>
                    <th {header_cell}>Post logit</th>
                    <th {header_cell}>Δ</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    display(HTML(markup))
    return pre_gap, post_gap


class SteeringDisplaySummary(NamedTuple):
    """Values extracted/derived from an ``intervention_from_concept`` result by ``display_steering_results``."""

    steered_features: list[tuple[int, int, int]]
    target_ids: tuple[int, int]
    pre_gap: float
    post_gap: float
    direction: torch.Tensor
    pre_logits: torch.Tensor
    post_logits: torch.Tensor


def display_steering_results(
    pipeline_results: Any,
    tokenizer: Any,
    target_tokens: Sequence[str],
    *,
    neuronpedia_model: str | None = None,
    neuronpedia_set: str = "gemmascope-transcoder-16k",
    neuronpedia_base_url: str = "https://www.neuronpedia.org",
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
    min_layer: int | None = None,
    use_best_variant: bool = True,
    features_label: str = "Steered Features (signed influence)",
    gap_title: str = "Feature-mediated steering — target gap",
) -> SteeringDisplaySummary:
    """One-call display + extraction for an ``it.intervention_from_concept`` result.

    Composes the underlying helpers — the linked/signed steered-features table
    (:func:`display_top_features_comparison`) and the consolidated pre/post target-gap table
    (:func:`display_target_gap`) — after extracting the steered features, unit concept direction,
    flattened pre/post logits, and best-variant target token ids from ``pipeline_results``. Returns a
    :class:`SteeringDisplaySummary` so callers can assert on the gaps (``post_gap > pre_gap``) and
    reuse the direction/target ids in later phases without re-deriving anything.

    Args:
        pipeline_results: ``AnalysisBatch`` returned by ``it.intervention_from_concept`` (needs
            ``top_feature_ids``, ``top_feature_scores``, ``concept_direction``, and
            ``pre/post_intervention_logits``).
        tokenizer: Tokenizer used for best-variant target-token resolution.
        target_tokens: Two-token target contrast ``(A, B)`` — the displayed/returned gap is ``A − B``.
        min_layer: When provided, validates every steered feature's layer is ``>= min_layer``
            (the ``FeatureSelectionSpec.layer_slice`` contract).
        use_best_variant: Resolve each target token to the plain-vs-leading-space variant the model
            favors pre-intervention (see :func:`best_variant_token_ids`).
    """
    steered_features = [tuple(f.tolist()) for f in pipeline_results.top_feature_ids]
    if min_layer is not None and any(f[0] < min_layer for f in steered_features):
        raise ValueError(
            f"layer_slice constraint violated: steered features {steered_features} include layers < {min_layer}"
        )
    direction = pipeline_results.concept_direction.detach().float().cpu().reshape(-1)
    pre_logits = _ensure_1d_logits(pipeline_results.pre_intervention_logits.float().cpu())
    post_logits = _ensure_1d_logits(pipeline_results.post_intervention_logits.float().cpu())

    target_a_id, target_b_id = best_variant_token_ids(
        tokenizer, list(target_tokens)[:2], pre_logits, use_best_variant=use_best_variant
    )
    display_top_features_comparison(
        {features_label: steered_features},
        {features_label: pipeline_results.top_feature_scores.tolist()},
        neuronpedia_model=neuronpedia_model,
        neuronpedia_set=neuronpedia_set,
        neuronpedia_base_url=neuronpedia_base_url,
        show_score_sign=True,
        feature_explanations=feature_explanations,
    )
    pre_gap, post_gap = display_target_gap(
        pre_logits,
        post_logits,
        (str(target_tokens[0]), target_a_id),
        (str(target_tokens[1]), target_b_id),
        title=gap_title,
    )
    return SteeringDisplaySummary(
        steered_features=steered_features,
        target_ids=(target_a_id, target_b_id),
        pre_gap=pre_gap,
        post_gap=post_gap,
        direction=direction,
        pre_logits=pre_logits,
        post_logits=post_logits,
    )


def display_concept_positions(tokenizer: Any, token_ids: Sequence[int], concept_positions: set[int]) -> None:
    """Render the prompt as token chips with concept-token positions highlighted."""
    chips = []
    for pos, tid in enumerate(token_ids):
        token_text = html.escape(tokenizer.decode([int(tid)]))
        if pos in concept_positions:
            style = "background:#2471A3;color:white;font-weight:bold;"
        else:
            style = "background:rgba(200,200,200,0.25);"
        chips.append(
            f'<span title="pos {pos}" style="{style}padding:2px 5px;margin:1px;border-radius:3px;'
            f'font-family:monospace;font-size:12px;display:inline-block;">{token_text}</span>'
        )
    markup = (
        '<div style="font-family:system-ui,-apple-system,sans-serif;font-size:13px;margin-bottom:10px;">'
        '<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;border-radius:3px;'
        'background:#555;color:white;display:inline-block;">Prompt tokens — concept positions highlighted</div>'
        f'<div style="line-height:2;">{"".join(chips)}</div>'
        f"<div>concept-token positions: <b>{sorted(concept_positions)}</b></div></div>"
    )
    display(HTML(markup))


def display_feature_decoupling_table(
    profiles: Sequence[Any],
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
    *,
    target_label: str = "A−B",
    title: str = "Feature input/output decoupling",
) -> None:
    """Render :class:`~interpretune.harness.notebook_inputs.FeatureIOProfile` rows as a styled table.

    Rows sort by ``|output_projection|`` (the steering-relevant quantity). The decoupling signature
    is a large ``|output proj|`` with a near-zero input concept share; a high input share with a
    projection *against* the concept marks a suppressor-motif exemplar.
    """
    header_cell = 'style="padding:3px 6px;border:1px solid rgba(150,150,150,0.5);background:rgba(200,200,200,0.3);"'

    def _cell(value: str, *, align: str = "right", bold: bool = False) -> str:
        weight = "font-weight:bold;" if bold else ""
        cell_style = f"text-align:{align};padding:3px 6px;border:1px solid rgba(150,150,150,0.5);{weight}"
        return f'<td style="{cell_style}">{value}</td>'

    rows = ""
    show_jlens = any(getattr(prof, "jlens_signature", None) for prof in profiles)
    for i, prof in enumerate(sorted(profiles, key=lambda r: -abs(r.output_projection))):
        explanation = (feature_explanations or {}).get((prof.layer, prof.feature), "")
        decoupled = abs(prof.output_projection) >= 0.03 and prof.input_concept_share < 0.05
        suppressor = prof.input_concept_share >= 0.3 and prof.output_projection < -0.03
        marker = "decoupled" if decoupled else ("suppressor-motif" if suppressor else "")
        row_class = "even-row" if i % 2 == 0 else "odd-row"
        cells = (
            f'<tr class="{row_class}">'
            + _cell(f"L{prof.layer}/{prof.feature}", align="left")
            + _cell(f"{prof.input_concept_share:.3f}", bold=prof.input_concept_share == 0.0)
            + _cell(f"{prof.activation_mass:.2f}")
            + _cell(f"{prof.output_projection:+.4f}", bold=abs(prof.output_projection) >= 0.03)
        )
        if show_jlens:
            signature = getattr(prof, "jlens_signature", None) or ()
            mass = getattr(prof, "jlens_concept_mass", None)
            top_tokens = ", ".join(tok for tok, _ in signature[:3])
            mass_text = f"{mass:.2f}" if mass is not None else "—"
            cells += _cell(html.escape(f"{top_tokens} ({mass_text})"), align="left")
        # truncate before escaping, so the cut never splits an entity
        cells += _cell(html.escape(marker), align="left", bold=bool(marker)) + _cell(
            html.escape(str(explanation)[:80]), align="left"
        )
        rows += cells + "</tr>\n"

    title_html = (
        f'<div style="font-weight:bold;font-size:14px;margin-bottom:4px;padding:4px 6px;'
        f'border-radius:3px;background:#555;color:white;display:inline-block;">{html.escape(title)}</div>'
    )
    jlens_header = f"<th {header_cell}>J-lens readout (concept mass)</th>" if show_jlens else ""
    markup = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:880px;margin-bottom:10px;font-size:13px;">
        {title_html}
        <table style="width:100%;border-collapse:collapse;">
            <thead><tr>
                <th {header_cell}>Feature</th>
                <th {header_cell}>Input concept share</th>
                <th {header_cell}>Act mass</th>
                <th {header_cell}>Output proj ({html.escape(target_label)})</th>
                {jlens_header}<th {header_cell}>Signature</th>
                <th {header_cell}>Explanation</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <div style="font-size:12px;color:#666;">decoupled = large |output proj| with ~zero input concept share;
        suppressor-motif = fires on concept contexts yet projects against the concept token.</div>
    </div>
    """
    display(HTML(markup))


_LABEL_CANDIDATE_POSITIONS = (
    "top center",
    "bottom center",
    "middle right",
    "middle left",
    "top right",
    "top left",
    "bottom right",
    "bottom left",
)


# matplotlib analogue of each plotly textposition: (offset-points, ha, va)
_MPL_LABEL_ANCHORS = {
    "top center": ((0, 6), "center", "bottom"),
    "bottom center": ((0, -6), "center", "top"),
    "middle right": ((8, 0), "left", "center"),
    "middle left": ((-8, 0), "right", "center"),
    "top right": ((6, 6), "left", "bottom"),
    "top left": ((-6, 6), "right", "bottom"),
    "bottom right": ((6, -6), "left", "top"),
    "bottom left": ((-6, -6), "right", "top"),
}


def _collision_aware_label_positions(
    points: Any,
    labels: Sequence[str],
    marker_px: Sequence[float],
    extent: tuple[float, float, float, float] | None = None,
    plot_px: tuple[float, float] = (600.0, 400.0),
    char_px: float = 5.5,
    label_h_px: float = 12.0,
) -> list[str]:
    """Choose a plotly ``textposition`` per labeled point so labels stay individually legible.

    Approximates each candidate label's bounding box in plot pixels (accounting for text length and the point's
    marker diameter) and greedily picks the first anchor position whose box intersects neither previously placed
    labels nor other analyzed markers — so near-coincident markers fan their labels out instead of rendering them
    on top of each other. Falls back to round-robin if every candidate collides (dense worst case). Pass ``extent``
    as the FULL data extent backing the axes (e.g. including background points) — normalizing by the labeled
    points' own extent alone overestimates the pixel space between them.
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if not xs:
        return []
    x0, x1, y0, y1 = extent if extent is not None else (min(xs), max(xs), min(ys), max(ys))
    span_x = (x1 - x0) or 1.0
    span_y = (y1 - y0) or 1.0
    pts = [((x - x0) / span_x * plot_px[0], (y - y0) / span_y * plot_px[1]) for x, y in zip(xs, ys)]

    def label_box(i: int, pos: str) -> tuple[float, float, float, float]:
        w, h, r = char_px * len(labels[i]), label_h_px, marker_px[i] / 2.0
        dx = -(r + w / 2.0) if "left" in pos else (r + w / 2.0) if "right" in pos else 0.0
        dy = (r + h / 2.0) if "top" in pos else -(r + h / 2.0) if "bottom" in pos else 0.0
        cx, cy = pts[i][0] + dx, pts[i][1] + dy
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def overlaps(b1, b2) -> bool:
        return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])

    marker_boxes = [(x - m / 2.0, y - m / 2.0, x + m / 2.0, y + m / 2.0) for (x, y), m in zip(pts, marker_px)]
    placed: list[tuple[float, float, float, float]] = []
    chosen_positions: list[str] = []
    for i in range(len(pts)):
        obstacles = placed + [b for j, b in enumerate(marker_boxes) if j != i]
        chosen = next(
            (pos for pos in _LABEL_CANDIDATE_POSITIONS if not any(overlaps(label_box(i, pos), b) for b in obstacles)),
            _LABEL_CANDIDATE_POSITIONS[i % len(_LABEL_CANDIDATE_POSITIONS)],
        )
        placed.append(label_box(i, chosen))
        chosen_positions.append(chosen)
    return chosen_positions


def _plotly_figure_html(fig: Any, id_prefix: str) -> str:
    """Embed a plotly figure as HTML that sizes itself correctly on a static docs page.

    `responsive` alone is not enough on a static docs page. Plotly sizes an autosize figure to its container AT
    INIT, and in the rendered docs the output container has no final width at that moment, so the figure draws too
    narrow and its colorbar and legend are clipped until the reader resizes the window, which is the first event
    that triggers a relayout. Force that relayout once layout has settled, and keep a ResizeObserver for later
    changes (theme toggle, sidebar collapse) that do not fire a window resize. Div ids come from a per-run counter,
    so rendered artifacts stay diffable.
    """
    _decoder_map_html_counter[0] += 1
    div_id = f"{id_prefix}-{_decoder_map_html_counter[0]}"
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True}, div_id=div_id)
    resize_shim = f"""
<script>
(function () {{
  var el = document.getElementById("{div_id}");
  if (!el) return;
  var resize = function () {{
    if (!window.Plotly || !el.isConnected) return;
    try {{ window.Plotly.Plots.resize(el); }} catch (e) {{ /* figure not ready yet */ }}
  }};
  if (window.requestAnimationFrame) {{
    requestAnimationFrame(function () {{ requestAnimationFrame(resize); }});
  }}
  setTimeout(resize, 0);
  setTimeout(resize, 300);
  if (window.ResizeObserver && el.parentNode) {{
    new window.ResizeObserver(resize).observe(el.parentNode);
  }}
}})();
</script>
"""
    return '<div style="width:100%;max-width:100%;overflow-x:hidden">' + plot_html + resize_shim + "</div>"


def plot_decoder_projection_map(
    analyzed_profiles: Sequence[Any],
    analyzed_vectors: torch.Tensor,
    background_vectors: torch.Tensor,
    *,
    feature_explanations: Mapping[tuple[int, int], str] | None = None,
    target_label: str = "A−B",
    title: str = "Decoder-vector projection map",
    static_companion: bool = False,
    color_by: str = "output_projection",
) -> None:
    """Interactive decoder-space projection (UMAP w/ PCA fallback) with per-feature hover details.

    Uses plotly for hover tooltips (feature id, input concept share, activation mass, signed output projection,
    explanation) with the analyzed features colored by signed output projection over a gray active-feature background.
    Pass ``color_by="jlens_concept_mass"`` to color by J-space disposition instead: each feature's share of
    signature mass on the concept tokens (0 to 1, sequential scale), which answers where the feature disposes
    the model to speak rather than how hard it pushes one logit difference. Profiles without a signature fall
    back to output projection, so mixed batches still render.
    Axis tick labels are hidden deliberately: UMAP coordinates are non-metric (arbitrary rotation/scale; only local
    neighborhood structure is meaningful), so raw axis numbers invite over-reading. The interactive figure is embedded
    as an HTML div bootstrapping plotly.js from CDN (``fig.to_html(include_plotlyjs="cdn")``) rather than via
    ``fig.show()`` — renderer autodetection can silently pick the browser renderer under headless kernels (spawning a
    local server and writing nothing usable into the notebook), while the embedded div renders with full hover in any
    HTML-capable notebook viewer, matching the latent-dynamics analysis charts. SVG traces only (``go.Scatter``):
    WebGL ``Scattergl`` traces render blank in several viewers. Set ``static_companion=True`` to ALSO render a static
    matplotlib PNG (useful for offline viewers without CDN access); matplotlib likewise serves as the fallback when
    plotly is unavailable. The trace legend renders horizontally BELOW the plot so it cannot collide with the
    colorbar on the right, and per-point labels use collision-aware anchor placement
    (``_collision_aware_label_positions``) so near-coincident analyzed features remain individually legible.
    """
    from interpretune.utils.latent_state_projection import project_embeddings

    matrix = torch.cat(
        [
            torch.as_tensor(analyzed_vectors, dtype=torch.float32),
            torch.as_tensor(background_vectors, dtype=torch.float32),
        ],
        dim=0,
    )
    proj_result = project_embeddings(matrix, method="umap", n_components=2)
    coords = proj_result.coordinates
    n_analyzed = len(analyzed_profiles)
    fg, bg = coords[:n_analyzed], coords[n_analyzed:]
    method_label = f"{proj_result.method.upper()} ({proj_result.backend})"
    axes_extent = (
        float(coords[:, 0].min()),
        float(coords[:, 0].max()),
        float(coords[:, 1].min()),
        float(coords[:, 1].max()),
    )

    if color_by == "jlens_concept_mass":
        color_values = [
            float(p.jlens_concept_mass)
            if getattr(p, "jlens_concept_mass", None) is not None
            else float(p.output_projection)
            for p in analyzed_profiles
        ]
        color_scale, color_mid, color_title = "Viridis", None, "J-space concept mass"
    elif color_by == "output_projection":
        color_values = [float(p.output_projection) for p in analyzed_profiles]
        color_scale, color_mid, color_title = (
            "RdBu_r",
            0.0,
            f"output proj<br>({target_label})",
        )
    else:
        raise ValueError(
            f"plot_decoder_projection_map color_by {color_by!r} is not a coloring: "
            "expected 'output_projection' or 'jlens_concept_mass'."
        )

    try:
        import plotly.graph_objects as go

        hover_text = []
        for prof in analyzed_profiles:
            explanation = (feature_explanations or {}).get((prof.layer, prof.feature), "") or "—"
            hover_text.append(
                f"L{prof.layer}/{prof.feature}<br>input concept share: {prof.input_concept_share:.3f}"
                f"<br>act mass: {prof.activation_mass:.2f}"
                f"<br>output proj ({target_label}): {prof.output_projection:+.4f}"
                f"<br>explanation: {html.escape(str(explanation))[:80]}"
            )
        point_labels = [f"L{p.layer}/{p.feature}" for p in analyzed_profiles]
        marker_px = [10 + 22 * p.input_concept_share for p in analyzed_profiles]
        label_positions = _collision_aware_label_positions(fg, point_labels, marker_px, extent=axes_extent)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bg[:, 0],
                y=bg[:, 1],
                mode="markers",
                name="active-feature background",
                marker=dict(size=4, color="lightgray"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fg[:, 0],
                y=fg[:, 1],
                mode="markers+text",
                name="analyzed features",
                text=point_labels,
                textposition=label_positions,
                textfont=dict(size=9),
                hovertext=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=marker_px,
                    color=color_values,
                    colorscale=color_scale,
                    cmid=color_mid,
                    showscale=True,
                    colorbar=dict(title=dict(text=color_title, side="right"), len=0.85),
                    line=dict(width=1, color="black"),
                ),
            )
        )
        fig.update_layout(
            title=f"{title} — {method_label}; marker size = input concept share",
            # No fixed width: a hard 820px overflows the docs content column and forces a horizontal
            # scrollbar on the rendered page. autosize + responsive lets the figure track its container.
            autosize=True,
            height=560,
            template="simple_white",
            xaxis=dict(showticklabels=False, title=None),
            yaxis=dict(showticklabels=False, title=None),
            legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="left", x=0),
        )
        display(HTML(_plotly_figure_html(fig, "it-decoder-map")))
        plotly_rendered = True
    except Exception:
        plotly_rendered = False

    if static_companion or not plotly_rendered:
        import matplotlib.pyplot as plt

        static_fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(bg[:, 0], bg[:, 1], s=8, c="lightgray", label="active-feature background")
        sc = ax.scatter(
            fg[:, 0],
            fg[:, 1],
            s=[80 + 320 * p.input_concept_share for p in analyzed_profiles],
            c=color_values,
            cmap="viridis" if color_by == "jlens_concept_mass" else "coolwarm",
            edgecolors="black",
            linewidths=0.8,
            label="analyzed features",
        )
        static_labels = [f"L{p.layer}/{p.feature}" for p in analyzed_profiles]
        static_marker_px = [10 + 22 * p.input_concept_share for p in analyzed_profiles]
        for prof, (x, y), pos in zip(
            analyzed_profiles,
            fg,
            _collision_aware_label_positions(fg, static_labels, static_marker_px, extent=axes_extent),
        ):
            (dx, dy), ha, va = _MPL_LABEL_ANCHORS[pos]
            ax.annotate(
                f"L{prof.layer}/{prof.feature}",
                (x, y),
                fontsize=8,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
            )
        static_fig.colorbar(sc, ax=ax, label=f"decoder proj onto {target_label} (signed)")
        static_suffix = " (static companion)" if plotly_rendered else ""
        ax.set_title(f"{title} — {method_label}; marker size = input concept share{static_suffix}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()
