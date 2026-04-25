#!/usr/bin/env python3
"""Extract semantic-intervention reference values from upstream circuit-tracer and Interpretune.

This is a manual debugging and sanity-check tool. It is intentionally not part of the normal pytest or coverage flows
because the regular Interpretune parity test should catch regressions first.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from circuit_tracer.attribution.attribute_nnsight import attribute as attribute_nnsight
from circuit_tracer.attribution.attribute_transformerlens import attribute as attribute_transformerlens
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.graph import compute_node_influence
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.demo_utils import get_unembed_vecs


DEFAULT_PROMPT = "Fact: the capital of the state containing Dallas is"
DEFAULT_CAPITALS = ["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"]
DEFAULT_STATES = ["▁Texas", "▁California", "▁Washington", "▁Georgia"]
DEFAULT_LABEL = "Concept: Capitals − States"
DEFAULT_TOP_N = 10


def _coerce_fixture_fn(fn):
    return getattr(fn, "__wrapped__", fn)


def _build_demo_semantic_target(
    model: ReplacementModel,
    *,
    prompt: str,
    group_a_tokens: list[str],
    group_b_tokens: list[str],
    label: str,
    backend: str,
) -> tuple[CustomTarget, list[int]]:
    assert len(group_a_tokens) == len(group_b_tokens), "Groups must have equal length for paired rejection"

    tokenizer = model.tokenizer
    ids_a = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in group_a_tokens]
    ids_b = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in group_b_tokens]
    vecs_a = get_unembed_vecs(model, ids_a, backend)
    vecs_b = get_unembed_vecs(model, ids_b, backend)

    residuals = []
    for vec_a, vec_b in zip(vecs_a, vecs_b):
        vec_a_f = vec_a.float()
        vec_b_f = vec_b.float()
        projection = (vec_a_f @ vec_b_f) / (vec_b_f @ vec_b_f) * vec_b_f
        residuals.append((vec_a_f - projection).to(vec_a.dtype))

    direction = torch.stack(residuals).mean(0)
    input_ids = model.ensure_tokenized(prompt)
    logits, _ = model.get_activations(input_ids)
    probs = torch.softmax(logits.squeeze(0)[-1], dim=-1)
    avg_prob = max(sum(probs[index].item() for index in ids_a) / len(ids_a), 1e-6)
    return CustomTarget(token_str=label, prob=avg_prob, vec=direction), ids_a


def _move_replacement_model(model: ReplacementModel, device: str | torch.device) -> None:
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    try:
        model.transcoders.to(device, torch.float32)
    except TypeError:
        model.transcoders.to(device)

    for attr in ("embed_weight", "unembed_weight"):
        tensor = getattr(model, attr, None)
        if tensor is not None and tensor.device != device:
            setattr(model, attr, tensor.to(device))

    if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
        model.cfg.device = device


@contextmanager
def _clean_cuda(model: ReplacementModel, min_bytes: int = 1 << 20):
    _move_replacement_model(model, "cuda")

    def _is_large_dense_cuda(candidate: object) -> bool:
        return (
            isinstance(candidate, torch.Tensor)
            and candidate.is_cuda
            and candidate.layout == torch.strided
            and candidate.nbytes >= min_bytes
        )

    known_ptrs = {obj.data_ptr() for obj in gc.get_objects() if _is_large_dense_cuda(obj)}
    try:
        yield
    finally:
        freed_ptrs: set[int] = set()
        for obj in gc.get_objects():
            if _is_large_dense_cuda(obj) and obj.data_ptr() not in known_ptrs and obj.data_ptr() not in freed_ptrs:
                freed_ptrs.add(obj.data_ptr())
                try:
                    obj.set_(torch.empty(0))
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()
        _move_replacement_model(model, "cpu")


def _top_features_from_graph(graph: Any, top_n: int) -> list[tuple[int, int, int]]:
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)
    logit_weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    _, top_idx = torch.topk(node_influence[:n_features], min(top_n, n_features))
    return [tuple(graph.active_features[graph.selected_features[index]].tolist()) for index in top_idx]


def _gap_from_logits(logits: torch.Tensor, preferred_token_id: int, baseline_token_id: int) -> float:
    final_logits = logits.squeeze(0)[-1]
    return float((final_logits[preferred_token_id] - final_logits[baseline_token_id]).cpu().item())


def _build_upstream_baseline(
    *,
    backend: str,
    model: ReplacementModel,
    prompt: str,
    capitals: list[str],
    states: list[str],
    label: str,
    top_n: int,
) -> dict[str, Any]:
    semantic_target, group_a_token_ids = _build_demo_semantic_target(
        model,
        prompt=prompt,
        group_a_tokens=capitals,
        group_b_tokens=states,
        label=label,
        backend=backend,
    )
    preferred_token_id = model.tokenizer.encode("▁Austin", add_special_tokens=False)[-1]
    baseline_token_id = model.tokenizer.encode("▁Dallas", add_special_tokens=False)[-1]

    attribute_fn = attribute_nnsight if backend == "nnsight" else attribute_transformerlens
    batch_size = 256 if backend == "nnsight" else 128
    graph = attribute_fn(
        prompt,
        model,
        attribution_targets=[semantic_target],
        verbose=False,
        batch_size=batch_size,
    )
    top_features = _top_features_from_graph(graph, top_n)
    input_ids = model.ensure_tokenized(prompt)
    orig_logits, activations = model.get_activations(input_ids, sparse=True)
    interventions = [
        (layer, position, feature, 10.0 * float(activations[layer, position, feature].item()))
        for layer, position, feature in top_features
    ]
    new_logits, _ = model.feature_intervention(input_ids, interventions)

    orig_gap = _gap_from_logits(orig_logits, preferred_token_id, baseline_token_id)
    new_gap = _gap_from_logits(new_logits, preferred_token_id, baseline_token_id)

    return {
        "backend": backend,
        "group_a_token_ids": group_a_token_ids,
        "top_feature_ids": top_features,
        "activation_values": [
            float(activations[layer, position, feature]) for layer, position, feature in top_features
        ],
        "intervention_values": [value for _, _, _, value in interventions],
        "pre_gap": orig_gap,
        "post_gap": new_gap,
        "gap_delta": new_gap - orig_gap,
        "concept_direction_norm": float(semantic_target.vec.float().norm().cpu().item()),
    }


def _load_interpretune_baselines(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(repo_root))
    from tests.core import test_analysis_backend_parity as parity_test

    case = _coerce_fixture_fn(parity_test.semantic_intervention_parity_case)()
    factory_builder = _coerce_fixture_fn(parity_test.ct_nnsight_session_factory)
    with tempfile.TemporaryDirectory(prefix="ct_semantic_reference_") as tmp_dir:
        session_factory = factory_builder(Path(tmp_dir))
        with session_factory("manual_native_reference") as native_session:
            native_baseline = parity_test._build_native_baseline(native_session, case)
        with session_factory("manual_op_reference") as op_session:
            op_baseline = parity_test._build_op_baseline(op_session, case)

    native_payload = {
        "group_a_token_ids": native_baseline.group_a_ids,
        "top_feature_ids": native_baseline.top_features,
        "activation_values": [float(value) for value in native_baseline.activation_values.tolist()],
        "intervention_values": [float(value) for _, _, _, value in native_baseline.interventions],
        "pre_gap": native_baseline.pre_gap,
        "post_gap": native_baseline.post_gap,
        "gap_delta": native_baseline.post_gap - native_baseline.pre_gap,
        "concept_direction_norm": float(native_baseline.concept_direction.norm().item()),
    }
    op_payload = {
        "direction_mode": op_baseline.direction_mode,
        "group_a_token_ids": op_baseline.group_a_token_ids,
        "top_feature_ids": op_baseline.top_feature_ids,
        "activation_values": [float(value) for value in op_baseline.activation_values.tolist()],
        "intervention_values": [float(value) for value in op_baseline.intervention_values.tolist()],
        "pre_gap": op_baseline.pre_gap,
        "post_gap": op_baseline.post_gap,
        "gap_delta": op_baseline.post_gap - op_baseline.pre_gap,
        "concept_direction_norm": float(op_baseline.concept_direction.norm().item()),
    }
    relationships = {
        "concept_direction_cosine": float(
            torch.nn.functional.cosine_similarity(
                op_baseline.concept_direction.unsqueeze(0),
                native_baseline.concept_direction.unsqueeze(0),
            ).item()
        ),
        "activation_max_abs_diff": float(
            torch.max(torch.abs(op_baseline.activation_values - native_baseline.activation_values)).item()
        ),
        "intervention_value_max_abs_diff": float(
            torch.max(
                torch.abs(
                    op_baseline.intervention_values
                    - torch.tensor(
                        [value for _, _, _, value in native_baseline.interventions],
                        dtype=torch.float32,
                    )
                )
            ).item()
        ),
        "pre_gap_abs_diff": abs(op_baseline.pre_gap - native_baseline.pre_gap),
        "post_gap_abs_diff": abs(op_baseline.post_gap - native_baseline.post_gap),
    }
    return native_payload, op_payload, relationships


def _build_markdown_table(payload: dict[str, Any]) -> str:
    upstream_nn = payload["upstream"]["nnsight"]
    interpretune_native = payload["interpretune"]["native"]
    interpretune_op = payload["interpretune"]["analysis_op"]
    tolerances = payload["tolerances"]
    rows = [
        (
            "pre_gap",
            upstream_nn["pre_gap"],
            interpretune_native["pre_gap"],
            interpretune_op["pre_gap"],
            "native≈op @ 1e-6",
        ),
        (
            "post_gap",
            upstream_nn["post_gap"],
            interpretune_native["post_gap"],
            interpretune_op["post_gap"],
            "all must exceed pre_gap",
        ),
        (
            "gap_delta",
            upstream_nn["gap_delta"],
            interpretune_native["gap_delta"],
            interpretune_op["gap_delta"],
            "all must stay positive",
        ),
        (
            "top_feature_ids",
            str(upstream_nn["top_feature_ids"][:3]),
            str(interpretune_native["top_feature_ids"][:3]),
            str(interpretune_op["top_feature_ids"][:3]),
            "native==op exact; upstream sanity-check",
        ),
        (
            "concept_direction_cosine(native/op)",
            "n/a",
            "1.000000",
            f"{payload['interpretune']['relationships']['concept_direction_cosine']:.6f}",
            f"> {tolerances['concept_direction_cosine_min']}",
        ),
    ]
    header = (
        "| Metric | Upstream CT NNsight | Interpretune Native CT | Interpretune Analysis Op | "
        "Expected Relation / Tolerance |"
    )
    separator = "|---|---:|---:|---:|---|"
    body = [
        f"| {metric} | {upstream} | {native} | {op} | {tolerance} |" for metric, upstream, native, op, tolerance in rows
    ]
    return "\n".join([header, separator, *body])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretune-repo", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--output-json", help="Optional JSON output path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the upstream semantic reference extractor.")

    models = {
        "nnsight": ReplacementModel.from_pretrained(
            "google/gemma-2-2b",
            "gemma",
            backend="nnsight",
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
        "transformerlens": ReplacementModel.from_pretrained(
            "google/gemma-2-2b",
            "gemma",
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
    }

    upstream_payload: dict[str, Any] = {}
    try:
        for backend, model in models.items():
            with _clean_cuda(model):
                upstream_payload[backend] = _build_upstream_baseline(
                    backend=backend,
                    model=model,
                    prompt=DEFAULT_PROMPT,
                    capitals=list(DEFAULT_CAPITALS),
                    states=list(DEFAULT_STATES),
                    label=DEFAULT_LABEL,
                    top_n=DEFAULT_TOP_N,
                )
    finally:
        del models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    native_payload, op_payload, relationships = _load_interpretune_baselines(args.interpretune_repo)
    payload = {
        "case": {
            "prompt": DEFAULT_PROMPT,
            "capitals": DEFAULT_CAPITALS,
            "states": DEFAULT_STATES,
            "label": DEFAULT_LABEL,
            "top_n": DEFAULT_TOP_N,
        },
        "tolerances": {
            "concept_direction_cosine_min": 0.999,
            "value_rtol": 1e-6,
            "value_atol": 1e-6,
            "upstream_post_gap_abs_diff_max": 0.5,
        },
        "upstream": {
            **upstream_payload,
            "post_gap_abs_diff": abs(
                upstream_payload["nnsight"]["post_gap"] - upstream_payload["transformerlens"]["post_gap"]
            ),
        },
        "interpretune": {
            "native": native_payload,
            "analysis_op": op_payload,
            "relationships": relationships,
        },
        "markdown_reference_table": _build_markdown_table(
            {
                "upstream": upstream_payload,
                "interpretune": {
                    "native": native_payload,
                    "analysis_op": op_payload,
                    "relationships": relationships,
                },
                "tolerances": {
                    "concept_direction_cosine_min": 0.999,
                },
            }
        ),
    }

    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.output_json:
        Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
