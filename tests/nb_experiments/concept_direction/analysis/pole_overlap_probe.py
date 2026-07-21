"""Quantify per-position-class push cancellation: polysemous orange vs token-disjoint capitals.

For each prompt's attribution graph (gemma-2-2b / 16k), classify pole-word / probe-token / answer
positions, then measure (a) how many active features fire on one pole vs BOTH poles, and (b) how
much of the gross signed push (activation x decoder projection onto the target unembed diff)
cancels in the net, per position class. Supports Finding 4 in concept_direction_analysis.md; kept
as a manual (GPU, non-test-suite) probe for the J-space follow-up (interpretune#225).
Run manually: HF auth resolves from the repo-root .env.

All work happens in ``main()`` so importing this module is side-effect-free — the test suite runs
pytest with ``--doctest-modules``, which imports every module under ``tests/`` during collection.
"""

import json
from pathlib import Path

SCENARIOS = {
    "orange_polysemous": dict(
        group_a=["apple", "banana", "grape", "peach"],
        group_b=["red", "blue", "green", "yellow"],
        label="Concept: Fruit - Color",
        prompt="Is orange a color or a fruit? Answer with one word: Color or Fruit. orange ->",
        pole_a_words=("fruit", "fruits"),
        pole_b_words=("color", "colors"),
        probe_words=("orange",),
        target_tokens=("Fruit", "Color"),
    ),
    "capitals_disjoint": dict(
        group_a=["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"],
        group_b=["▁Texas", "▁California", "▁Washington", "▁Georgia"],
        label="Concept: Capitals - States",
        prompt="Fact: the capital of the state containing Dallas is",
        pole_a_words=("capital", "capitals"),
        pole_b_words=("state", "states"),
        probe_words=("dallas",),
        target_tokens=("▁Austin", "▁Dallas"),
    ),
}


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    import torch

    import interpretune as it
    from it_examples import _ACTIVE_PATCHES  # noqa: F401
    from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY
    from it_examples.utils.example_helpers import concept_token_positions
    from interpretune import ITSession, ITSessionConfig
    from interpretune.analysis.backends import require_analysis_backend
    from interpretune.analysis.ops.base import AnalysisBatch
    from interpretune.config import AnalysisCfg, init_analysis_cfgs

    base_itdm_cfg, base_it_cfg, dm_cls, m_cls = MODULE_EXAMPLE_REGISTRY.get("gemma2.rte_demo.circuit_tracer")
    base_it_cfg.circuit_tracer_cfg.backend = "nnsight"
    base_it_cfg.circuit_tracer_cfg.transcoder_set = "gemma"
    session_cfg = ITSessionConfig(
        adapter_ctx=(it.Adapter.core, it.Adapter.nnsight, it.Adapter.circuit_tracer),
        datamodule_cfg=base_itdm_cfg,
        module_cfg=base_it_cfg,
        datamodule_cls=dm_cls,
        module_cls=m_cls,
    )
    it_session = ITSession(session_cfg)
    it.it_init(**it_session)
    module = it_session.module
    tokenizer = module.replacement_model.tokenizer
    backend = require_analysis_backend(module)
    module.analysis_cfg = AnalysisCfg(target_op=it.compute_attribution_graph, ignore_manual=True, save_tokens=False)
    init_analysis_cfgs(module, [module.analysis_cfg])

    embed_weight = backend.get_embedding_weight(module).detach().float().cpu()
    transcoders = getattr(module.replacement_model.transcoders, "_module", module.replacement_model.transcoders)
    results = {}
    for name, sc in SCENARIOS.items():
        res = it.intervention_from_concept(
            module,
            AnalysisBatch(
                concept_group_a=sc["group_a"],
                concept_group_b=sc["group_b"],
                concept_label=sc["label"],
                concept_direction_mode="paired_rejection",
                prompts=[sc["prompt"]],
            ),
            None,
            0,
            top_n=5,
            intervention_scale_factor=10.0,
        )
        graph = backend.hydrate_graph_from_batch(res)
        token_ids = [int(t) for t in graph.input_tokens]
        pos_a = concept_token_positions(tokenizer, token_ids, sc["pole_a_words"])
        pos_b = concept_token_positions(tokenizer, token_ids, sc["pole_b_words"])
        pos_probe = concept_token_positions(tokenizer, token_ids, sc["probe_words"])
        pos_answer = {len(token_ids) - 1}
        tid_a = tokenizer.encode(sc["target_tokens"][0], add_special_tokens=False)[-1]
        tid_b = tokenizer.encode(sc["target_tokens"][1], add_special_tokens=False)[-1]
        tdiff = embed_weight[tid_a] - embed_weight[tid_b]
        tdiff = tdiff / tdiff.norm()

        rows = graph.active_features.cpu()
        vals = graph.activation_values.detach().float().cpu().abs()
        # per (layer, feature): activation mass at pole A / pole B positions
        pole_mass = {}
        for (layer, pos, feat), v in zip(rows.tolist(), vals.tolist()):
            key = (int(layer), int(feat))
            a_m, b_m = pole_mass.get(key, (0.0, 0.0))
            if int(pos) in pos_a:
                a_m += v
            if int(pos) in pos_b:
                b_m += v
            pole_mass[key] = (a_m, b_m)
        pole_active = {k: (a, b) for k, (a, b) in pole_mass.items() if a > 0 or b > 0}
        bilateral = {k: (a, b) for k, (a, b) in pole_active.items() if min(a, b) / max(a, b) >= 0.2}
        only_a = sum(1 for a, b in pole_active.values() if b == 0)
        only_b = sum(1 for a, b in pole_active.values() if a == 0)

        # signed push and cancellation over pole-active features (batch decoder reads per layer)
        keys = list(pole_active.keys())
        proj = {}
        by_layer = {}
        for layer, feat in keys:
            by_layer.setdefault(layer, []).append(feat)
        for layer, feats in by_layer.items():
            dec = transcoders._get_decoder_vectors(layer, torch.tensor(feats)).detach().float().cpu()
            for f_id, d_vec in zip(feats, dec):
                proj[(layer, f_id)] = float(d_vec @ tdiff)
        pushes = [proj[k] * (a + b) for k, (a, b) in pole_active.items()]
        gross = sum(abs(x) for x in pushes)
        net = abs(sum(pushes))

        # per-position-class cancellation for probe and answer positions
        def _position_class_stats(target_positions):
            class_mass = {}
            for (layer, pos, feat), v in zip(rows.tolist(), vals.tolist()):
                if int(pos) in target_positions:
                    key = (int(layer), int(feat))
                    class_mass[key] = class_mass.get(key, 0.0) + v
            cls_keys = list(class_mass.keys())
            cls_by_layer = {}
            for layer, feat in cls_keys:
                cls_by_layer.setdefault(layer, []).append(feat)
            cls_proj = {}
            for layer, feats in cls_by_layer.items():
                dec = transcoders._get_decoder_vectors(layer, torch.tensor(feats)).detach().float().cpu()
                for f_id, d_vec in zip(feats, dec):
                    cls_proj[(layer, f_id)] = float(d_vec @ tdiff)
            cls_pushes = {k: cls_proj[k] * m for k, m in class_mass.items()}
            cls_gross = sum(abs(x) for x in cls_pushes.values())
            cls_net = abs(sum(cls_pushes.values()))
            top = sorted(cls_pushes.items(), key=lambda kv: -abs(kv[1]))[:5]
            return {
                "n_features": len(class_mass),
                "gross": round(cls_gross, 2),
                "net": round(cls_net, 2),
                "cancellation": round(1.0 - (cls_net / cls_gross if cls_gross else 0.0), 4),
                "top_push_features": [
                    {"layer": k[0], "feature": k[1], "push": round(v, 3), "proj": round(cls_proj[k], 4)} for k, v in top
                ],
            }

        probe_stats = _position_class_stats(pos_probe)
        answer_stats = _position_class_stats(pos_answer)
        top_bilateral = sorted(bilateral.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:6]
        results[name] = {
            "pole_a_positions": sorted(pos_a),
            "pole_b_positions": sorted(pos_b),
            "n_pole_active_features": len(pole_active),
            "n_only_pole_a": only_a,
            "n_only_pole_b": only_b,
            "n_bilateral(min/max>=0.2)": len(bilateral),
            "bilateral_fraction": len(bilateral) / len(pole_active) if pole_active else 0.0,
            "gross_signed_push": gross,
            "net_signed_push": net,
            "cancellation_ratio(1-net/gross)": 1.0 - (net / gross if gross else 0.0),
            "probe_positions": sorted(pos_probe),
            "probe_position_stats": probe_stats,
            "answer_position_stats": answer_stats,
            "top_bilateral_features": [
                {
                    "layer": k[0],
                    "feature": k[1],
                    "mass_a": round(a, 2),
                    "mass_b": round(b, 2),
                    "proj": round(proj[k], 4),
                }
                for k, (a, b) in top_bilateral
            ],
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
