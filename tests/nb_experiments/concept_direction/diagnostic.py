import os
import sys

import torch

# Resolve repo root from this file's location:
# tests/nb_experiments/concept_direction/diagnostic.py -> repo root is two parents up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

IMPORT_ERROR: ImportError | None = None

try:
    import tests.nb_experiments.concept_direction.test_concept_direction_backend_parity as parity
except ImportError as exc:
    parity = None
    IMPORT_ERROR = exc


def _load_legacy_analysis_runtime():
    try:
        from interpretune.core.analysis import NNSightAnalysisSession, ModulePath
    except ImportError as exc:
        raise RuntimeError(
            "diagnostic.py still depends on the legacy interpretune.core.analysis runtime. "
            "Run this helper only after updating it to a current analysis-session API."
        ) from exc
    return NNSightAnalysisSession, ModulePath


def run_diagnostic():
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Diagnostic imports are unavailable: {IMPORT_ERROR}") from IMPORT_ERROR
    assert parity is not None
    NNSightAnalysisSession, ModulePath = _load_legacy_analysis_runtime()

    print("Loading case...")
    case = parity._load_gemma3_1b_it_concept_direction_parity_case(
        concept_pair_config_name="cp_ohio_entities_gemma_it.yaml"
    )

    model_name = "google/gemma-3-1b-it"
    transcoder_set = "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine"
    module = ModulePath("blocks.14")

    print("Creating session...")
    session = NNSightAnalysisSession(
        model_name=model_name, transcoder_set=transcoder_set, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    with session:
        print("Computing store_context_stage...")
        store_context_stage = parity._compute_store_concept_direction_stage(module, case, context_enhanced=True)

        print("Building store_context_artifacts...")
        store_context_artifacts = parity._build_concept_direction_graph_artifacts(module, case, store_context_stage)

        print("Building reference payload...")
        reference_payload = parity._build_reference_graph_target_artifact(
            module, case, target_label="store_context", concept_direction=store_context_stage.concept_direction
        )

    print("Printing results:")
    direct_artifact = store_context_artifacts.graph_stage_artifact

    print(f"direct top_feature_ids: {direct_artifact.top_feature_ids}")
    print(f"direct top_feature_scores: {direct_artifact.top_feature_scores}")
    print(f"reference top_feature_ids: {reference_payload.top_feature_ids}")
    print(f"reference top_feature_scores: {reference_payload.top_feature_scores}")

    diffs = []
    d_kwargs = getattr(direct_artifact, "graph_call_kwargs", None)
    r_kwargs = getattr(reference_payload, "graph_call_kwargs", None)
    if d_kwargs != r_kwargs:
        diffs.append("graph_call_kwargs differ")

    d_targets = getattr(direct_artifact, "attribution_targets", None)
    r_targets = getattr(reference_payload, "attribution_targets", None)
    if d_targets != r_targets:
        diffs.append("attribution_targets differ")

    if not diffs:
        print("Metadata: No differences found in graph_call_kwargs or attribution_targets.")
    else:
        print(f"Metadata: {', '.join(diffs)}")


if __name__ == "__main__":
    try:
        run_diagnostic()
    except Exception as e:
        print(f"Diagnostic failed with error: {e}")
        import traceback

        traceback.print_exc()
