from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

utils = importlib.import_module("interpretune.utils")
DEFAULT_NEURONPEDIA_BASE_URL = utils.DEFAULT_NEURONPEDIA_BASE_URL
DEFAULT_COPILOT_TIMEOUT_SECONDS = utils.DEFAULT_COPILOT_TIMEOUT_SECONDS
DEFAULT_COPILOT_MODEL = utils.DEFAULT_COPILOT_MODEL
DEFAULT_GENERATED_OUTPUT_DIR = utils.DEFAULT_GENERATED_OUTPUT_DIR
DEFAULT_EXPLANATION_AUTHOR_ID = utils.DEFAULT_EXPLANATION_AUTHOR_ID
DEFAULT_EXPLANATION_TYPE_NAME = utils.DEFAULT_EXPLANATION_TYPE_NAME
build_feature_ref = utils.build_feature_ref
generate_explanation_artifact = utils.generate_explanation_artifact


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Neuronpedia explanation generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a Neuronpedia-style feature explanation artifact via GitHub Copilot CLI from "
            "either a full feature URL or explicit feature metadata."
        )
    )
    parser.add_argument("--feature-url", help="Full Neuronpedia feature URL or API URL.")
    parser.add_argument("--model-id", help="Neuronpedia model ID.")
    parser.add_argument(
        "--layer",
        help=(
            "Neuronpedia layer identifier. This can be either the full value such as "
            "23-gemmascope-2-transcoder-262k or just the numeric layer when --source-set is also provided."
        ),
    )
    parser.add_argument("--source-set", help="Optional source-set suffix when --layer is only a numeric layer.")
    parser.add_argument("--index", help="Neuronpedia feature index.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_NEURONPEDIA_BASE_URL,
        help=f"Neuronpedia base URL. Defaults to {DEFAULT_NEURONPEDIA_BASE_URL}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_GENERATED_OUTPUT_DIR),
        help="Directory where the markdown artifact will be written.",
    )
    parser.add_argument(
        "--copilot-model",
        default=DEFAULT_COPILOT_MODEL,
        help="COPILOT_MODEL passed to the GitHub Copilot CLI.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_COPILOT_TIMEOUT_SECONDS,
        help="Timeout for the Copilot CLI call.",
    )
    parser.add_argument(
        "--write-neuronpedia-import-data",
        action="store_true",
        help="Write a Neuronpedia-formatted explanation import bundle under the output directory.",
    )
    parser.add_argument(
        "--insert-into-local-db",
        action="store_true",
        help="Insert the generated explanation into a local Neuronpedia Postgres database.",
    )
    parser.add_argument("--local-db-url", help="Postgres connection string for a local Neuronpedia database.")
    parser.add_argument(
        "--explanation-author-id",
        default=DEFAULT_EXPLANATION_AUTHOR_ID,
        help="Author ID to use for generated explanation rows.",
    )
    parser.add_argument(
        "--triggered-by-user-id",
        help="Optional triggeredByUserId for Neuronpedia explanation import rows.",
    )
    parser.add_argument(
        "--type-name",
        default=DEFAULT_EXPLANATION_TYPE_NAME,
        help="Explanation type name to use for Neuronpedia import rows.",
    )
    parser.add_argument(
        "--explanation-model-name",
        help="Optional Neuronpedia explanation model type override for import rows.",
    )
    return parser


def main() -> int:
    """Run the explanation generation CLI."""

    args = build_parser().parse_args()
    feature_ref = build_feature_ref(
        feature_url=args.feature_url,
        model_id=args.model_id,
        layer=args.layer,
        source_set=args.source_set,
        index=args.index,
        base_url=args.base_url,
    )
    result = generate_explanation_artifact(
        feature_ref=feature_ref,
        output_dir=Path(args.output_dir),
        copilot_model=args.copilot_model,
        timeout_seconds=args.timeout_seconds,
        write_neuronpedia_import_data=args.write_neuronpedia_import_data,
        insert_into_local_db=args.insert_into_local_db,
        local_db_url=args.local_db_url,
        explanation_author_id=args.explanation_author_id,
        triggered_by_user_id=args.triggered_by_user_id,
        type_name=args.type_name,
        explanation_model_name=args.explanation_model_name,
    )
    print(result.artifact_path)
    print(result.cleaned_explanation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
