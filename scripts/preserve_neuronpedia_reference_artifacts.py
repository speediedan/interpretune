from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from interpretune.utils.neuronpedia_db_utils import (
    compare_neuronpedia_export_bundle_to_import_summary,
    import_neuronpedia_export_bundle_local_db,
    summarize_neuronpedia_export_bundle,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/reference_artifacts"
)


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonify(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve a real Neuronpedia golden batch plus bundle/import timing metadata so later "
            "serializer and bundle refactors can be checked against a fixed live artifact."
        )
    )
    parser.add_argument("--export-root", type=Path, required=True, help="Existing Neuronpedia export bundle root.")
    parser.add_argument(
        "--golden-batch-path",
        type=Path,
        required=True,
        help="Existing batch-0.json (or equivalent) to preserve alongside the bundle summary.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Parent directory for the preserved artifact set. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="Optional name for the artifact directory. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--local-db-url",
        default="",
        help=(
            "Local Neuronpedia DB URL. Leave empty to resolve it from the standard environment variables before "
            "capturing the import timing baseline."
        ),
    )
    parser.add_argument(
        "--skip-local-import",
        action="store_true",
        help="Skip the local DB import timing baseline and preserve only the bundle summary plus golden batch.",
    )
    parser.add_argument(
        "--no-copy-golden-batch",
        action="store_true",
        help="Record the source batch metadata only and do not copy the JSON payload into the artifact directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    export_root = args.export_root.resolve()
    golden_batch_path = args.golden_batch_path.resolve()
    if not export_root.exists():
        raise FileNotFoundError(f"Export root does not exist: {export_root}")
    if not golden_batch_path.exists():
        raise FileNotFoundError(f"Golden batch path does not exist: {golden_batch_path}")

    artifact_name = args.artifact_name or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    artifact_dir = args.output_root.resolve() / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=False)

    preserved_batch_path = None
    if not args.no_copy_golden_batch:
        preserved_batch_path = artifact_dir / golden_batch_path.name
        shutil.copy2(golden_batch_path, preserved_batch_path)

    bundle_summary = summarize_neuronpedia_export_bundle(export_root)
    import_summary = None
    parity_summary = None
    if not args.skip_local_import:
        import_summary = import_neuronpedia_export_bundle_local_db(export_root, local_db_url=args.local_db_url)
        parity_summary = compare_neuronpedia_export_bundle_to_import_summary(bundle_summary, import_summary)

    manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "export_root": str(export_root),
        "golden_batch": {
            "source_path": str(golden_batch_path),
            "preserved_path": str(preserved_batch_path) if preserved_batch_path is not None else None,
            "size_bytes": golden_batch_path.stat().st_size,
            "sha256": _sha256_file(golden_batch_path),
        },
        "bundle_summary": _jsonify(bundle_summary),
        "import_summary": _jsonify(import_summary),
        "bundle_import_parity": _jsonify(parity_summary),
    }
    manifest_path = artifact_dir / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Preserved Neuronpedia reference artifacts at {artifact_dir}")
    print(f"Manifest: {manifest_path}")
    if preserved_batch_path is not None:
        print(f"Golden batch copy: {preserved_batch_path}")
    if parity_summary is not None:
        print(f"Attempted bundle counts: {parity_summary.import_attempted_row_counts}")
        print(f"Inserted row counts: {parity_summary.imported_row_counts}")
        print(f"Skipped existing rows: {parity_summary.skipped_existing_row_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
