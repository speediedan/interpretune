from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urlparse

import psycopg


@dataclass(frozen=True)
class LocalGraphArtifact:
    slug: str
    path: Path
    model_id: str | None


@dataclass(frozen=True)
class LocalGraphCleanupSummary:
    work_root: str
    graph_root: str
    local_db_url: str | None
    local_webapp_url: str | None
    discovered_slugs: tuple[str, ...]
    deleted_slugs: tuple[str, ...]
    deleted_remote_slugs: tuple[str, ...]
    skipped_remote_slugs: tuple[str, ...]
    remote_delete_errors: dict[str, str]
    deleted_files: tuple[str, ...]
    dry_run: bool


DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL = "http://localhost:3000"


def _is_s3_backed_graph_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and "amazonaws.com" in parsed.netloc


def _delete_remote_graph_payload(
    *,
    model_id: str,
    slug: str,
    local_webapp_url: str,
    local_api_key: str,
) -> None:
    payload = json.dumps({"modelId": model_id, "slug": slug}).encode("utf-8")
    req = request.Request(
        f"{local_webapp_url.rstrip('/')}/api/graph/delete",
        data=payload,
        headers={
            "X-Api-Key": local_api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise RuntimeError("graph metadata not found in local webapp") from exc
        raise RuntimeError(f"delete request failed with status {exc.code}: {body[:400]}") from exc

    if status_code >= 400:
        raise RuntimeError(f"delete request failed with status {status_code}: {body[:400]}")


def _extract_graph_artifact(graph_path: Path) -> LocalGraphArtifact | None:
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return None

    slug = metadata.get("slug")
    if not isinstance(slug, str) or not slug:
        return None

    model_id = metadata.get("scan")
    if not isinstance(model_id, str) or not model_id:
        model_id = None
        info = metadata.get("info")
        if isinstance(info, dict):
            candidate_model_id = info.get("neuronpedia_model")
            if isinstance(candidate_model_id, str) and candidate_model_id:
                model_id = candidate_model_id

    return LocalGraphArtifact(
        slug=slug,
        path=graph_path,
        model_id=model_id,
    )


def discover_local_graph_artifacts(
    work_root: str | Path, *, slug_prefix: str | None = None
) -> list[LocalGraphArtifact]:
    graph_root = Path(work_root).expanduser().resolve() / "graph_artifacts"
    if not graph_root.exists():
        return []

    artifacts: list[LocalGraphArtifact] = []
    for graph_path in sorted(graph_root.rglob("*.json")):
        artifact = _extract_graph_artifact(graph_path)
        if artifact is None:
            continue
        if slug_prefix and not artifact.slug.startswith(slug_prefix):
            continue
        artifacts.append(artifact)
    return artifacts


def cleanup_local_graph_artifacts(
    work_root: str | Path,
    *,
    local_db_url: str | None = None,
    local_webapp_url: str | None = None,
    local_api_key: str | None = None,
    slug_prefix: str | None = None,
    dry_run: bool = False,
    remove_files: bool = True,
) -> LocalGraphCleanupSummary:
    resolved_work_root = Path(work_root).expanduser().resolve()
    graph_root = resolved_work_root / "graph_artifacts"
    artifacts = discover_local_graph_artifacts(resolved_work_root, slug_prefix=slug_prefix)
    discovered_slugs = tuple(artifact.slug for artifact in artifacts)

    if dry_run or not artifacts:
        return LocalGraphCleanupSummary(
            work_root=str(resolved_work_root),
            graph_root=str(graph_root),
            local_db_url=local_db_url,
            local_webapp_url=local_webapp_url,
            discovered_slugs=discovered_slugs,
            deleted_slugs=(),
            deleted_remote_slugs=(),
            skipped_remote_slugs=(),
            remote_delete_errors={},
            deleted_files=(),
            dry_run=dry_run,
        )

    resolved_db_url = local_db_url or os.environ.get("LOCAL_NEURONPEDIA_DB_URL")
    if not resolved_db_url:
        raise ValueError("local_db_url is required unless LOCAL_NEURONPEDIA_DB_URL is set.")

    resolved_webapp_url = local_webapp_url or os.environ.get("LOCAL_NEURONPEDIA_WEBAPP_URL")
    resolved_api_key = (
        local_api_key or os.environ.get("DEV_NEURONPEDIA_API_KEY") or os.environ.get("NEURONPEDIA_API_KEY")
    )

    deleted_remote_slugs: list[str] = []
    skipped_remote_slugs: list[str] = []
    remote_delete_errors: dict[str, str] = {}

    graph_rows_by_slug: dict[str, tuple[str | None, str | None]] = {}
    with psycopg.connect(resolved_db_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                'SELECT slug, "modelId", url FROM public."GraphMetadata" WHERE slug = ANY(%s);',
                ([artifact.slug for artifact in artifacts],),
            )
            graph_rows_by_slug = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

    remote_delete_enabled = bool(resolved_webapp_url and resolved_api_key)
    if remote_delete_enabled:
        for artifact in artifacts:
            model_id = artifact.model_id
            stored_model_id, stored_url = graph_rows_by_slug.get(artifact.slug, (None, None))
            if model_id is None:
                model_id = stored_model_id

            if model_id is None:
                skipped_remote_slugs.append(artifact.slug)
                remote_delete_errors[artifact.slug] = "missing model_id for remote delete"
                continue
            if not isinstance(stored_url, str) or not stored_url or not _is_s3_backed_graph_url(stored_url):
                skipped_remote_slugs.append(artifact.slug)
                remote_delete_errors[artifact.slug] = "graph metadata url is missing or not S3-backed"
                continue

            try:
                assert resolved_webapp_url is not None
                assert resolved_api_key is not None
                _delete_remote_graph_payload(
                    model_id=model_id,
                    slug=artifact.slug,
                    local_webapp_url=resolved_webapp_url,
                    local_api_key=resolved_api_key,
                )
            except Exception as exc:
                skipped_remote_slugs.append(artifact.slug)
                remote_delete_errors[artifact.slug] = str(exc)
            else:
                deleted_remote_slugs.append(artifact.slug)

    db_deleted_slugs: tuple[str, ...] = ()
    filenames = [artifact.path.name for artifact in artifacts]
    with psycopg.connect(resolved_db_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                'DELETE FROM public."GraphMetadataDataPutRequest" WHERE filename = ANY(%s);',
                (filenames,),
            )
            cursor.execute(
                'DELETE FROM public."GraphMetadata" WHERE slug = ANY(%s) RETURNING slug;',
                ([artifact.slug for artifact in artifacts],),
            )
            db_deleted_slugs = tuple(row[0] for row in cursor.fetchall())
        conn.commit()

    deleted_slugs = tuple(dict.fromkeys([*deleted_remote_slugs, *db_deleted_slugs]).keys())

    deleted_files: list[str] = []
    if remove_files:
        for artifact in artifacts:
            if artifact.path.exists():
                artifact.path.unlink()
                deleted_files.append(str(artifact.path))

        for directory in sorted(graph_root.rglob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    continue

    return LocalGraphCleanupSummary(
        work_root=str(resolved_work_root),
        graph_root=str(graph_root),
        local_db_url=resolved_db_url,
        local_webapp_url=resolved_webapp_url,
        discovered_slugs=discovered_slugs,
        deleted_slugs=deleted_slugs,
        deleted_remote_slugs=tuple(deleted_remote_slugs),
        skipped_remote_slugs=tuple(skipped_remote_slugs),
        remote_delete_errors=remote_delete_errors,
        deleted_files=tuple(deleted_files),
        dry_run=False,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean up local Neuronpedia graph artifacts generated by notebook runs."
    )
    parser.add_argument(
        "--work-root",
        required=True,
        help="Notebook work root containing the graph_artifacts directory.",
    )
    parser.add_argument(
        "--local-db-url",
        default=None,
        help="Local Neuronpedia PostgreSQL connection URL.",
    )
    parser.add_argument("--local-webapp-url", default=None, help="Local Neuronpedia webapp base URL.")
    parser.add_argument(
        "--local-api-key",
        default=None,
        help="Local Neuronpedia API key used for remote graph deletion.",
    )
    parser.add_argument("--slug-prefix", default=None, help="Optional slug prefix filter.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report candidate graph artifacts without deleting DB rows or files.",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Delete local DB rows only and leave graph artifact files on disk.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = cleanup_local_graph_artifacts(
        args.work_root,
        local_db_url=args.local_db_url,
        local_webapp_url=args.local_webapp_url,
        local_api_key=args.local_api_key,
        slug_prefix=args.slug_prefix,
        dry_run=args.dry_run,
        remove_files=not args.keep_files,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
