# pyright: reportMissingTypeStubs=false
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from neuronpedia_utils.local_db_import import (  # type: ignore[import-untyped]
    DEFAULT_COLUMNAR_COPY_IMPORT_TABLES,
    DEFAULT_COLUMNAR_IMPORT_TABLES,
    DEFAULT_IMPORT_MODE_CONFIGS,
    NeuronpediaBundleImportParity,
    NeuronpediaBundleSummaryParity,
    NeuronpediaExportBundleSummary,
    NeuronpediaLocalDBImportError,
    NeuronpediaLocalImportSummary,
    benchmark_neuronpedia_export_bundle_local_db_modes as _benchmark_neuronpedia_export_bundle_local_db_modes,
    compare_neuronpedia_export_bundle_summaries,
    compare_neuronpedia_export_bundle_to_import_summary,
    summarize_neuronpedia_export_bundle,
    summarize_neuronpedia_export_bundle_arrow,
    summarize_neuronpedia_export_bundle_parquet,
)
from neuronpedia_utils.local_db_import import (  # type: ignore[import-untyped]
    import_saedashboard_columnar_bundle_local_db as _import_saedashboard_columnar_bundle_local_db,
    import_neuronpedia_export_bundle_local_db as _import_neuronpedia_export_bundle_local_db,
)

DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL = os.getenv("LOCAL_NEURONPEDIA_WEBAPP_URL", "http://localhost:3000")
DEFAULT_LOCAL_NEURONPEDIA_DB_ENV_VARS = (
    "LOCAL_NEURONPEDIA_DB_URL",
    "POSTGRES_URL_NON_POOLING",
    "DATABASE_URL",
)
DEFAULT_LOCAL_NEURONPEDIA_DOCKER_HOSTNAME = "postgres"
DEFAULT_LOCAL_NEURONPEDIA_HOST = "127.0.0.1"
DEFAULT_LOCAL_NEURONPEDIA_POSTGRES_PORT_ENV = "POSTGRES_HOST_PORT"
DEFAULT_LOCAL_NEURONPEDIA_DB_TIMEOUT_SECONDS = 5

NeuronpediaDBError = NeuronpediaLocalDBImportError

__all__ = [
    "DEFAULT_COLUMNAR_COPY_IMPORT_TABLES",
    "DEFAULT_COLUMNAR_IMPORT_TABLES",
    "DEFAULT_IMPORT_MODE_CONFIGS",
    "DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL",
    "LocalNeuronpediaServiceStatus",
    "NeuronpediaBundleImportParity",
    "NeuronpediaBundleSummaryParity",
    "NeuronpediaDBError",
    "NeuronpediaExportBundleSummary",
    "NeuronpediaLocalImportSummary",
    "benchmark_neuronpedia_export_bundle_local_db_modes",
    "check_local_neuronpedia_services",
    "compare_neuronpedia_export_bundle_summaries",
    "compare_neuronpedia_export_bundle_to_import_summary",
    "import_neuronpedia_export_bundle_local_db",
    "import_saedashboard_columnar_bundle_local_db",
    "resolve_local_neuronpedia_db_url",
    "rewrite_container_postgres_url_for_host",
    "summarize_neuronpedia_export_bundle",
    "summarize_neuronpedia_export_bundle_arrow",
    "summarize_neuronpedia_export_bundle_parquet",
]


@dataclass(frozen=True)
class LocalNeuronpediaServiceStatus:
    """Availability snapshot for the local Neuronpedia webapp and Postgres services."""

    webapp_url: str
    webapp_available: bool
    webapp_status_code: int | None
    webapp_error: str | None
    db_url_redacted: str | None
    db_available: bool
    db_error: str | None


def _redact_connection_url(connection_url: str) -> str:
    parsed = urlparse(connection_url)
    if parsed.password is None:
        return connection_url
    username = parsed.username or ""
    netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@", 1)
    if username and not netloc.startswith(username):
        netloc = f"{username}:***@{parsed.hostname or ''}"
        if parsed.port is not None:
            netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def rewrite_container_postgres_url_for_host(
    connection_url: str,
    *,
    env: Mapping[str, str] | None = None,
    container_hostname: str = DEFAULT_LOCAL_NEURONPEDIA_DOCKER_HOSTNAME,
    host: str = DEFAULT_LOCAL_NEURONPEDIA_HOST,
    port_env_var: str = DEFAULT_LOCAL_NEURONPEDIA_POSTGRES_PORT_ENV,
) -> str:
    """Rewrite a docker-only Postgres URL to the host-mapped local port when configured."""

    env_map = dict(os.environ if env is None else env)
    parsed = urlparse(connection_url)
    if parsed.hostname != container_hostname:
        return connection_url
    host_port = env_map.get(port_env_var)
    if not host_port:
        return connection_url
    username = parsed.username or ""
    password = parsed.password or ""
    auth_prefix = username
    if password:
        auth_prefix = f"{auth_prefix}:{password}"
    if auth_prefix:
        auth_prefix = f"{auth_prefix}@"
    netloc = f"{auth_prefix}{host}:{host_port}"
    return urlunparse(parsed._replace(netloc=netloc))


def resolve_local_neuronpedia_db_url(
    local_db_url: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the best local Neuronpedia Postgres URL from explicit input or environment."""

    env_map = dict(os.environ if env is None else env)
    candidate = local_db_url
    if not candidate:
        for env_var in DEFAULT_LOCAL_NEURONPEDIA_DB_ENV_VARS:
            candidate = env_map.get(env_var)
            if candidate:
                break
    if not candidate:
        raise NeuronpediaDBError(
            "Could not resolve a local Neuronpedia DB URL. Set LOCAL_NEURONPEDIA_DB_URL or POSTGRES_URL_NON_POOLING."
        )
    return rewrite_container_postgres_url_for_host(candidate, env=env_map)


def check_local_neuronpedia_services(
    *,
    local_db_url: str | None = None,
    webapp_url: str = DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL,
    timeout_seconds: int = DEFAULT_LOCAL_NEURONPEDIA_DB_TIMEOUT_SECONDS,
) -> LocalNeuronpediaServiceStatus:
    """Probe the local Neuronpedia webapp and Postgres services without raising on failure."""

    import psycopg

    resolved_db_url: str | None = None
    db_url_redacted: str | None = None
    db_available = False
    db_error: str | None = None
    try:
        resolved_db_url = resolve_local_neuronpedia_db_url(local_db_url)
        db_url_redacted = _redact_connection_url(resolved_db_url)
        with psycopg.connect(resolved_db_url, connect_timeout=timeout_seconds) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        db_available = True
    except Exception as exc:  # pragma: no cover - exercised in integration contexts
        db_error = str(exc)

    webapp_available = False
    webapp_status_code: int | None = None
    webapp_error: str | None = None
    try:
        request = Request(webapp_url, method="GET")
        with urlopen(request, timeout=timeout_seconds) as response:
            webapp_status_code = getattr(response, "status", None)
            webapp_available = webapp_status_code is not None and 200 <= webapp_status_code < 500
    except Exception as exc:  # pragma: no cover - exercised in integration contexts
        webapp_error = str(exc)

    return LocalNeuronpediaServiceStatus(
        webapp_url=webapp_url,
        webapp_available=webapp_available,
        webapp_status_code=webapp_status_code,
        webapp_error=webapp_error,
        db_url_redacted=db_url_redacted,
        db_available=db_available,
        db_error=db_error,
    )


def import_neuronpedia_export_bundle_local_db(
    export_root: Path | str,
    *,
    local_db_url: str | None = None,
    prefer_arrow_for_tables: Iterable[str] = (),
    prefer_copy_for_tables: Iterable[str] = (),
    artifact_format_by_table: dict[str, str] | None = None,
) -> NeuronpediaLocalImportSummary:
    """Import a Neuronpedia export bundle through the Neuronpedia-owned importer."""

    return _import_neuronpedia_export_bundle_local_db(
        export_root,
        local_db_url=resolve_local_neuronpedia_db_url(local_db_url),
        prefer_arrow_for_tables=prefer_arrow_for_tables,
        prefer_copy_for_tables=prefer_copy_for_tables,
        artifact_format_by_table=artifact_format_by_table,
    )


def import_saedashboard_columnar_bundle_local_db(
    columnar_root: Path | str,
    *,
    local_db_url: str | None = None,
    model_id: str,
    source_set_name: str,
    source_id: str,
    creator_id: str,
    decode_token_ids: Any,
    created_at: Any = None,
    activation_id_prefix: str = "columnar-activation",
    pad_token_id: int | None = None,
    hook_name: str | None = None,
    chunk_size: int = 65000,
    **metadata_kwargs: Any,
) -> NeuronpediaLocalImportSummary:
    """Import SAEDashboard columnar artifacts through the Neuronpedia-owned importer."""

    return _import_saedashboard_columnar_bundle_local_db(
        columnar_root,
        local_db_url=resolve_local_neuronpedia_db_url(local_db_url),
        model_id=model_id,
        source_set_name=source_set_name,
        source_id=source_id,
        creator_id=creator_id,
        decode_token_ids=decode_token_ids,
        created_at=created_at,
        activation_id_prefix=activation_id_prefix,
        pad_token_id=pad_token_id,
        hook_name=hook_name,
        chunk_size=chunk_size,
        **metadata_kwargs,
    )


def benchmark_neuronpedia_export_bundle_local_db_modes(
    export_root: Path | str,
    *,
    local_db_url: str | None = None,
    import_modes: dict[str, dict[str, Any]] | None = None,
    rollback_each_mode: bool = True,
) -> dict[str, NeuronpediaLocalImportSummary]:
    """Benchmark Neuronpedia-owned local import modes after Interpretune URL resolution."""

    return _benchmark_neuronpedia_export_bundle_local_db_modes(
        export_root,
        local_db_url=resolve_local_neuronpedia_db_url(local_db_url),
        import_modes=import_modes,
        rollback_each_mode=rollback_each_mode,
    )
