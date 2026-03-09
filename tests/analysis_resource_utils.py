"""Shared helpers for memory-aware analysis fixtures in tests.

Preferred pattern for expensive analysis-session tests:

1. Choose fixture scope dynamically: ``class`` on higher-RAM runners and
   ``function`` on lower-RAM runners.
2. Deep-copy only the values a test class needs.
3. Reuse those copied values across methods via the public ``extract_values()``
   helper on ``AnalysisExtractionMixin``.

Tradeoff summary:
- Dynamic fixture scope is preferred over ad hoc transient session builders
  because it keeps lifecycle management inside pytest's fixture machinery and
  avoids duplicating analysis-session construction logic.
- Low-RAM runners trade some setup time for prompt teardown.
- High-RAM runners retain class-scoped reuse for throughput.
"""

from __future__ import annotations

import gc
import os
import shutil
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Mapping

import psutil
import torch

from interpretune.analysis.core import LatentAnalysisDict

from tests.runif import get_runner_ram_gb

ANALYSIS_LOW_RAM_GB = 32
FixtureScope = Literal["function", "class", "module", "session"]
RESOURCE_DEBUG_ENV_VARS = ("IT_ANALYSIS_RESOURCE_DEBUG", "IT_OP_SERIALIZATION_RESOURCE_DEBUG")


class ExtractedAnalysisStore:
    """Minimal ``AnalysisStore`` view backed by a selected subset of fields."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def __deepcopy__(self, memo: dict[int, Any]) -> ExtractedAnalysisStore:
        copied = type(self)(**deepcopy(self._fields, memo))
        memo[id(self)] = copied
        return copied

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def by_latent_model(self, field_name: str, stack_latents: bool = True) -> LatentAnalysisDict:
        """Match ``AnalysisStore.by_latent_model`` for selected nested fields."""

        values = getattr(self, field_name)
        assert values, f"No values found for field {field_name}"
        if not isinstance(values[0], dict):
            raise TypeError(
                f"Values for field {field_name} must be dictionaries to be transformed into a LatentAnalysisDict"
            )

        result = LatentAnalysisDict()
        sae_names = values[0].keys()
        for sae in sae_names:
            if isinstance(values[0][sae], dict) and stack_latents:
                batch_tensors = []
                for batch in values:
                    latent_tensors = [tensor for tensor in batch[sae].values()]
                    batch_tensors.append(torch.stack(latent_tensors) if latent_tensors else None)
                result[sae] = batch_tensors  # type: ignore[assignment]
            else:
                result[sae] = [  # type: ignore[assignment]
                    None if isinstance(batch[sae], list) and not batch[sae] else batch[sae] for batch in values
                ]
        return result


class ExtractedFixturePayload:
    """Minimal generic payload container for extracted fixture data."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def __deepcopy__(self, memo: dict[int, Any]) -> ExtractedFixturePayload:
        copied = type(self)(**deepcopy(self._fields, memo))
        memo[id(self)] = copied
        return copied

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


def analysis_resource_debug_enabled(env_var_names: tuple[str, ...] = RESOURCE_DEBUG_ENV_VARS) -> bool:
    """Return ``True`` when any configured resource-debug environment flag is enabled."""

    return any(os.environ.get(env_name, "0") == "1" for env_name in env_var_names)


def _existing_disk_target(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def log_resource_snapshot(
    label: str,
    *,
    paths: list[str | Path] | tuple[str | Path, ...] = (),
    prefix: str = "analysis_resource_debug",
    env_var_names: tuple[str, ...] = RESOURCE_DEBUG_ENV_VARS,
) -> None:
    """Emit a compact RSS and disk-usage snapshot when resource debugging is enabled."""

    if not analysis_resource_debug_enabled(env_var_names=env_var_names):
        return

    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024**3)
    parts = [f"rss_gb={rss_gb:.2f}"]

    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        disk_target = _existing_disk_target(path)
        disk_usage = shutil.disk_usage(disk_target)
        parts.extend(
            [
                f"path{index}={path}",
                f"used_gb{index}={disk_usage.used / (1024**3):.2f}",
                f"free_gb{index}={disk_usage.free / (1024**3):.2f}",
            ]
        )

    print(f"[{prefix}] {label}: " + " ".join(parts))


def extract_result_dataset_metadata(result: Any) -> dict[str, list[str] | int]:
    """Extract lightweight dataset metadata from an analysis result."""

    dataset = getattr(result, "dataset", None)
    column_names = list(getattr(dataset, "column_names", []) or [])
    num_rows = int(getattr(dataset, "num_rows", 0) or 0)
    return {"column_names": column_names, "num_rows": num_rows}


def build_analysis_fixture_payload_extractor(
    *,
    field_names: list[str] | tuple[str, ...] = (),
    include_dataset_metadata: bool = False,
    extra_extractors: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Callable[[Any], ExtractedFixturePayload]:
    """Build a reusable extractor for lightweight, cacheable analysis-fixture payloads."""

    selected_fields = tuple(field_names)
    extra_extractors = dict(extra_extractors or {})

    def _extractor(fixture: Any) -> ExtractedFixturePayload:
        payload_fields: dict[str, Any] = {}
        if include_dataset_metadata:
            payload_fields["metadata"] = extract_result_dataset_metadata(fixture.result)
        if selected_fields:
            payload_fields["store"] = ExtractedAnalysisStore(
                **{field_name: getattr(fixture.result, field_name) for field_name in selected_fields}
            )
        for name, extractor in extra_extractors.items():
            payload_fields[name] = extractor(fixture)
        return ExtractedFixturePayload(**payload_fields)

    return _extractor


def analysis_fixture_scope(
    *,
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
    low_ram_scope: FixtureScope = "function",
    high_ram_scope: FixtureScope = "class",
) -> FixtureScope:
    """Return a pytest fixture scope based on available system RAM."""

    return low_ram_scope if get_runner_ram_gb() <= min_ram_gb else high_ram_scope


@contextmanager
def conditional_clean_cpu(fixture: Any, min_ram_gb: int = ANALYSIS_LOW_RAM_GB):
    """Clear large fixture payloads after extraction on low-RAM runners."""

    try:
        yield fixture
    finally:
        if get_runner_ram_gb() <= min_ram_gb:
            for attr in ("result", "runner", "run_config", "it_session"):
                if hasattr(fixture, attr):
                    setattr(fixture, attr, None)
            gc.collect()


def extract_fixture_result(request: Any, fixture_key: str, min_ram_gb: int = ANALYSIS_LOW_RAM_GB) -> Any:
    """Deep-copy a fixture result and release heavyweight fixture state when possible."""

    fixture = request.getfixturevalue(fixture_key)
    with conditional_clean_cpu(fixture, min_ram_gb=min_ram_gb) as active_fixture:
        return deepcopy(active_fixture.result)


def extract_fixture_data(
    request: Any,
    fixture_key: str,
    extractor: Any,
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
) -> Any:
    """Extract arbitrary data from a fixture and clear heavyweight state on low-RAM runners."""

    fixture = request.getfixturevalue(fixture_key)
    with conditional_clean_cpu(fixture, min_ram_gb=min_ram_gb) as active_fixture:
        extracted = extractor(active_fixture)
    return deepcopy(extracted)


def extract_analysis_store_fields(
    request: Any,
    fixture_key: str,
    field_names: list[str] | tuple[str, ...],
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
) -> ExtractedAnalysisStore:
    """Copy selected ``fixture.result`` fields into a lightweight analysis-store view."""

    selected_fields = tuple(field_names)
    return extract_fixture_data(
        request,
        fixture_key,
        lambda fixture: ExtractedAnalysisStore(
            **{field_name: getattr(fixture.result, field_name) for field_name in selected_fields}
        ),
        min_ram_gb=min_ram_gb,
    )


class AnalysisExtractionMixin:
    """Class-level cache helper for memory-heavy analysis tests.

    Subclasses implement ``build_extracted_values()`` and test methods call the
    public ``extract_values()`` helper to populate and reuse the copied payloads.
    """

    _extracted: ClassVar[Any | None] = None
    _fixture_payload_cache: ClassVar[dict[Any, Any]] = {}

    def build_extracted_values(self, request: Any) -> Any:
        raise NotImplementedError

    def extract_values(self, request: Any) -> Any:
        cls = type(self)
        extracted = getattr(cls, "_extracted", None)
        if extracted is None:
            extracted = self.build_extracted_values(request)
            cls._extracted = extracted
        return extracted

    def extract_cached_fixture_data(
        self,
        request: Any,
        fixture_key: str,
        extractor: Callable[[Any], Any],
        *,
        cache_key: Any | None = None,
        min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
    ) -> Any:
        cls = type(self)
        cache = cls.__dict__.get("_fixture_payload_cache")
        if cache is None:
            cache = {}
            cls._fixture_payload_cache = cache

        resolved_cache_key = fixture_key if cache_key is None else cache_key
        cached = cache.get(resolved_cache_key)
        if cached is None:
            cached = extract_fixture_data(request, fixture_key, extractor, min_ram_gb=min_ram_gb)
            cache[resolved_cache_key] = cached
        return cached

    @classmethod
    def clear_extracted_values(cls) -> None:
        cls._extracted = None
        if "_fixture_payload_cache" in cls.__dict__:
            cls._fixture_payload_cache = {}
        gc.collect()
