"""Shared helpers for memory-aware analysis fixtures in tests.

Preferred pattern for expensive analysis-session tests:

1. Choose fixture scope dynamically: ``class`` on higher-RAM runners and
    ``function`` on lower-RAM runners.
2. Declare fixture extraction needs at the class level with
    ``AnalysisFixtureSpec``.
3. Reuse the copied values across methods via ``extract_values()`` for eager
    whole-class caches or via ``extract_field_store()`` /
    ``extract_dataset_metadata()`` for lazy per-fixture access.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Mapping

import psutil
import torch

from interpretune.analysis.core import LatentAnalysisDict

from tests.runif import get_runner_ram_gb

ANALYSIS_LOW_RAM_GB = 32
FixtureScope = Literal["function", "class", "module", "session"]
RESOURCE_DEBUG_ENV_VARS = ("IT_ANALYSIS_RESOURCE_DEBUG", "IT_OP_SERIALIZATION_RESOURCE_DEBUG")


@dataclass(frozen=True, kw_only=True)
class AnalysisFixtureSpec:
    """Declarative configuration for extracting and caching a test fixture payload."""

    fixture_key: str
    field_names: tuple[str, ...] = ()
    include_dataset_metadata: bool = False
    include_result: bool = False
    result_field_name: str = "result"
    extra_extractors: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict)
    cache_key: Any | None = None
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB

    @property
    def uses_payload_extractor(self) -> bool:
        return bool(self.field_names or self.include_dataset_metadata or self.include_result or self.extra_extractors)


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
        latent_model_names = values[0].keys()
        for latent_model in latent_model_names:
            if isinstance(values[0][latent_model], dict) and stack_latents:
                batch_tensors = []
                for batch in values:
                    latent_tensors = [tensor for tensor in batch[latent_model].values()]
                    batch_tensors.append(torch.stack(latent_tensors) if latent_tensors else None)
                result[latent_model] = batch_tensors  # type: ignore[assignment]
            else:
                result[latent_model] = [  # type: ignore[assignment]
                    None if isinstance(batch[latent_model], list) and not batch[latent_model] else batch[latent_model]
                    for batch in values
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
    include_result: bool = False,
    result_field_name: str = "result",
    extra_extractors: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Callable[[Any], ExtractedFixturePayload]:
    """Build a reusable extractor for lightweight, cacheable analysis-fixture payloads."""

    selected_fields = tuple(field_names)
    extra_extractors = dict(extra_extractors or {})

    def _extractor(fixture: Any) -> ExtractedFixturePayload:
        payload_fields: dict[str, Any] = {}
        if include_dataset_metadata:
            payload_fields["metadata"] = extract_result_dataset_metadata(fixture.result)
        if include_result:
            payload_fields[result_field_name] = fixture.result
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

    Subclasses can either override ``build_extracted_values()`` directly or set
    ``_analysis_fixture_specs`` so the mixin can lazily extract full results,
    lightweight stores, dataset metadata, and custom payloads on their behalf.
    """

    _extracted: ClassVar[Any | None] = None
    _fixture_payload_cache: ClassVar[dict[Any, Any]] = {}
    _analysis_fixture_specs: ClassVar[dict[str, AnalysisFixtureSpec]] = {}

    @classmethod
    def get_analysis_fixture_specs(cls) -> dict[str, AnalysisFixtureSpec]:
        """Return the configured fixture specs for the test class."""

        return dict(getattr(cls, "_analysis_fixture_specs", {}))

    @classmethod
    def resolve_fixture_alias(cls, fixture_ref: str) -> str:
        """Resolve an alias or raw fixture key to the configured alias name."""

        specs = cls.get_analysis_fixture_specs()
        if fixture_ref in specs:
            return fixture_ref
        for alias, spec in specs.items():
            if spec.fixture_key == fixture_ref:
                return alias
        raise KeyError(f"Unknown fixture alias or key: {fixture_ref}")

    @classmethod
    def get_analysis_fixture_spec(cls, fixture_ref: str) -> AnalysisFixtureSpec:
        """Return the declarative fixture spec for an alias or raw fixture key."""

        return cls.get_analysis_fixture_specs()[cls.resolve_fixture_alias(fixture_ref)]

    @staticmethod
    def build_fixture_payload_extractor(spec: AnalysisFixtureSpec) -> Callable[[Any], Any]:
        """Build the effective extractor for a configured analysis fixture."""

        if spec.uses_payload_extractor:
            return build_analysis_fixture_payload_extractor(
                field_names=spec.field_names,
                include_dataset_metadata=spec.include_dataset_metadata,
                include_result=spec.include_result,
                result_field_name=spec.result_field_name,
                extra_extractors=spec.extra_extractors,
            )
        return lambda fixture: fixture.result

    def build_extracted_values(self, request: Any) -> Any:
        specs = type(self).get_analysis_fixture_specs()
        if not specs:
            raise NotImplementedError

        return {alias: self.extract_fixture_value(request, alias) for alias in specs}

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

    def extract_fixture_value(self, request: Any, fixture_ref: str) -> Any:
        """Extract and cache the configured value for a fixture alias or raw fixture key."""

        cls = type(self)
        alias = cls.resolve_fixture_alias(fixture_ref)
        spec = cls.get_analysis_fixture_spec(alias)
        cache_key = alias if spec.cache_key is None else spec.cache_key
        return self.extract_cached_fixture_data(
            request,
            spec.fixture_key,
            cls.build_fixture_payload_extractor(spec),
            cache_key=cache_key,
            min_ram_gb=spec.min_ram_gb,
        )

    def extract_payload(self, request: Any, fixture_ref: str) -> ExtractedFixturePayload:
        """Return the configured lightweight payload for a fixture alias or raw key."""

        payload = self.extract_fixture_value(request, fixture_ref)
        if not isinstance(payload, ExtractedFixturePayload):
            raise TypeError(f"Fixture {fixture_ref} is configured for full-result extraction, not payload extraction")
        return payload

    def extract_dataset_metadata(self, request: Any, fixture_ref: str) -> dict[str, list[str] | int]:
        """Return cached dataset metadata for a configured payload fixture."""

        payload = self.extract_payload(request, fixture_ref)
        metadata = getattr(payload, "metadata", None)
        if metadata is None:
            raise AttributeError(f"Fixture {fixture_ref} does not include dataset metadata")
        return metadata

    def extract_field_store(
        self,
        request: Any,
        fixture_ref: str,
        *field_names: str,
    ) -> ExtractedAnalysisStore:
        """Return a cached lightweight store and validate required extracted fields."""

        payload = self.extract_payload(request, fixture_ref)
        store = getattr(payload, "store", None)
        if not isinstance(store, ExtractedAnalysisStore):
            raise AttributeError(f"Fixture {fixture_ref} does not include an extracted store")

        missing_fields = [field_name for field_name in field_names if not hasattr(store, field_name)]
        assert not missing_fields, f"Fixture {fixture_ref} missing requested extracted fields: {missing_fields}"
        return store

    @classmethod
    def clear_extracted_values(cls) -> None:
        cls._extracted = None
        if "_fixture_payload_cache" in cls.__dict__:
            cls._fixture_payload_cache = {}
        gc.collect()
