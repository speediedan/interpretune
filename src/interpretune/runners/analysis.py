from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL

import contextlib
import torch
from typing import Any, TYPE_CHECKING
import logging
from functools import partialmethod
from pathlib import Path
from inspect import signature

from tqdm.auto import tqdm
from datasets import Dataset
from datasets.fingerprint import generate_random_fingerprint

from interpretune.analysis import schema_to_features
from interpretune.analysis.execution import activated_analysis_cfg
from interpretune.base import _call_itmodule_hook, ITDataModule
from interpretune.runners.core import SessionRunner, _print_metrics, profiling, run_step
from interpretune.runners.distributed import (
    DistributedRun,
    distributed_dataloader,
    gathered_weights,
    gradient_sync_off,
)
from interpretune.analysis.share_exec import ShareExecution, plan_share_execution, share_execution
from interpretune.utils import MisconfigurationException
from interpretune.protocol import AllPhases, AnalysisStoreProtocol, TestLoadable
from interpretune.config import AnalysisRunnerCfg, AnalysisCfg, init_analysis_cfgs
from interpretune.utils.exceptions import handle_exception_with_debug_dump


if TYPE_CHECKING:
    from interpretune.adapters import ITModule


log = logging.getLogger(__name__)


class ShareLog:
    """The rows each of a rank's batches yielded, as ``(epoch, batch index, rows)``, in the order it yielded them.

    An object rather than a list: ``datasets`` splits list-valued generator kwargs into shards.
    """

    def __init__(self) -> None:
        self.batches: list[tuple[int, int, int]] = []


def merged_row_order(logs: list[list[tuple[int, int, int]]]) -> list[int]:
    """Row positions, in the ranks' shards laid end to end, that put every row in single-process order.

    ``logs`` holds each rank's :class:`ShareLog` entries, by rank. A row's single-process position is set by its
    batch's epoch and index in the whole batch sequence, then its place among that batch's rows.
    """
    keyed, offset = [], 0
    for log_entries in logs:
        for epoch, batch_idx, rows in log_entries:
            keyed.extend(((epoch, batch_idx, i), offset + i) for i in range(rows))
            offset += rows
    return [position for _, position in sorted(keyed)]


def analysis_store_generator(
    module: ITModule,
    datamodule: TestLoadable,
    limit_analysis_batches: int = -1,
    step_fn: str = "analysis_step",
    max_epochs: int = 1,
    *args,
    dist_run: DistributedRun | None = None,
    share_log: ShareLog | None = None,
    **kwargs,
):
    """Yield analysis results batch by batch, so a store can be written incrementally.

    A generator rather than a list: analysis outputs (caches, graphs, attributions) are large enough
    that materializing every batch before writing is what exhausts memory on a long run.

    Under a ``dist_run`` (placement ``tp``) the loader is rebuilt so that every rank takes the same batches in the
    same order, shuffled alike from a shared seed each epoch: the ranks make every call on each batch together. Under a
    data-parallel ``dist_run`` (``ddp``) each rank takes its own batches of that shared sequence instead
    (:class:`~interpretune.runners.distributed.BatchShare`). Steps see each batch's index in the whole sequence, as one
    process would, and ``limit_analysis_batches`` counts the whole sequence's batches. ``share_log`` records the rows
    each batch yielded, which is what puts the ranks' rows back in order.
    """
    # TODO: should we create separate dataset phase subsplits (per epoch)?
    # TODO: allow for custom dataloader associations
    if max_epochs < 1:
        # `range(max_epochs)` with the runner config's default of -1 is EMPTY, so the store would be built from
        # zero rows and fail two frames away inside `datasets` ("corresponds to no data"), with nothing pointing
        # at the setting. An analysis run with no epochs is a request for nothing; refuse it by name.
        raise ValueError(
            f"analysis requires max_epochs >= 1 (got {max_epochs}); pass max_epochs=1 to the analysis runner config"
        )
    dataloader = datamodule.test_dataloader()
    if dist_run is not None:
        dataloader = distributed_dataloader(dataloader, dist_run, share_batches=True)
    shares, index = (dist_run.data_shares, dist_run.data_index) if dist_run is not None else (1, 0)
    test_ctx = {"module": module, "as_generator": True}
    module.reset_analysis_metrics()
    with profiling(module, "analysis") as prof:
        for epoch_idx in range(max_epochs):
            module.current_epoch = epoch_idx
            for sampler in (getattr(dataloader, "batch_sampler", None), getattr(dataloader, "sampler", None)):
                if (set_epoch := getattr(sampler, "set_epoch", None)) is not None:
                    set_epoch(epoch_idx)
                    break
            for local_idx, batch in tqdm(enumerate(dataloader)):
                batch_idx = local_idx * shares + index  # the batch's index in the whole sequence
                if batch_idx >= limit_analysis_batches >= 0:
                    break
                # TODO: move previous inference_mode toggle to IT dispatch logic
                rows = 0
                for row in run_step(step_fn=step_fn, batch=batch, batch_idx=batch_idx, **test_ctx):
                    rows += 1
                    yield row
                if share_log is not None:
                    share_log.batches.append((epoch_idx, batch_idx, rows))
                if prof is not None:
                    prof.step()
            _call_itmodule_hook(module, hook_name="on_analysis_epoch_end", hook_msg="Running analysis epoch end hooks")
            if dist_run is not None and dist_run.data_shares > 1:
                _pool_analysis_metrics(module, dist_run)
            if metrics := module.flush_analysis_metrics():
                _print_metrics(f"Analysis epoch {epoch_idx} end", metrics)
            # Declared op state (`op_state` trait) is reset here only for ops that ask for it; by default
            # it accumulates across epochs. `batch_idx` restarts at 0 above, so the previous
            # `batch_idx == 0` reset heuristic silently discarded every epoch but the last.
            epoch_cfg = getattr(module, "analysis_cfg", None)
            if epoch_cfg is not None and hasattr(epoch_cfg, "reset_op_state"):
                epoch_cfg.reset_op_state(epoch_boundary=True)


def maybe_init_analysis_cfg(module: "ITModule", analysis_cfg: AnalysisCfg | None = None, **kwargs) -> dict:
    """Initialize analysis configuration if needed and return updated kwargs.

    Args:
        module: The module to set up the analysis configuration for
        analysis_cfg: Optional analysis configuration to use
        **kwargs: Additional keyword arguments

    Returns:
        Updated kwargs with init_analysis_cfg parameters removed
    """
    if analysis_cfg is not None:
        # Set module's analysis_cfg
        module.analysis_cfg = analysis_cfg  # protocol compatibility

        # Extract and pop any AnalysisCfg.apply-specific kwargs
        init_analysis_cfg_params = signature(init_analysis_cfgs).parameters
        init_analysis_cfg_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in init_analysis_cfg_params and k != "analysis_cfgs"
        }

        # Only initialize if not already applied to this module
        if not module.analysis_cfg.applied_to(module):  # protocol provides applied_to
            init_analysis_cfgs(module, module.analysis_cfg, **init_analysis_cfg_kwargs)  # protocol compatibility

    return kwargs


def dataset_features_and_format(module: "ITModule", kwargs: dict) -> tuple[dict, dict, dict]:
    """Generate dataset features and formatting parameters based on module configuration.

    Args:
        module: The module containing analysis configuration

    Returns:
        Tuple containing:
        - features: Dataset features derived from schema
        - it_format_kwargs: Kwargs for interpretune format
        - kwargs: Updated kwargs dict
    """
    # Generate appropriate features based on module configuration and current op context
    # Handle different schema sources based on configuration
    explicit_schema = getattr(module.analysis_cfg, "output_schema", None)  # protocol provides output_schema
    if explicit_schema is not None:
        # Use explicitly provided output schema when op is None
        features = schema_to_features(module=module, schema=explicit_schema)
        schema_source = explicit_schema
        # Handle the case where schema_source is an OpWrapper or AnalysisOp instead of OpSchema
        from interpretune.analysis.ops.base import AnalysisOpLike

        if isinstance(schema_source, AnalysisOpLike):
            schema_source = schema_source.output_schema
    elif hasattr(module.analysis_cfg, "op") and module.analysis_cfg.op is not None:  # protocol provides op
        features = schema_to_features(module=module, op=module.analysis_cfg.op)  # protocol compatibility
        schema_source = module.analysis_cfg.op.output_schema  # protocol provides op
    else:
        # For manual analysis steps without any schema, use a default empty features dict
        features = {}
        schema_source = {}

    # Convert ColCfg objects to dicts for JSON serialization
    serializable_col_cfg = {k: v.to_dict() for k, v in schema_source.items()} if schema_source else {}
    it_format_kwargs = dict(col_cfg=serializable_col_cfg)
    kwargs.update(schema_source=schema_source, serializable_col_cfg=serializable_col_cfg)
    return features, it_format_kwargs, kwargs


def generate_analysis_dataset(
    module,
    features,
    it_format_kwargs,
    gen_kwargs,
    split="validation",
    dataset_fingerprint: str | None = None,
    generator_cache_dir: str | Path | None = None,
    writer_batch_size: int | None = None,
    dist_run: DistributedRun | None = None,
    **kwargs,
) -> Dataset | None:
    """Generate a dataset for analysis using the ITAnalysisFormatter.

    Args:
        module: The module to analyze
        features: Features derived from schema
        it_format_kwargs: Kwargs for interpretune format
        gen_kwargs: Dictionary of generator parameters (module, datamodule, limit_analysis_batches, etc.)
        split: The split to use for dataset generation
        dataset_fingerprint: A stable base key the split and output features are mixed into. It must
            identify the whole computation: two generations sharing it, the split and the output features
            share cache files, so a key that omits an op's inputs serves one computation another's rows.
            None (default) keeps the per-run random fingerprint. Named as the `AnalysisRunnerCfg` field
            because the runner forwards its config by field name.
        generator_cache_dir: Where datasets writes generator cache files. None (default) keeps
            `output_store.cache_dir`. A persistent directory requires `dataset_fingerprint`: with random
            fingerprints it would only accumulate files no run can reuse, so that combination is refused.
        writer_batch_size: How many records the writer buffers before each flush to the cache file. Peak host
            memory during generation follows this times the record size, not the number of records. None (default)
            keeps the `datasets` default.
        dist_run: This rank's place in a run under placement ``tp``, or None for one process. The ranks compute
            every batch together, and only the rank that writes outputs (:attr:`DistributedRun.writes_outputs`)
            writes the dataset; the others run the same steps and discard their rows (see
            :func:`_follow_generation`).
        **kwargs: Additional arguments for error context

    Returns:
        Dataset: The generated dataset with interpretune format, or None on a rank that does not write outputs

    Raises:
        Exception: If dataset generation fails, with detailed debug information
    """
    # The runner forwards its whole config as kwargs, and anything unrecognized lands in **kwargs as error context.
    # A key passed under the wrong name was absorbed there once, so every run fell back to a random fingerprint
    # while still writing into a persistent directory: cache files accumulated that nothing could ever read.
    if "fingerprint" in kwargs:
        raise TypeError(
            "generate_analysis_dataset takes its cache key as `dataset_fingerprint` (the AnalysisRunnerCfg field); "
            "`fingerprint` would be absorbed as error context and silently ignored"
        )
    if writer_batch_size is not None and writer_batch_size < 1:
        raise ValueError(f"writer_batch_size must be a positive number of records or None, got {writer_batch_size}")
    if generator_cache_dir is not None and dataset_fingerprint is None:
        raise ValueError(
            f"generator_cache_dir={generator_cache_dir!s} was given without a dataset_fingerprint: every generation "
            "would get a random fingerprint, so a persistent directory would only accumulate cache files no run "
            "can reuse. Pass a dataset_fingerprint that identifies the whole computation, or leave "
            "generator_cache_dir unset."
        )
    # Use an explicit fingerprint so datasets doesn't hash gen_kwargs via dill.
    # gen_kwargs includes the module/datamodule, and hashing those objects can serialize model
    # weights and fail on memory-constrained runners. The fingerprint is random per run unless the
    # caller passes a deterministic base key, which is mixed with the split and output features. Those
    # two do not separate different op inputs, so the base has to: a general deterministic AnalysisStore
    # cache key remains post-MVP work under issue #183.
    if dataset_fingerprint is None:
        resolved_fingerprint = generate_random_fingerprint()
    else:
        import hashlib
        import json

        from datasets.fingerprint import Hasher

        mix = Hasher.hash(
            {
                "base": dataset_fingerprint,
                "split": str(split),
                "features": json.dumps(features, sort_keys=True, default=str),
            }
        )
        resolved_fingerprint = hashlib.sha256(mix.encode()).hexdigest()[:32]
    # None passes nothing, so the datasets default stays in place
    writer_kwargs: dict[str, Any] = {} if writer_batch_size is None else {"writer_batch_size": writer_batch_size}
    if dist_run is not None:
        _require_supported_groups(dist_run)
        if dist_run.data_shares > 1 and dataset_fingerprint is not None:
            raise MisconfigurationException(
                f"Under placement {dist_run.placement!r} each rank generates its own share of the store's rows, and"
                " a share served from the generator cache would not say which batches its rows came from, so the"
                " shares could not be put back in order. Leave `dataset_fingerprint` (and `case_key_target_part`)"
                " unset under a data-parallel placement."
            )
        if not dist_run.writes_outputs:
            _follow_generation(gen_kwargs, dist_run)
            return None
    # only a lockstep group's ranks wait on each other's generation; a data share's rank generates alone
    lockstep = dist_run if dist_run is not None and dist_run.lockstep_size > 1 else None
    generator, announce = _announcing_generator(lockstep)
    try:
        dataset = Dataset.from_generator(
            generator,
            features=features,
            cache_dir=str(
                generator_cache_dir
                if generator_cache_dir is not None
                else module.analysis_cfg.output_store.cache_dir  # protocol provides output_store
            ),
            gen_kwargs=gen_kwargs,
            split=split,  # pyright: ignore[reportArgumentType]  # str acceptable for NamedSplit at runtime
            fingerprint=resolved_fingerprint,
            **writer_kwargs,
        ).with_format("interpretune", **it_format_kwargs)
        announce(False)  # a cache hit never started the generator, so the other ranks must not start it either
        return dataset  # datasets compatibility
    except Exception as e:
        announce(False)  # releases the other ranks when the failure came before generation started
        # improve visibility of errors since they can otherwise be obscured by the dataset generator
        context_data = (
            features,  # Features derived from schema
            gen_kwargs,  # Arguments for the analysis generator
            it_format_kwargs,  # Arguments for interpretune format
            kwargs,  # Additional user-provided arguments
        )
        handle_exception_with_debug_dump(e, context_data, "dataset_generation")
        raise  # Re-raise after logging to ensure proper error handling


class AnalysisRowWriter:
    """Write analysis rows to an Arrow cache file as they arrive, for a loop that cannot be a generator.

    ``generate_analysis_dataset`` pulls rows from a generator it drives. A loop driven elsewhere (a Lightning
    ``Trainer``) pushes them instead: each row is encoded with the run's features and buffered up to
    ``writer_batch_size`` rows before it is flushed, as ``Dataset.from_generator`` does, so host memory does not follow
    the number of rows. :meth:`finalize` returns the dataset; :meth:`abort` removes the file of a run that failed.

    Args:
        features: The dataset features, from :func:`dataset_features_and_format`.
        cache_dir: Where the Arrow file is written; the returned dataset reads it from there.
        it_format_kwargs: Kwargs for the ``interpretune`` format.
        writer_batch_size: Rows buffered before each flush. ``None`` keeps the ``datasets`` default.
    """

    def __init__(
        self,
        features: Any,
        cache_dir: str | Path,
        it_format_kwargs: dict[str, Any],
        writer_batch_size: int | None = None,
    ) -> None:
        from datasets import Features
        from datasets.arrow_writer import ArrowWriter

        self.features = Features(features) if features else None
        self.it_format_kwargs = it_format_kwargs
        fingerprint = generate_random_fingerprint()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(cache_dir) / f"analysis-rows-{fingerprint}.arrow"
        self._writer = ArrowWriter(
            features=self.features,
            path=str(self.path),
            fingerprint=fingerprint,
            writer_batch_size=writer_batch_size,
        )

    def __len__(self) -> int:
        """The number of rows written so far, buffered rows included."""
        return len(self._writer)

    def write(self, row: Any) -> None:
        """Encode and write one row."""
        encoded: Any = self.features.encode_example(row) if self.features is not None else row
        self._writer.write(encoded)

    def finalize(self) -> Dataset:
        """Flush and close the file, and return its rows as a dataset in the ``interpretune`` format."""
        from datasets import DatasetInfo, NamedSplit

        self._writer.finalize()
        dataset = Dataset.from_file(
            str(self.path), info=DatasetInfo(features=self.features), split=NamedSplit("validation")
        )
        return dataset.with_format("interpretune", **self.it_format_kwargs)

    def abort(self) -> None:
        """Close the file and remove it."""
        self._writer.close()
        self.path.unlink(missing_ok=True)


def _fingerprintable(value: Any, *, path: str = "root") -> Any:
    """Reduce ``value`` to plain data the datasets Hasher takes deterministically.

    Tensors hash by value (shape, dtype and bytes), mappings sort by key, and every branch that cannot
    be identified stably -- a closure, an unhandled type -- raises :class:`TypeError` naming the path
    instead of falling back to ``repr``, whose address component would silently make the key unique per
    run (the failure #618 measured: entries nothing could reuse).
    """
    import dataclasses
    import enum

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().to("cpu", copy=True).contiguous()
        import hashlib

        return {
            "kind": "tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, range):
        return {"kind": "range", "start": value.start, "stop": value.stop, "step": value.step}
    import hashlib

    import numpy as np  # guaranteed present: datasets (a hard dependency) requires it

    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, enum.Enum):
        return {"kind": "enum", "type": f"{type(value).__module__}.{type(value).__qualname__}", "value": value.value}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": sorted(
                ((repr(k), _fingerprintable(v, path=f"{path}[{k!r}]")) for k, v in value.items()),
                key=lambda item: item[0],
            ),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "list" if isinstance(value, list) else "tuple",
            "items": [_fingerprintable(v, path=f"{path}[{i}]") for i, v in enumerate(value)],
        }
    if isinstance(value, (set, frozenset)):
        return {
            "kind": "set",
            "items": sorted((repr(_fingerprintable(v, path=f"{path}{{...}}")) for v in value)),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": sorted(
                (
                    (f.name, _fingerprintable(getattr(value, f.name), path=f"{path}.{f.name}"))
                    for f in dataclasses.fields(value)
                ),
                key=lambda item: item[0],
            ),
        }
    from interpretune.analysis.ops.base import AnalysisOp

    if isinstance(value, AnalysisOp):
        return {
            "kind": "op",
            "name": value.name,
            "definition": _fingerprintable(
                {
                    "description": value.description,
                    "input_schema": value.input_schema,
                    "output_schema": value.output_schema,
                    "impl_params": value.impl_params,
                    "uses_default_hooks": value.uses_default_hooks,
                    "requires_grad": value.requires_grad,
                    "per_latent_preds": value.per_latent_preds,
                    "required_tensor_ops": sorted(value.required_tensor_ops),
                    "required_position_scopes": sorted(str(s) for s in value.required_position_scopes),
                    "required_techniques": sorted(value.required_techniques),
                },
                path=f"{path}<op:{value.name}>",
            ),
        }
    if callable(value):
        return _fingerprint_callable(value, path)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return {
            "kind": "todict",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _fingerprintable(value.to_dict(), path=f"{path}.to_dict()"),
        }
    raise TypeError(
        f"cannot fingerprint {path}: no stable normalization for {type(value).__module__}.{type(value).__qualname__}"
    )


def _fingerprint_callable(value: Any, path: str) -> Any:
    """Identify a callable by what it does, not where it lives.

    Module-level named callables key on ``module.qualname``. Closures and lambdas have no stable
    address, but their behavior is stable: key on the code (bytecode plus constants, recursively,
    so a nested comprehension counts) together with default arguments and closed-over cell values.
    Same behavior always keys together; different closed-over values key apart. Anything without
    introspectable code (builtins without ``__code__`` get the qualname path only if module-level,
    bound methods whose receiver is not a module or class) is refused by name: guessing would serve
    one computation another's rows.
    """
    import types

    if isinstance(value, types.MethodType):
        receiver = value.__self__
        if isinstance(receiver, types.ModuleType) or isinstance(receiver, type):
            return _fingerprint_callable(value.__func__, path)
        raise TypeError(
            f"cannot fingerprint {path}: bound method {value!r} closes over a "
            f"{type(receiver).__name__} instance with no stable normalization"
        )
    qualname = getattr(value, "__qualname__", None)
    module = getattr(value, "__module__", None)
    code = getattr(value, "__code__", None)
    if code is None:
        if qualname and module and "<locals>" not in qualname and "<lambda>" not in qualname:
            return {"kind": "callable", "id": f"{module}.{qualname}"}
        raise TypeError(f"cannot fingerprint {path}: {value!r} has neither stable identity nor introspectable code")
    return {
        "kind": "function-code",
        "id": f"{module}.{qualname}" if qualname and module else None,
        "code": _fingerprint_code(code, path),
        "defaults": _fingerprintable(getattr(value, "__defaults__", None), path=f"{path}.__defaults__"),
        "kwdefaults": _fingerprintable(getattr(value, "__kwdefaults__", None), path=f"{path}.__kwdefaults__"),
        "closure": _fingerprintable(
            tuple(cell.cell_contents for cell in (value.__closure__ or ())), path=f"{path}.__closure__"
        ),
    }


def _fingerprint_code(code: Any, path: str) -> Any:
    """A code object as plain data: bytecode plus recursively normalized constants.

    ``co_code`` fixes the operations; ``co_consts`` fixes the values baked into them (numbers,
    strings, and nested code objects from comprehensions and inner lambdas). Names, varnames and
    filenames are deliberately excluded: renaming a local or moving the file changes neither behavior
    nor rows, so keying on them would only split entries that should share.
    """
    import hashlib
    import types

    def _const(value: Any, at: str) -> Any:
        if isinstance(value, types.CodeType):
            return _fingerprint_code(value, at)
        if value is Ellipsis:
            return {"kind": "ellipsis"}
        return _fingerprintable(value, path=at)

    return {
        "kind": "code",
        "sha256": hashlib.sha256(code.co_code).hexdigest(),
        "consts": [_const(c, f"{path}.co_consts[{i}]") for i, c in enumerate(code.co_consts)],
    }


def analysis_case_fingerprint(
    *,
    target_part: str,
    op: Any,
    run_inputs: dict[str, Any] | None = None,
    names_filter: Any = None,
    package_version: str | None = None,
    extra_case_part: dict[str, Any] | None = None,
) -> str:
    """Build the per-case generator-cache key #621 needs: one entry per computation, shared across runs.

    The mix covers the target part (composition, datamodule flavour, batch shape, suite inputs -- built
    by the caller, which owns those), the op (name AND definition, so a redefined hub op cannot serve
    another definition's rows), the case's ``run_inputs`` hashed by value, the ``names_filter``, and the
    package version. Two cases sharing any of those share cache files, which is exactly the #606 failure
    (eight intervention cases landing in two entries); two runs sharing all of them share everything.
    ``extra_case_part`` carries small execution selectors (step function, manual-step handling) that are
    part of the computation but live outside op, inputs and filter.

    The version must be the full code identity (commit included): a coarse version would serve rows
    generated by different code. That makes CI write-only per commit by construction, so a shared
    directory also needs :func:`prune_generator_cache` -- key correctness and storage boundedness are
    the two halves of this issue, and neither substitutes for the other.
    """
    from datasets.fingerprint import Hasher

    import hashlib

    if package_version is None:
        try:
            from importlib.metadata import version

            package_version = version("interpretune")
        except Exception:
            package_version = "unknown"
    mix = Hasher.hash(
        {
            "target": target_part,
            "op": _fingerprintable(op, path="op"),
            "run_inputs": _fingerprintable(dict(run_inputs or {}), path="run_inputs"),
            "names_filter": _fingerprintable(names_filter, path="names_filter"),
            "extra": _fingerprintable(dict(extra_case_part or {}), path="extra"),
            "package_version": str(package_version),
        }
    )
    return hashlib.sha256(mix.encode()).hexdigest()[:32]


def analysis_cfg_fingerprint(
    cfg: AnalysisCfg,
    *,
    target_part: str,
    package_version: str | None = None,
) -> str:
    """Thin wrapper reading the case identity off a resolved :class:`AnalysisCfg`.

    Takes the resolved ``cfg.op`` (post ``init_analysis_cfgs``) rather than ``target_op`` so the key
    names what actually runs, plus ``run_inputs`` and ``names_filter`` as given. ``step_fn``,
    ``ignore_manual`` and ``latent_analysis_targets`` ride along because each changes what executes:
    in particular two cases with ``names_filter=None`` but different latent targets materialize
    different filters, so the targets must key apart even before materialization.
    """
    return analysis_case_fingerprint(
        target_part=target_part,
        op=cfg.op if cfg.op is not None else cfg.target_op,
        run_inputs={**(cfg.run_inputs or {}), "__latent_targets__": cfg.latent_analysis_targets},
        names_filter=cfg.names_filter,
        package_version=package_version,
        extra_case_part={"step_fn": cfg.step_fn, "ignore_manual": cfg.ignore_manual},
    )


def prune_generator_cache(cache_dir: str | Path, *, max_bytes: int) -> dict[str, int]:
    """Delete oldest-first generator cache files over ``max_bytes``; the bounded-storage half of #621.

    Datasets names generator files ``<fingerprint>-*.arrow`` (plus adjacent ``.json``/``.lock`` sidecars
    sharing the fingerprint stem), so eviction is per fingerprint stem by oldest mtime. Returns counts
    ``{"kept_stems": n, "evicted_stems": m, "evicted_bytes": b}``; a missing directory is a no-op.
    """
    import os

    root = Path(cache_dir)
    stats = {"kept_stems": 0, "evicted_stems": 0, "evicted_bytes": 0}
    if not root.is_dir():
        return stats
    stems: dict[str, dict[str, Any]] = {}
    for entry in os.scandir(root):
        if not entry.is_file():
            continue
        stem = entry.name.split("-", 1)[0] if "-" in entry.name else entry.name
        record = stems.setdefault(stem, {"size": 0, "mtime": entry.stat().st_mtime, "files": []})
        stat = entry.stat()
        record["size"] += stat.st_size
        record["mtime"] = min(record["mtime"], stat.st_mtime)
        record["files"].append(entry.path)
    total = sum(record["size"] for record in stems.values())
    for stem in sorted(stems, key=lambda s: stems[s]["mtime"]):
        if total <= max_bytes:
            break
        record = stems[stem]
        for path in record["files"]:
            try:
                os.remove(path)
            except OSError:
                pass
        total -= record["size"]
        stats["evicted_stems"] += 1
        stats["evicted_bytes"] += record["size"]
    stats["kept_stems"] = len(stems) - stats["evicted_stems"]
    return stats


def _leader(dist_run: DistributedRun) -> int:
    """The rank that writes the outputs of ``dist_run``'s lockstep group."""
    return dist_run.rank - dist_run.rank % dist_run.lockstep_size


def _require_supported_groups(dist_run: DistributedRun) -> None:
    """One lockstep group of every rank (``tp``), or one rank per data share (``ddp``): not groups of several ranks
    that each take a share, which FSDP2 shard dimensions above one would make."""
    if dist_run.lockstep_size not in (1, dist_run.world_size):
        raise NotImplementedError(
            f"The analysis runner places one lockstep group ('tp') or one rank per data share ('ddp'), not"
            f" {dist_run.data_shares} groups of {dist_run.lockstep_size} ranks ({dist_run.placement!r})."
        )


def _broadcast_flag(value: int, dist_run: DistributedRun) -> int:
    """``value`` from the leader, on every rank.

    A CPU tensor, so a ``cpu:gloo,cuda:nccl`` group sends it by gloo.
    """
    import torch.distributed as dist

    flag = torch.tensor([value], dtype=torch.long)
    dist.broadcast(flag, src=_leader(dist_run))
    return int(flag.item())


def _announcing_generator(dist_run: DistributedRun | None) -> tuple[Any, Any]:
    """The generator the writing rank hands ``datasets``, and the function that tells the other ranks whether it
    ran.

    ``Dataset.from_generator`` does not call the generator when its cache already holds the fingerprint's rows, so
    under placement ``tp`` the other ranks cannot know in advance whether to run the shared steps: running them alone
    would wait forever on a collective the writing rank never makes. The generator announces that it started before
    its first step, and a generation that ends without starting announces that it did not. Each run announces once.
    """
    if dist_run is None:
        return analysis_store_generator, lambda started: None
    announced = False

    def announce(started: bool) -> None:
        nonlocal announced
        if not announced:
            announced = True
            _broadcast_flag(int(started), dist_run)

    def generator(**gen_kwargs: Any):
        announce(True)
        yield from analysis_store_generator(**gen_kwargs)

    return generator, announce


def _follow_generation(gen_kwargs: dict, dist_run: DistributedRun) -> None:
    """Run the steps of the writing rank's generation alongside it, discarding the rows.

    Under placement ``tp`` each step is computed by all ranks together (the split layers' collectives), so a rank
    that writes nothing still runs every step. It waits for the writing rank's announcement first, and runs nothing
    when the writing rank's cache served the rows without generating them.
    """
    if _broadcast_flag(0, dist_run):
        for _ in analysis_store_generator(**gen_kwargs):
            pass


def _share_store(
    dataset: Dataset | None, save_dir: Path, it_format_kwargs: dict, dist_run: DistributedRun | None
) -> Dataset:
    """Save the writing rank's dataset, and give every rank the saved store.

    Under placement ``tp`` the writing rank saves it and the other ranks load the same files once it has: ranks on
    several hosts need a filesystem they share. The writing rank's directory is used on every rank, since each
    rank's own session may name another.
    """
    if dist_run is None:
        assert dataset is not None
        dataset.save_to_disk(save_dir)
        return dataset
    import torch.distributed as dist
    from datasets import load_from_disk

    if dist_run.writes_outputs:
        assert dataset is not None
        dataset.save_to_disk(save_dir)
    shared = [str(save_dir)]
    dist.broadcast_object_list(shared, src=_leader(dist_run), device=torch.device("cpu"))
    dist.barrier()
    if dataset is not None:
        return dataset
    loaded = load_from_disk(shared[0])
    assert isinstance(loaded, Dataset)
    return loaded.with_format("interpretune", **it_format_kwargs)


def _merge_shares(
    dataset: Dataset | None,
    save_dir: Path,
    it_format_kwargs: dict,
    dist_run: DistributedRun,
    share_log: ShareLog,
    module: Any = None,
    plan: ShareExecution | None = None,
) -> Dataset:
    """Put the ranks' shares of the store back in single-process order, and give every rank the merged store.

    Each rank saves its share beside the store's directory and reports it with its :class:`ShareLog`; rank 0 lays the
    shares end to end, orders the rows by their batches (:func:`merged_row_order`) and saves the merged store, which
    every rank then loads. Each rank removes its share once the merged store is saved. Ranks on several hosts need a
    filesystem they share.

    With a ``plan`` whose op is replayed, each rank also reports the inputs it recorded for the op, and rank 0 runs
    the op again over them in one-process order and rewrites its output columns before saving
    (:func:`~interpretune.analysis.share_exec.replay_rows`).
    """
    import shutil

    import torch.distributed as dist
    from datasets import concatenate_datasets, load_from_disk

    assert dataset is not None
    share_dir = save_dir.with_name(f"{save_dir.name}.share{dist_run.data_index}")
    dataset.save_to_disk(share_dir)
    # a manual analysis step never calls the configured op, so it records nothing and nothing is replayed
    replaying = plan is not None and plan.replay_op is not None and bool(plan.recorded)
    recorded = plan.recorded if replaying and plan is not None else {}
    reports: list[Any] = [None] * dist_run.world_size
    dist.all_gather_object(reports, (str(share_dir), share_log.batches, recorded))
    if dist_run.rank == 0:
        shares = [load_from_disk(path) for path, _, _ in reports]
        assert all(isinstance(share, Dataset) for share in shares)
        laid = concatenate_datasets(shares)  # pyright: ignore[reportArgumentType]  # Datasets, checked above
        merged_out = laid.select(merged_row_order([entries for _, entries, _ in reports]))
        if replaying and plan is not None:
            merged_out = _replayed(merged_out, module, plan, reports)
        merged_out.save_to_disk(save_dir)
    shared = [str(save_dir)]
    dist.broadcast_object_list(shared, src=0, device=torch.device("cpu"))
    dist.barrier()
    merged = load_from_disk(shared[0])
    assert isinstance(merged, Dataset)
    shutil.rmtree(share_dir, ignore_errors=True)
    return merged.with_format("interpretune", **it_format_kwargs)


def _pool_analysis_metrics(module: Any, dist_run: DistributedRun) -> None:
    """Give every rank the epoch's ``log_analysis`` values from all the data shares, before they are averaged.

    Each share analyzes its own batches, so one rank's values average a part of the epoch: a plausible number that is
    not the one-process average. The writing rank of each lockstep group contributes its group's values (the group's
    ranks computed the same ones), and every rank then averages the pooled values, which is the one-process average.
    """
    pool_analysis_metrics(module, contribute=dist_run.writes_outputs)


def pool_analysis_metrics(module: Any, contribute: bool = True) -> None:
    """Replace this rank's pending ``log_analysis`` values with every rank's, over the default process group.

    A collective: every rank calls it. A rank that passes ``contribute=False`` adds nothing but still receives the
    pooled values, which is how ranks that computed the same values as another rank avoid counting them twice.
    """
    import torch.distributed as dist

    pending = module.__dict__.get("_pending_analysis_metrics") or {}
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, pending if contribute else {})
    pooled: dict[str, list[float]] = {}
    for share in gathered:
        for name, values in (share or {}).items():
            pooled.setdefault(name, []).extend(values)
    module.__dict__["_pending_analysis_metrics"] = pooled


def _replayed(merged: Dataset, module: Any, plan: ShareExecution, reports: list[Any]) -> Dataset:
    """``merged`` with the replay op's output columns rewritten from its replay in one-process order."""
    from interpretune.analysis.share_exec import replay_rows

    entries = sorted(entry for _, batches, _ in reports for entry in batches)
    if any(rows != 1 for _, _, rows in entries):
        raise MisconfigurationException(
            f"Replaying {plan.replay_op!r} after the shares are merged needs one row per batch, but some batches"
            " yielded another number."
        )
    recorded = {key: value for _, _, rank_recorded in reports for key, value in rank_recorded.items()}
    order = [(epoch, batch_idx) for epoch, batch_idx, _ in entries]
    rows = replay_rows(module, module.analysis_cfg, plan, recorded, order)

    def rewrite(row: dict, index: int) -> dict:
        return {**row, **rows[order[index]]}

    return merged.with_format(None).map(rewrite, with_indices=True, features=merged.features)


def _prefix_pass(module: Any, datamodule: Any, plan: ShareExecution, dist_run: DistributedRun, kwargs: dict) -> None:
    """Run the composition up to its last consumed stateful op over the whole batch sequence, in one-process order,
    keeping the outputs of the batches this rank takes (:func:`~interpretune.analysis.share_exec.prefix_pass`).

    Every rank runs it alike, so no rank sends another its prefix outputs; the prefix is the cheap part when it reads
    an input store, as ``concept_direction`` does. State starts fresh, resets at epoch boundaries as the op asks, and
    is cleared again afterwards, so generation starts from fresh state as it would without a prefix.
    """
    from interpretune.analysis.execution import execute_analysis_op
    from interpretune.analysis.share_exec import prefix_pass

    cfg = module.analysis_cfg
    loader = distributed_dataloader(datamodule.test_dataloader(), dist_run, share_batches=True, whole_sequence=True)
    limit = int(kwargs.get("limit_analysis_batches", -1))
    cfg.reset_op_state()
    with prefix_pass(plan):
        for epoch_idx in range(int(kwargs.get("max_epochs", 1))):
            module.current_epoch = epoch_idx
            if (set_epoch := getattr(getattr(loader, "batch_sampler", None), "set_epoch", None)) is not None:
                set_epoch(epoch_idx)
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= limit >= 0:
                    break
                prefix = execute_analysis_op(module, module.batch_to_device(batch), batch_idx, analysis_cfg=cfg)
                if batch_idx % dist_run.data_shares == dist_run.data_index:
                    plan.prefix_outputs[(epoch_idx, batch_idx)] = prefix
            cfg.reset_op_state(epoch_boundary=True)
    cfg.reset_op_state()


def core_analysis_loop(
    module: ITModule,
    datamodule: ITDataModule,
    limit_analysis_batches: int = -1,
    step_fn: str = "analysis_step",
    max_epochs: int = 1,
    analysis_cfg: AnalysisCfg | None = None,
    *args,
    dist_run: DistributedRun | None = None,
    **kwargs,
):
    """Create dataset using the ITAnalysisFormatter for optimal handling of analysis data.

    Under a ``dist_run`` (placement ``tp``) every rank runs every step, one rank writes the store, and every rank
    returns it. Under a data-parallel ``dist_run`` (``ddp``, FSDP2 replication) each rank analyzes its own batches with
    FSDP2's gradient reduction off, and the shares are merged into the store one process would write, which every rank
    returns. Weights an op reads directly are gathered whole once for the run (:func:`gathered_weights`).
    """
    # Initialize analysis configuration if needed
    kwargs = maybe_init_analysis_cfg(module, analysis_cfg, **kwargs)
    if analysis_cfg is not None:
        step_fn = analysis_cfg.step_fn

    # The runner owns the process-global grad state for the whole run: the start hook sets it from the active op and
    # the end hook restores it, but an op raising inside dataset generation used to skip the end hook and leak the
    # state into whatever ran next. Restoring in a `finally` leaves the process as the run found it on every exit.
    shared_out = dist_run is not None and dist_run.data_shares > 1
    plan = None
    if dist_run is not None and shared_out:
        op = getattr(getattr(module, "analysis_cfg", None), "op", None)
        plan = plan_share_execution(op) if op is not None else None
    share_log = ShareLog() if shared_out else None
    grad_enabled_before = torch.is_grad_enabled()
    _call_itmodule_hook(module, hook_name="on_analysis_start", hook_msg="Running analysis start hooks")
    try:
        with (
            gathered_weights(),
            gradient_sync_off(module) if shared_out else contextlib.nullcontext(),
            share_execution(plan),
        ):
            if plan is not None and plan.prefix_end is not None and dist_run is not None:
                _prefix_pass(
                    module,
                    datamodule,
                    plan,
                    dist_run,
                    {"limit_analysis_batches": limit_analysis_batches, "max_epochs": max_epochs},
                )
            # Generate features and format parameters
            features, it_format_kwargs, kwargs = dataset_features_and_format(module, kwargs)

            # Create generator kwargs dictionary
            gen_kwargs = dict(
                module=module,
                datamodule=datamodule,
                limit_analysis_batches=limit_analysis_batches,
                step_fn=step_fn,
                max_epochs=max_epochs,
                dist_run=dist_run,
                share_log=share_log,
            )

            # Generate the dataset (None on a rank of a 'tp' run that does not write it)
            dataset = generate_analysis_dataset(
                module, features, it_format_kwargs, gen_kwargs, dist_run=dist_run, **kwargs
            )

            save_dir = Path(module.analysis_cfg.output_store.save_dir)  # protocol provides output_store
            if dist_run is not None and share_log is not None:
                dataset = _merge_shares(dataset, save_dir, it_format_kwargs, dist_run, share_log, module, plan)
            else:
                dataset = _share_store(dataset, save_dir, it_format_kwargs, dist_run)
            # Assign dataset to analysis store
            module.analysis_cfg.output_store.dataset = dataset  # protocol provides output_store

            # Run analysis end hooks
            _call_itmodule_hook(module, hook_name="on_analysis_end", hook_msg="Running analysis end hooks")
    finally:
        torch.set_grad_enabled(grad_enabled_before)
    return module.analysis_cfg.output_store  # protocol provides output_store


class AnalysisRunner(SessionRunner):
    """Trainer subclass with analysis orchestration logic."""

    # Narrowed from the base runner: __init__ only ever stores an AnalysisRunnerCfg here.
    run_cfg: AnalysisRunnerCfg

    def __init__(self, run_cfg: AnalysisRunnerCfg | dict[str, Any], *args: Any, **kwargs: Any) -> None:
        run_cfg = run_cfg if isinstance(run_cfg, AnalysisRunnerCfg) else AnalysisRunnerCfg(**run_cfg)
        super().__init__(run_cfg, *args, **kwargs)
        # Extend supported commands to include analysis
        self.supported_commands = (*self.supported_commands, "analysis")
        self.analysis_results = {}

    def _run(self, phase, loop_fn, step_fn: str | None = None, *args: Any, **kwargs: Any) -> Any | None:
        self.phase = AllPhases[phase]
        phase_artifacts = loop_fn(step_fn=step_fn, **self.run_cfg.__dict__, dist_run=getattr(self, "dist_run", None))
        self.it_session_end()
        return phase_artifacts

    analysis = partialmethod(_run, phase="analysis", loop_fn=core_analysis_loop)

    def run_analysis(
        self,
        analysis_cfgs: AnalysisCfg | Any | list[AnalysisCfg | Any] | None = None,
        cache_dir: str | Path | None = None,
        op_output_dataset_path: str | Path | None = None,
        **kwargs,
    ) -> AnalysisStoreProtocol | dict[str, AnalysisStoreProtocol]:
        """Unified method to run analysis operations based on the provided configuration.

        Args:
            analysis_cfgs: Configuration for analysis operations. Can be:
                - An AnalysisCfg instance
                - An AnalysisOp instance
                - A list of AnalysisCfg or AnalysisOp instances
            cache_dir: Optional override for the cache directory
            op_output_dataset_path: Optional override for the output dataset path
            **kwargs: Additional arguments to pass to the analysis function

        Returns:
            For a single analysis configuration: The result of that analysis
            For multiple analysis configurations: A dictionary mapping configuration names to results
        """
        # Reset analysis results for this run
        self.analysis_results = {}

        # Update configuration parameters if provided
        if cache_dir is not None:
            self.run_cfg.cache_dir = cache_dir
        if op_output_dataset_path is not None:
            self.run_cfg.op_output_dataset_path = op_output_dataset_path

        # Update analysis_cfgs if provided
        if analysis_cfgs is not None:
            self.run_cfg.analysis_cfgs = analysis_cfgs
            # # Process the new analysis_cfgs
            # self.run_cfg._process_analysis_cfgs()

        # Initialize analysis configurations
        init_analysis_cfgs(
            module=self.run_cfg.module,  # pyright: ignore[reportArgumentType]  # protocol compatibility
            analysis_cfgs=self.run_cfg._processed_analysis_cfgs,
            cache_dir=self.run_cfg.cache_dir,
            op_output_dataset_path=self.run_cfg.op_output_dataset_path,
            latent_analysis_targets=self.run_cfg.latent_analysis_targets,
            ignore_manual=self.run_cfg.ignore_manual,
        )

        # Check if we have any analysis configurations to process
        if not self.run_cfg._processed_analysis_cfgs:
            raise ValueError("No analysis configurations provided. Please specify analysis_cfgs.")

        # Run each analysis configuration
        for cfg in self.run_cfg._processed_analysis_cfgs:
            result = self._run_analysis_cfg(cfg)
            self.analysis_results[cfg.name] = result

        # Return a single result for a single config, dictionary for multiple
        if len(self.analysis_results) == 1:
            return next(iter(self.analysis_results.values()))
        else:
            return self.analysis_results

    def _run_analysis_cfg(self, analysis_cfg: AnalysisCfg) -> Any:
        """Run a single analysis operation with the provided configuration.

        Owns the run boundary of the declared `op_state` lifecycle: cleared before the first batch,
        released once the store has been materialized. Note this is NOT `activated_analysis_cfg`,
        which is re-entered per batch by `execute_analysis_op` and so cannot delimit a run.
        """
        ignore_manual = bool(getattr(self.run_cfg, "ignore_manual", False))
        analysis_cfg.reset_op_state()
        try:
            with activated_analysis_cfg(self.run_cfg.module, analysis_cfg, ignore_manual=ignore_manual):
                self._derive_case_key(analysis_cfg)
                return self.analysis(step_fn=analysis_cfg.step_fn, **self.run_cfg.__dict__)
        finally:
            analysis_cfg.finalize_op_state()

    def _derive_case_key(self, analysis_cfg: AnalysisCfg) -> None:
        """Set this run's ``dataset_fingerprint`` from the case identity, post-activation.

        Called after the cfg is applied (op resolved, ``names_filter`` materialized): setup mutates
        the cfg in place, so a key taken before the run would differ from the same case's key on any
        later run. Only when ``run_cfg.case_key_target_part`` is set; otherwise the configured
        ``dataset_fingerprint`` (usually ``None``: random per run) passes through untouched.
        """
        target_part = getattr(self.run_cfg, "case_key_target_part", None)
        if target_part is None:
            return
        self.run_cfg.dataset_fingerprint = analysis_cfg_fingerprint(analysis_cfg, target_part=target_part)
