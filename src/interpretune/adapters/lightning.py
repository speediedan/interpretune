from typing import TYPE_CHECKING, Any

import torch

from interpretune.base import ITDataModule, BaseITModule
from interpretune.utils import _LIGHTNING_AVAILABLE, MisconfigurationException
from interpretune.protocol import Adapter
from interpretune.adapters import CompositionRegistry

if TYPE_CHECKING:
    from interpretune.config import ITConfig


if _LIGHTNING_AVAILABLE:
    from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
    from lightning.pytorch import LightningDataModule, LightningModule

    class LightningAdapter:
        """Adapter composing interpretune modules with Lightning's ``LightningModule``/``DataModule``."""

        # Host-provided attributes: supplied by BaseITModule and LightningModule through the MRO.
        if TYPE_CHECKING:
            it_cfg: ITConfig
            model: torch.nn.Module

        from interpretune.metadata import ITClassMetadata  # local import to avoid cycles

        _it_cls_metadata = ITClassMetadata(
            core_to_framework_attrs_map={
                "_it_lr_scheduler_configs": (
                    "trainer.strategy.lr_scheduler_configs",
                    None,
                    "No lr_scheduler_configs have been set.",
                ),
                "_it_optimizers": ("trainer.optimizers", None, "No optimizers have been set yet."),
                "_log_dir": ("trainer.model._trainer.log_dir", None, "No log_dir has been set yet."),
                "_datamodule": (
                    "trainer.datamodule",
                    None,
                    "Could not find datamodule reference (has it been attached yet?)",
                ),
                "_current_epoch": ("trainer.current_epoch", 0, ""),
                "_global_step": ("trainer.global_step", 0, ""),
            },
            property_composition={
                "device": {
                    "enabled": True,
                    "target": _DeviceDtypeModuleMixin,
                    "dispatch": _DeviceDtypeModuleMixin.device,
                }
            },
            gen_prepares_inputs_sigs=("_prepare_model_inputs",),
        )

        def setup(self, *args, **kwargs) -> None:
            """Refuse the core-runner profiler selection, which a Lightning Trainer never reads, then set up.

            The Trainer runs its own loops, so ``it_cfg.profiler_cfg`` would profile nothing here; Lightning's native
            ``--trainer.profiler`` is the supported route.
            """
            if self.it_cfg.profiler_cfg.which != "none":
                raise MisconfigurationException(
                    f"it_cfg.profiler_cfg.which={self.it_cfg.profiler_cfg.which!r} selects a core-runner profiler, "
                    "which a Lightning Trainer never runs. Configure Lightning's profiler instead "
                    "(--trainer.profiler, or Trainer(profiler=...)) and leave profiler_cfg at its default."
                )
            return super().setup(*args, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]  # cooperative MRO call

        def on_train_start(self) -> None:
            """Force the model into training mode before training starts, then defer to Lightning.

            Lightning normally handles this, but a run that skips sanity checking can reach training with the model
            still in eval mode; setting it explicitly closes that edge case.
            """
            # ensure model is in training mode (e.g. needed for some edge cases w/ skipped sanity checking)
            self.model.train()
            return super().on_train_start()  # pyright: ignore[reportAttributeAccessIssue]  # LightningModule method when mixed in

        def _analysis_cfg_of(self):
            """Read the bound analysis config without tripping the unset warning.

            The ``analysis_cfg`` property warns on every access when no config is set; routing and
            hooks read it on every batch, so they must go through ``it_cfg`` directly. A missing
            config reads as None either way.
            """
            return getattr(getattr(self, "it_cfg", None), "analysis_cfg", None)

        def _ensure_analysis_setup(self) -> None:
            """Run the analysis_cfg setup a Trainer loop needs, exactly as the AnalysisRunner does.

            Under a Trainer, nothing invokes the AnalysisRunner, so a module composed for analysis
            would reach its first batch without an analysis_step. This runs the runner's own setup
            (``init_analysis_cfgs`` for a config not yet applied to this module) and leaves the choice of
            step to ``AnalysisCfg.apply``, so ``ignore_manual`` means the same thing here as there: a
            re-derived check that kept any existing manual step ran it even when the config said to
            ignore it and generate one from the op. Its output directory comes from ``core_log_dir``,
            which under a Trainer is the Trainer's log directory.

            It then routes predict batches through the configured step. This cannot live on the class:
            task mixins (e.g. RTEBoolqSteps) define their own predict_step earlier in the MRO, so a
            class-level override would never run for them. Wrapping the instance attribute once reaches
            every composition regardless of MRO order.
            """
            analysis_cfg = self._analysis_cfg_of()
            if analysis_cfg is None or getattr(analysis_cfg, "op", None) is None:
                return
            if not analysis_cfg.applied_to(self):
                from interpretune.config.runner import init_analysis_cfgs

                init_analysis_cfgs(self, analysis_cfg)  # pyright: ignore[reportArgumentType]  # a mixin; the composed module satisfies it
            if getattr(self, "_it_predict_wrapped", False):
                return
            original = self.predict_step

            def _it_analysis_predict_step(batch, batch_idx: int, dataloader_idx: int = 0):
                ran, result = self._run_configured_step(
                    self._analysis_cfg_of() or analysis_cfg, batch, batch_idx, dataloader_idx
                )
                return result if ran else original(batch, batch_idx, dataloader_idx)

            self.predict_step = _it_analysis_predict_step
            self._it_predict_wrapped = True

        def _run_configured_step(self, analysis_cfg, batch, batch_idx: int, dataloader_idx: int):
            """Run the step the config names (its ``step_fn``); ``(False, None)`` when there is none to run.

            Generated steps stream, so an iterator is materialized; any other return passes through untouched,
            preserving the Trainer's per-batch output contract.
            """
            step = getattr(self, getattr(analysis_cfg, "step_fn", "analysis_step"), None)
            if not callable(step):
                return False, None
            from collections.abc import Iterator, Mapping

            result = step(batch, batch_idx, dataloader_idx)
            result = list(result) if isinstance(result, Iterator) else result
            writer = getattr(self, "_it_analysis_row_writer", None)
            if writer is not None:
                # an AnalysisStoreCallback is writing this predict run's rows: every item the step yields is a row,
                # as in the core analysis loop, and a mapping returned whole is one row
                yielded: Any = result
                try:
                    rows = [yielded] if isinstance(yielded, Mapping) else iter(yielded)
                except TypeError as err:
                    raise MisconfigurationException(
                        f"the analysis step `{getattr(analysis_cfg, 'step_fn', 'analysis_step')}` returned a"
                        f" {type(result).__name__}, which is not an iterable of rows, so no AnalysisStore can be built"
                        " from it"
                    ) from err
                for row in rows:
                    writer.write(row)
            return True, result

        def on_predict_start(self) -> None:
            """Ensure analysis setup ran before a Trainer predict loop, then defer to Lightning."""
            self._ensure_analysis_setup()
            return super().on_predict_start()  # pyright: ignore[reportAttributeAccessIssue]  # LightningModule method when mixed in

        def on_test_start(self) -> None:
            """Ensure analysis setup ran before a Trainer test loop, then defer to Lightning."""
            self._ensure_analysis_setup()
            return super().on_test_start()  # pyright: ignore[reportAttributeAccessIssue]  # LightningModule method when mixed in

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            """Route predict batches through analysis_step when analysis is configured.

            When the module carries an analysis_cfg with an op, predict executes the configured analysis step (manual or
            generated, resolved through the config's step_fn). Generated steps stream, so only iterators are
            materialized; any other return value passes through untouched, preserving the Trainer's per-batch output
            contract. Without analysis configured, this defers to the predict_step it shadowed.
            """
            analysis_cfg = self._analysis_cfg_of()
            if analysis_cfg is not None and getattr(analysis_cfg, "op", None) is not None:
                self._ensure_analysis_setup()
                ran, result = self._run_configured_step(analysis_cfg, batch, batch_idx, dataloader_idx)
                if ran:
                    return result
            return super().predict_step(batch, batch_idx, dataloader_idx)  # pyright: ignore[reportAttributeAccessIssue]

        @classmethod
        def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
            """Register the Lightning datamodule and module compositions."""
            adapter_ctx_registry.register(
                Adapter.lightning,
                component_key="datamodule",
                adapter_combination=(Adapter.lightning,),
                composition_classes=(ITDataModule, LightningDataModule),
                description="lightning adapter to be used with lightning",
            )
            adapter_ctx_registry.register(
                Adapter.lightning,
                component_key="module",
                adapter_combination=(Adapter.lightning,),
                composition_classes=(LightningAdapter, BaseITModule, LightningModule),
                description="lightning adapter to be used with lightning",
            )

    from lightning.pytorch.callbacks import Callback

    class AnalysisStoreCallback(Callback):
        """Write an analysis composition's ``AnalysisStore`` from a ``Trainer.predict`` run.

        ``Trainer.predict`` runs an analysis composition's configured step on each batch. With this callback in the
        Trainer's callbacks, the rows those steps yield are also written as the analysis config's ``AnalysisStore``,
        as ``AnalysisRunner.run_analysis`` writes them: the same features and format from the op's output schema, the
        same ``output_store.save_dir``, ``output_store.dataset`` set to the result, the ``on_analysis_start``,
        ``on_analysis_epoch_end`` and ``on_analysis_end`` hooks around the run, and the same run-scoped op state. The
        metrics a step records with ``log_analysis`` are averaged per epoch and sent to the Trainer's loggers.

        Rows are streamed to disk as each batch's step yields them, so the run's host memory follows
        ``writer_batch_size`` rather than its length. ``Trainer.predict`` still keeps every batch's outputs to return
        them unless it is called with ``return_predictions=False``, which a long run should pass.

        Scope: predict in one process. A module without an analysis config and op, a multi-process predict, and a run
        that yields no rows are each refused by name. A run that raises leaves no partial store behind.

        Args:
            writer_batch_size: How many rows the writer buffers before each flush to disk. ``None`` (default) keeps the
                ``datasets`` default, as ``AnalysisRunnerCfg.writer_batch_size`` does for the core loop.
        """

        def __init__(self, writer_batch_size: int | None = None) -> None:
            super().__init__()
            if writer_batch_size is not None and writer_batch_size < 1:
                raise MisconfigurationException(
                    f"writer_batch_size must be a positive number of rows or None, got {writer_batch_size}"
                )
            self.writer_batch_size = writer_batch_size

        def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
            """Check the run can build a store, run the analysis start hooks and open the row writer."""
            analysis_cfg = getattr(getattr(pl_module, "it_cfg", None), "analysis_cfg", None)
            if analysis_cfg is None or getattr(analysis_cfg, "op", None) is None:
                raise MisconfigurationException(
                    "AnalysisStoreCallback writes the AnalysisStore of an analysis composition, but this module has no"
                    " analysis config with an op. Set `analysis_cfg` on the module, or drop the callback."
                )
            if trainer.world_size > 1:
                raise MisconfigurationException(
                    f"AnalysisStoreCallback builds the AnalysisStore in one process, but this Trainer runs"
                    f" {trainer.world_size} processes, whose rows the store would miss. Run the analysis predict on"
                    " one device."
                )
            from interpretune.base.call import _call_itmodule_hook
            from interpretune.runners.analysis import AnalysisRowWriter, dataset_features_and_format

            pl_module._ensure_analysis_setup()  # runs before the module's own predict-start hook
            # the run boundary of declared op state, as `AnalysisRunner` owns it: a second predict starts clean
            analysis_cfg.reset_op_state()
            _call_itmodule_hook(pl_module, hook_name="on_analysis_start", hook_msg="Running analysis start hooks")
            pl_module.reset_analysis_metrics()
            features, it_format_kwargs, _ = dataset_features_and_format(pl_module, {})
            pl_module._it_analysis_row_writer = AnalysisRowWriter(
                features,
                cache_dir=analysis_cfg.output_store.cache_dir,
                it_format_kwargs=it_format_kwargs,
                writer_batch_size=self.writer_batch_size,
            )

        def on_predict_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            """Run the analysis epoch end hook, report the epoch's ``log_analysis`` metrics to the Trainer's
            loggers, then clear the op state of ops that reset at an epoch boundary.

            The metrics go to each logger directly: Lightning refuses ``self.log`` in every predict hook, and a
            logger's ``log_metrics`` is the route it leaves open. They are logged as ``analysis/<name>``, with the
            analysis epoch as the step.
            """
            from interpretune.base.call import _call_itmodule_hook

            _call_itmodule_hook(
                pl_module, hook_name="on_analysis_epoch_end", hook_msg="Running analysis epoch end hooks"
            )
            metrics = pl_module.flush_analysis_metrics()
            if metrics:
                epoch = len(pl_module.analysis_metrics) - 1
                for logger in trainer.loggers:
                    logger.log_metrics({f"analysis/{name}": value for name, value in metrics.items()}, step=epoch)
                    logger.save()
            pl_module.it_cfg.analysis_cfg.reset_op_state(epoch_boundary=True)

        def on_predict_end(self, trainer: Any, pl_module: Any) -> None:
            """Save the written rows as the store, then run the analysis end hooks."""
            from pathlib import Path

            from interpretune.base.call import _call_itmodule_hook

            writer, pl_module._it_analysis_row_writer = pl_module._it_analysis_row_writer, None
            try:
                try:
                    if len(writer) == 0:
                        raise MisconfigurationException(
                            "Trainer.predict produced no analysis rows, so there is no AnalysisStore to write (check"
                            " limit_predict_batches and the predict dataloader)."
                        )
                    dataset = writer.finalize()
                except BaseException:
                    writer.abort()
                    raise
                output_store = pl_module.it_cfg.analysis_cfg.output_store
                dataset.save_to_disk(Path(output_store.save_dir))
                output_store.dataset = dataset
            finally:
                try:
                    _call_itmodule_hook(pl_module, hook_name="on_analysis_end", hook_msg="Running analysis end hooks")
                finally:
                    pl_module.it_cfg.analysis_cfg.finalize_op_state()

        def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
            """Drop the partial rows of a predict run that raised, and release its op state.

            The grad state the start hook set needs nothing here: Lightning's loop runs inside its own grad context,
            which has already restored the caller's state by the time this is called.
            """
            writer = getattr(pl_module, "_it_analysis_row_writer", None)
            if writer is not None:
                pl_module._it_analysis_row_writer = None
                writer.abort()
                pl_module.it_cfg.analysis_cfg.finalize_op_state()

else:
    LightningDataModule = object
    LightningModule = object
    LightningAdapter = object  # pyright: ignore[reportAssignmentType]
    AnalysisStoreCallback = object  # pyright: ignore[reportAssignmentType]
