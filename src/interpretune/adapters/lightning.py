from interpretune.base import ITDataModule, BaseITModule
from interpretune.utils import _LIGHTNING_AVAILABLE
from interpretune.protocol import Adapter
from interpretune.adapters import CompositionRegistry


if _LIGHTNING_AVAILABLE:
    from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
    from lightning.pytorch import LightningDataModule, LightningModule

    class LightningAdapter:
        """Adapter composing interpretune modules with Lightning's ``LightningModule``/``DataModule``."""

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

        def on_train_start(self) -> None:
            """Force the model into training mode before training starts, then defer to Lightning.

            Lightning normally handles this, but a run that skips sanity checking can reach training with the model
            still in eval mode; setting it explicitly closes that edge case.
            """
            # ensure model is in training mode (e.g. needed for some edge cases w/ skipped sanity checking)
            self.model.train()  # type: ignore[attr-defined]  # provided by LightningModule when mixed in
            return super().on_train_start()  # type: ignore[misc]  # LightningModule method when mixed in

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

                init_analysis_cfgs(self, analysis_cfg)  # type: ignore[arg-type]  # a mixin; the composed module satisfies it
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
            from collections.abc import Iterator

            result = step(batch, batch_idx, dataloader_idx)
            return True, (list(result) if isinstance(result, Iterator) else result)

        def on_predict_start(self) -> None:
            """Ensure analysis setup ran before a Trainer predict loop, then defer to Lightning."""
            self._ensure_analysis_setup()
            return super().on_predict_start()  # type: ignore[misc]  # LightningModule method when mixed in

        def on_test_start(self) -> None:
            """Ensure analysis setup ran before a Trainer test loop, then defer to Lightning."""
            self._ensure_analysis_setup()
            return super().on_test_start()  # type: ignore[misc]  # LightningModule method when mixed in

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
            return super().predict_step(batch, batch_idx, dataloader_idx)  # type: ignore[misc]

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
else:
    LightningDataModule = object
    LightningModule = object
    LightningAdapter = object  # type: ignore[assignment]
