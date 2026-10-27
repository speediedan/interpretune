from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Any, Iterator, Mapping, Protocol, Sequence, TYPE_CHECKING, runtime_checkable
import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partialmethod
from pathlib import Path
import torch

from interpretune.config import SessionRunnerCfg
from interpretune.base import _call_itmodule_hook, it_init, it_session_end, ITDataModule
from interpretune.protocol import Optimizable, CorePhases
from interpretune.utils import MisconfigurationException, rank_zero_warn

if TYPE_CHECKING:
    from interpretune.adapters import ITModule


log = logging.getLogger(__name__)


@contextmanager
def profiling(module: ITModule) -> Iterator[torch.profiler.profile | None]:
    """Run the enclosed batch loop under the PyTorch profiler when ``it_cfg.profiler_cfg`` selects it.

    Yields the active profiler, which the loop steps once per batch, or ``None`` when profiling is off. Traces go to
    ``pytorch_profiler_cfg.trace_dir``, defaulting to ``<core_log_dir>/pytorch_profiler``.
    """
    profiler_cfg = module.it_cfg.profiler_cfg
    if profiler_cfg.which != "pytorch":
        yield None
        return
    default_trace_dir = Path(module.core_log_dir) / "pytorch_profiler"  # mixin attr
    with torch.profiler.profile(**profiler_cfg.torch_profiler_kwargs(default_trace_dir)) as prof:
        yield prof


@dataclass
class CoreLoopContext:
    """The state the core training loop shares with its callbacks.

    Attributes:
        module: The module being trained.
        epoch: The current epoch.
        optimizer_step: Optimizer steps taken so far in this run. ``module.global_step`` counts every step the runner
            dispatches (validation and test steps included), so step budgets and step-interval lr schedulers count
            this instead.
        max_epochs: The epoch budget (``-1`` for none).
        max_steps: The optimizer-step budget (``-1`` for none).
        epoch_metrics: The current epoch's averaged ``log()``/``log_dict()`` values, set after validation.
        epoch_lr_pending: Whether the current epoch's epoch-interval lr scheduler step is still to come: set while
            the callbacks' ``on_train_epoch_end`` runs, so that a checkpoint taken there resumes with that step (see
            ``resume_from``).
        should_stop: Set by a callback to end training after the current step (or, if set at an epoch boundary, before
            the next epoch).
    """

    module: Any
    epoch: int = 0
    optimizer_step: int = 0
    max_epochs: int = -1
    max_steps: int = -1
    epoch_metrics: dict[str, float] = field(default_factory=dict)
    epoch_lr_pending: bool = False
    should_stop: bool = False
    callbacks: Sequence[Any] = ()

    @property
    def optimizer(self) -> Optimizable | None:
        """The module's first optimizer, read when needed rather than once, so that a callback may replace it."""
        optimizers = self.module.optimizers
        return optimizers[0] if optimizers else None

    @property
    def steps_exhausted(self) -> bool:
        """Whether the optimizer-step budget is spent."""
        return self.max_steps >= 0 and self.optimizer_step >= self.max_steps

    def checkpoint_state(self) -> dict[str, Any]:
        """This run's checkpoint state (see :func:`checkpoint_state`)."""
        return checkpoint_state(self.module, self)

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore state from :meth:`checkpoint_state` (see :func:`load_checkpoint_state`)."""
        load_checkpoint_state(self.module, state, self)


@runtime_checkable
class CoreCallback(Protocol):
    """A callback for the core training loop. Every method is optional: the loop calls only those a callback
    defines.

    - ``on_train_start(ctx)`` and ``on_train_end(ctx)``: once per run.
    - ``on_train_epoch_start(ctx)``: before each epoch's first batch.
    - ``on_train_step_end(ctx, outputs)``: after each optimizer step (and any step-interval lr scheduler step).
    - ``on_validation_end(ctx, metrics)``: after each epoch's validation, with the epoch's averaged metrics.
    - ``on_train_epoch_end(ctx, metrics)``: after the module's own ``on_train_epoch_end`` hook, before epoch-interval
      lr schedulers step.
    - ``state_dict()`` and ``load_state_dict(state)``: included in :func:`checkpoint_state`. A callback's state is
      loaded before the module's and the optimizers', after every callback's ``on_train_start`` when the loop
      resumes (``resume_from``).
    """


def _callbacks(ctx: CoreLoopContext, hook: str, *args: Any) -> None:
    for callback in ctx.callbacks:
        fn = getattr(callback, hook, None)
        if callable(fn):
            fn(ctx, *args)


def epoch_metrics(module: Any) -> dict[str, float]:
    """Average and clear the values ``log()``/``log_dict()`` accumulated on ``module`` since the last call."""
    logged = getattr(module, "_logged_metrics", None)
    if not isinstance(logged, dict) or not logged:
        return {}
    metrics = {k: sum(v) / len(v) for k, v in logged.items() if v}
    logged.clear()
    return metrics


def _print_metrics(label: str, metrics: Mapping[str, float]) -> None:
    metrics_str = ", ".join(f"'{k}': {v:.4f}" for k, v in metrics.items())
    print(f"\n{label}: {{{metrics_str}}}")


def step_lr_schedulers(module: Any, interval: str, counter: int, metrics: Mapping[str, Any] | None = None) -> None:
    """Step each of ``module``'s lr schedulers configured for ``interval`` (``"step"`` or ``"epoch"``) whose
    ``frequency`` divides ``counter``.

    A ``ReduceLROnPlateau`` scheduler is stepped with its ``monitor`` metric.

    Raises:
        MisconfigurationException: If a ``ReduceLROnPlateau`` scheduler names no ``monitor``, or its metric is missing
            from ``metrics`` (unless its config sets ``strict=False``, which skips the step with a warning).
    """
    configs = getattr(module, "lr_scheduler_configs", None) or []
    for config in configs:
        if getattr(config, "interval", "epoch") != interval or counter % max(getattr(config, "frequency", 1), 1):
            continue
        scheduler = config.scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor = getattr(config, "monitor", None)
            if monitor is None:
                raise MisconfigurationException(
                    "A `ReduceLROnPlateau` lr scheduler needs a `monitor` metric in its config (e.g. `pl_lrs_cfg`)."
                )
            if monitor not in (metrics or {}):
                if getattr(config, "strict", True):
                    raise MisconfigurationException(
                        f"The `ReduceLROnPlateau` lr scheduler monitors '{monitor}', which was not logged this"
                        f" {interval}. Logged: {sorted(metrics or {})}."
                    )
                rank_zero_warn(f"Skipping a `ReduceLROnPlateau` step: '{monitor}' was not logged this {interval}.")
                continue
            scheduler.step(metrics[monitor])  # type: ignore[index]  # membership checked above
        else:
            scheduler.step()


def core_train_loop(
    module: ITModule,
    datamodule: ITDataModule,
    limit_train_batches: int,
    limit_val_batches: int,
    max_epochs: int,
    *args,
    max_steps: int = -1,
    callbacks: Sequence[Any] = (),
    validate_with_grad: bool = False,
    resume_from: str | Path | Mapping[str, Any] | None = None,
    **kwargs,
) -> CoreLoopContext:
    """Run the core (framework-free) training loop, with optional interleaved validation.

    Each epoch trains (stopping early once ``max_steps`` optimizer steps are taken or a callback sets
    ``ctx.should_stop``), then validates, then averages and reports the epoch's ``log()``/``log_dict()`` values,
    then runs the epoch-end hooks and steps epoch-interval lr schedulers. Step-interval lr schedulers step after each
    optimizer step.

    Args:
        module: The module to train.
        datamodule: Provides ``train_dataloader`` and, optionally, ``val_dataloader``.
        limit_train_batches: Training batches per epoch (``-1`` for all).
        limit_val_batches: Validation batches per epoch (``-1`` for all).
        max_epochs: Epochs to run.
        max_steps: Optimizer steps to run at most (``-1`` for no limit). The epoch in which the budget is spent still
            validates and runs its epoch-end hooks.
        callbacks: Objects with any of :class:`CoreCallback`'s methods.
        validate_with_grad: Validate with gradients enabled (for objectives that call ``torch.autograd.grad``)
            instead of under ``torch.inference_mode()``.
        resume_from: A :func:`checkpoint_state` (or a path to one saved with ``torch.save``) taken at the end of an
            epoch, in a callback's ``on_train_epoch_end`` (after ``FTSCoreCallback`` in ``callbacks``, when it is
            used). It is loaded once every callback's ``on_train_start`` has run, and the loop continues at the epoch
            after the checkpoint's.

    Returns:
        The loop's final context.
    """
    assert hasattr(datamodule, "train_dataloader"), "Datamodule is expected to have a train dataloader"
    train_dataloader = datamodule.train_dataloader()  # type: ignore[attr-defined]
    val_dataloader = datamodule.val_dataloader() if hasattr(datamodule, "val_dataloader") else None  # type: ignore[attr-defined]  # duck typing
    assert module.optimizers, "Module has no optimizers configured"
    ctx = CoreLoopContext(module=module, max_epochs=max_epochs, max_steps=max_steps, callbacks=tuple(callbacks))
    val_grad_ctx = torch.enable_grad if validate_with_grad else torch.inference_mode
    _callbacks(ctx, "on_train_start")
    start_epoch = _resume(ctx, resume_from) if resume_from is not None else 0
    with profiling(module) as prof:
        for epoch_idx in range(start_epoch, max_epochs):
            if ctx.should_stop or ctx.steps_exhausted:
                break
            assert module.model is not None, "Model must be initialized before training"
            module.model.train()
            module.current_epoch = ctx.epoch = epoch_idx
            _call_itmodule_hook(
                module, hook_name="on_train_epoch_start", hook_msg="Running train epoch start hooks", optional=True
            )
            _callbacks(ctx, "on_train_epoch_start")
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= limit_train_batches >= 0 or ctx.should_stop or ctx.steps_exhausted:
                    break
                output = run_step(
                    step_fn="training_step", module=module, batch=batch, batch_idx=batch_idx, optimizer=ctx.optimizer
                )
                ctx.optimizer_step += 1
                step_lr_schedulers(module, "step", ctx.optimizer_step)
                _callbacks(ctx, "on_train_step_end", output)
                if prof is not None:
                    prof.step()
            if val_dataloader is not None:
                module.model.eval()
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= limit_val_batches >= 0:
                        break
                    with val_grad_ctx():
                        run_step(step_fn="validation_step", module=module, batch=batch, batch_idx=batch_idx)
                    if prof is not None:
                        prof.step()
            ctx.epoch_metrics = epoch_metrics(module)
            if ctx.epoch_metrics:
                _print_metrics(f"Epoch {epoch_idx} end", ctx.epoch_metrics)
            _callbacks(ctx, "on_validation_end", ctx.epoch_metrics)
            module.model.train()
            _call_itmodule_hook(
                module, hook_name="on_train_epoch_end", hook_msg="Running train epoch end hooks", optional=True
            )
            ctx.epoch_lr_pending = True
            _callbacks(ctx, "on_train_epoch_end", ctx.epoch_metrics)
            step_lr_schedulers(module, "epoch", epoch_idx + 1, ctx.epoch_metrics)
            ctx.epoch_lr_pending = False
    _callbacks(ctx, "on_train_end")
    return ctx


def _resume(ctx: CoreLoopContext, resume_from: str | Path | Mapping[str, Any]) -> int:
    """Load ``resume_from`` into ``ctx``, completing the epoch it was taken in, and return the epoch to start
    at."""
    checkpoint = resume_from if isinstance(resume_from, Mapping) else torch.load(resume_from, weights_only=False)
    load_checkpoint_state(ctx.module, checkpoint, ctx)
    loop = checkpoint["loop"]
    if loop.get("epoch_lr_pending"):  # taken in `on_train_epoch_end`, before the epoch-interval lr schedulers stepped
        step_lr_schedulers(ctx.module, "epoch", loop["epoch"] + 1, loop.get("epoch_metrics"))
    return loop["epoch"] + 1


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:  # numpy is optional
        pass
    else:
        state["numpy"] = np.random.get_state()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


def checkpoint_state(module: Any, ctx: CoreLoopContext | None = None) -> dict[str, Any]:
    """The state needed to resume a core-loop run: the module's ``state_dict``, every optimizer's and lr scheduler's
    state, the RNG states (Python, torch, CUDA and, when installed, numpy), the loop counters and each callback's
    ``state_dict``.

    The module's ``on_save_checkpoint(checkpoint)`` hook, if defined, may add to the result. Like ``state_dict()``,
    the result references live tensors: serialize it (``torch.save``) or deep-copy it to keep it.
    """
    checkpoint: dict[str, Any] = {
        "module": module.state_dict(),
        "optimizers": [opt.state_dict() for opt in (module.optimizers or [])],
        "lr_schedulers": [cfg.scheduler.state_dict() for cfg in (getattr(module, "lr_scheduler_configs", None) or [])],
        "rng": _rng_state(),
        "loop": {"epoch": module.current_epoch, "global_step": module.global_step},
    }
    if ctx is not None:
        checkpoint["loop"].update(
            optimizer_step=ctx.optimizer_step, epoch_lr_pending=ctx.epoch_lr_pending, epoch_metrics=ctx.epoch_metrics
        )
        checkpoint["callbacks"] = {
            f"{i}:{type(cb).__name__}": cb.state_dict()
            for i, cb in enumerate(ctx.callbacks)
            if hasattr(cb, "state_dict")
        }
    _call_itmodule_hook(
        module,
        hook_name="on_save_checkpoint",
        hook_msg="Running save checkpoint hook",
        optional=True,
        checkpoint=checkpoint,
    )
    return checkpoint


def load_checkpoint_state(module: Any, checkpoint: Mapping[str, Any], ctx: CoreLoopContext | None = None) -> None:
    """Restore what :func:`checkpoint_state` saved. The module's ``on_load_checkpoint(checkpoint)`` hook, if
    defined, runs first, then the callbacks' ``load_state_dict`` (a callback may rebuild the module's optimizers,
    as ``FTSCoreCallback`` does), then the module's, optimizers' and lr schedulers' state.

    Raises:
        MisconfigurationException: If the checkpoint's optimizer or lr scheduler count differs from the module's.
    """
    _call_itmodule_hook(
        module,
        hook_name="on_load_checkpoint",
        hook_msg="Running load checkpoint hook",
        optional=True,
        checkpoint=checkpoint,
    )
    if ctx is not None:
        for i, cb in enumerate(ctx.callbacks):
            key = f"{i}:{type(cb).__name__}"
            if hasattr(cb, "load_state_dict") and key in checkpoint.get("callbacks", {}):
                cb.load_state_dict(checkpoint["callbacks"][key])
    module.load_state_dict(checkpoint["module"])
    optimizers = module.optimizers or []
    schedulers = [cfg.scheduler for cfg in (getattr(module, "lr_scheduler_configs", None) or [])]
    for kind, have, saved in (
        ("optimizers", optimizers, checkpoint["optimizers"]),
        ("lr schedulers", schedulers, checkpoint["lr_schedulers"]),
    ):
        if len(have) != len(saved):
            raise MisconfigurationException(f"The checkpoint holds {len(saved)} {kind} but the module has {len(have)}.")
        for obj, state in zip(have, saved):
            obj.load_state_dict(state)
    _set_rng_state(checkpoint["rng"])
    module.current_epoch = checkpoint["loop"]["epoch"]
    module.global_step = checkpoint["loop"]["global_step"]
    if ctx is not None:
        ctx.optimizer_step = checkpoint["loop"].get("optimizer_step", 0)


def core_test_loop(module: ITModule, datamodule: ITDataModule, limit_test_batches: int, *args, **kwargs):
    """Run the test loop over the datamodule's test dataloader.

    Raises:
        AssertionError: the datamodule exposes no ``test_dataloader``.
    """
    if not hasattr(datamodule, "test_dataloader"):
        raise AssertionError("Datamodule is expected to have a test dataloader")
    dataloader = datamodule.test_dataloader()  # type: ignore[attr-defined]  # duck typing, checked above
    test_ctx = {}
    module._it_state._current_epoch = 0
    assert module.model is not None, "Model must be initialized before testing"
    module.model.eval()
    with profiling(module) as prof:
        for batch_idx, batch in enumerate(dataloader):
            with torch.inference_mode():
                if batch_idx >= limit_test_batches >= 0:
                    break
                run_step(step_fn="test_step", module=module, batch=batch, batch_idx=batch_idx, **test_ctx)
            if prof is not None:
                prof.step()
    # Print accumulated metrics from log/log_dict calls (core framework epoch-end reporting)
    if metrics := epoch_metrics(module):
        _print_metrics("Test epoch end", metrics)
    _call_itmodule_hook(module, hook_name="on_test_epoch_end", hook_msg="Running test epoch end hooks", optional=True)


def run_step(step_fn, module, batch, batch_idx, optimizer: Optimizable | None = None, as_generator: bool = False):
    """Run one step: move the batch to device, dispatch the step function, optionally optimize.

    ``as_generator`` yields the step output instead of returning it, which is what lets the analysis
    runner stream results rather than accumulate them.
    """
    batch = module.batch_to_device(batch)
    step_func = getattr(module, step_fn)
    if module.global_step == 0 and step_fn == "test_step":
        _call_itmodule_hook(
            module,
            hook_name="on_test_batch_start",
            hook_msg="Running test batch start hook",
            optional=True,
            batch=batch,
            batch_idx=batch_idx,
        )
    elif module.global_step == 0 and step_fn == "training_step":
        _call_itmodule_hook(
            module,
            hook_name="on_train_batch_start",
            hook_msg="Running train batch start hook",
            optional=True,
            batch=batch,
            batch_idx=batch_idx,
        )
    if step_fn == "training_step" and optimizer is not None:
        optimizer.zero_grad()
    if module.dtype == torch.bfloat16:
        with torch.autocast(device_type=module.device.type, dtype=module.dtype):
            output = step_func(batch, batch_idx)
    else:
        output = step_func(batch, batch_idx)
    if step_fn == "training_step" and optimizer is not None:
        _call_itmodule_hook(
            module,
            hook_name="on_train_batch_end",
            hook_msg="Running custom on_train_batch end hook",
            optional=True,
            outputs=output,
            batch=batch,
            batch_idx=batch_idx,
        )
        output.backward()
        optimizer.step()
    module.global_step += 1

    if as_generator:
        # yield from output
        def generator():
            yield from output

        return generator()  # Return a generator object
    else:
        return output


class SessionRunner:
    """A barebones trainer that can be used to orchestrate training when no adapter is specified during ITSession
    composition."""

    def __init__(self, run_cfg: SessionRunnerCfg | dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_cfg = run_cfg if isinstance(run_cfg, SessionRunnerCfg) else SessionRunnerCfg(**run_cfg)
        # Only training and testing commands are supported in SessionRunner
        self.supported_commands = (None, "train", "test")
        self.it_init()

    @property
    def phase(self) -> CorePhases | None:
        """The phase currently executing, or None before a phase starts."""
        return self._current_phase

    @phase.setter
    def phase(self, phase: CorePhases):
        self._current_phase = phase

    def it_init(self):
        """Dispatch the framework-independent session init unless a subclass overrides it."""
        # Unless overridden we dispatch the trainer-independent `it_init`
        assert self.run_cfg.it_session is not None, "Expected ITSession object"
        it_init(**self.run_cfg.it_session)

    def it_session_end(self):
        """Dispatch any phase-specific session end hooks."""
        assert self.run_cfg.it_session is not None, "Expected ITSession object"
        assert self.phase is not None, "Expected phase to be set"
        it_session_end(session_type=self.phase, **self.run_cfg.it_session)

    def _run(self, phase, loop_fn, *args: Any, **kwargs: Any) -> Any | None:
        self.phase = CorePhases[phase]
        phase_artifacts = loop_fn(**self.run_cfg.__dict__)
        self.it_session_end()
        return phase_artifacts

    test = partialmethod(_run, phase="test", loop_fn=core_test_loop)
    train = partialmethod(_run, phase="train", loop_fn=core_train_loop)
