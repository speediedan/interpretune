# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Phased fine-tuning on the core training loop: finetuning-scheduler's ``TuningScheduler`` driven at the loop's
boundaries (#762).

:class:`FTSCoreCallback` is a core-loop callback (``SessionRunnerCfg.callbacks``). It builds the tuning scheduler over
the session's module once training starts, so the tuning scheduler's view is the module's attached ``tunable_view``
(#758, #759) and its physical names are the module's own. The callback places the tuning scheduler's calls; phase hooks,
payloads, transition policies and gates (veto included), RNG forking and checkpoint policies stay
finetuning-scheduler's. Lightning compositions keep using the ``FinetuningScheduler`` callback.

The tuning scheduler builds the optimizer (over phase 0's parameters, growing it at each phase) and the lr schedulers,
so the callback installs them as the module's: the loop reads the module's optimizer at every step, and the module's own
lr schedulers are dropped so that nothing is stepped twice. Step-interval lr schedulers are stepped without metrics, so
a metric-stepped scheduler (``ReduceLROnPlateau``) must use the epoch interval, where it receives the epoch's metrics.
Those are the core loop's ``log()`` averages, in which a training and a validation value logged under one name are
averaged together: a transition or checkpoint policy monitoring that name sees the blend.

Resume a run through the tuning scheduler (``tuning_scheduler.save`` and ``tuning_scheduler.load``): the core loop's
``checkpoint_state`` captures the module and the tuning scheduler's optimizer but none of the tuning scheduler's own
state (its phase, pending decision, gate records and RNG forks), so loading such a checkpoint into a callback is refused
rather than resuming half a run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from interpretune.utils import MisconfigurationException, instantiate_class, rank_zero_warn

# the step configuration keys finetuning-scheduler's `lr_scheduler_cfg` accepts, out of a module's `pl_lrs_cfg`
_LR_STEP_KEYS = ("interval", "frequency", "monitor")


class FTSCoreCallback:
    """Drive finetuning-scheduler's ``TuningScheduler`` from the core training loop.

    Args:
        schedule: The fine-tuning schedule: a mapping, a path to a schedule YAML file, or ``None`` for the default
            schedule.
        tuning_scheduler_kwargs: Further ``TuningScheduler`` keyword arguments (transition policies, a checkpoint
            policy, ``max_depth`` and so on). Without ``optimizer``, the tuning scheduler's optimizer factory is the
            module's ``optimizer_init``; without ``lr_scheduler``, the module's ``lr_scheduler_init`` (stepped as its
            ``pl_lrs_cfg`` says), when it has one.
        tuning_scheduler_factory: Builds the tuning scheduler from the module instead, for anything
            ``tuning_scheduler_kwargs`` cannot express. Its tuning scheduler must be rooted at the module it is given,
            which is checked once it is set up.

    Attributes:
        tuning_scheduler: The tuning scheduler, once training has started.
    """

    def __init__(
        self,
        schedule: Any = None,
        *,
        tuning_scheduler_kwargs: Mapping[str, Any] | None = None,
        tuning_scheduler_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if tuning_scheduler_factory is not None and (schedule is not None or tuning_scheduler_kwargs):
            raise MisconfigurationException(
                "Pass either `tuning_scheduler_factory` or `schedule`/`tuning_scheduler_kwargs` to FTSCoreCallback,"
                " not both."
            )
        self.schedule = schedule
        self.tuning_scheduler_kwargs = dict(tuning_scheduler_kwargs or {})
        self.tuning_scheduler_factory = tuning_scheduler_factory
        self.tuning_scheduler: Any | None = None

    def build_tuning_scheduler(self, module: Any) -> Any:
        """The tuning scheduler over ``module``, from the factory or the schedule and keyword arguments."""
        if self.tuning_scheduler_factory is not None:
            return self.tuning_scheduler_factory(module)
        from finetuning_scheduler.core.tuning_scheduler import TuningScheduler

        kwargs = dict(self.tuning_scheduler_kwargs)
        it_cfg = getattr(module, "it_cfg", None)
        if "optimizer" not in kwargs:
            optimizer_init = getattr(it_cfg, "optimizer_init", None)
            if not optimizer_init:
                raise MisconfigurationException(
                    "FTSCoreCallback needs an optimizer: set the module's `optimizer_init`, or pass `optimizer` in"
                    " `tuning_scheduler_kwargs`."
                )
            kwargs["optimizer"] = partial(instantiate_class, optimizer_init)
        if "lr_scheduler" not in kwargs and (lr_scheduler_init := getattr(it_cfg, "lr_scheduler_init", None)):
            kwargs["lr_scheduler"] = partial(instantiate_class, lr_scheduler_init)
            lrs_cfg = getattr(it_cfg, "pl_lrs_cfg", None) or {}
            kwargs.setdefault("lr_scheduler_cfg", {k: lrs_cfg[k] for k in _LR_STEP_KEYS if k in lrs_cfg})
        return TuningScheduler(module, self.schedule, **kwargs)

    def _install(self, ctx: Any) -> None:
        """Make the tuning scheduler's optimizer the module's (it can change at a phase transition), with no
        module-owned lr schedulers left for the loop to step."""
        assert self.tuning_scheduler is not None
        optimizers, lr_scheduler_configs = [self.tuning_scheduler.optimizer], []
        state = getattr(ctx.module, "_it_state", None)
        if state is not None:
            state._it_optimizers, state._it_lr_scheduler_configs = optimizers, lr_scheduler_configs
        else:
            ctx.module.optimizers, ctx.module.lr_scheduler_configs = optimizers, lr_scheduler_configs

    def state_dict(self) -> dict[str, Any]:
        """A marker only: the tuning scheduler's state is saved with ``tuning_scheduler.save``, not by the core loop's
        ``checkpoint_state``."""
        return {"resume_through_tuning_scheduler": True}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Refuse a core-loop checkpoint, which lacks the tuning scheduler's state."""
        raise MisconfigurationException(
            "FTSCoreCallback cannot resume from the core loop's `checkpoint_state`, which holds none of the tuning"
            " scheduler's state (phase, pending decision, gate records, RNG forks); resume through the tuning"
            " scheduler (`tuning_scheduler.save` and `tuning_scheduler.load`) instead."
        )

    def _sync_stop(self, ctx: Any) -> None:
        assert self.tuning_scheduler is not None
        if self.tuning_scheduler.should_stop:
            ctx.should_stop = True

    def on_train_start(self, ctx: Any) -> None:
        """Build and set up the tuning scheduler, then install its optimizer."""
        self.tuning_scheduler = tuning_scheduler = self.build_tuning_scheduler(ctx.module)
        tuning_scheduler.setup()
        if tuning_scheduler.view.root is not ctx.module:
            raise MisconfigurationException(
                f"FTSCoreCallback's tuning scheduler is rooted at a {type(tuning_scheduler.view.root).__name__}, not"
                f" at the session's module ({type(ctx.module).__name__}), so the loop would step another module's"
                " parameters: build the tuning scheduler over the module the factory is given."
            )
        self._install(ctx)
        final = tuning_scheduler.schedule[max(tuning_scheduler.schedule)]
        final_epochs = final.get("max_transition_epoch", -1)
        if ctx.max_epochs >= 0 and final_epochs is not None and final_epochs > ctx.max_epochs:
            rank_zero_warn(
                f"The schedule's final phase sets `max_transition_epoch` {final_epochs}, beyond the core loop's"
                f" `max_epochs` {ctx.max_epochs}. The core loop keeps its own budget (the Lightning callback would"
                " extend the run): raise `max_epochs` to run the whole schedule."
            )

    def on_train_epoch_start(self, ctx: Any) -> None:
        """Begin the tuning scheduler's epoch, advancing a phase when one is due."""
        assert self.tuning_scheduler is not None
        self.tuning_scheduler.epoch_start()
        self._install(ctx)
        self._sync_stop(ctx)

    def on_train_step_end(self, ctx: Any, outputs: Any) -> None:
        """Step the tuning scheduler's step-interval lr schedulers (without metrics) and evaluate step-granular
        transitions."""
        assert self.tuning_scheduler is not None
        self.tuning_scheduler.step_lr_schedulers("step")
        self.tuning_scheduler.step_end()
        self._install(ctx)
        self._sync_stop(ctx)

    def on_train_epoch_end(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        """Offer the epoch's metrics to the tuning scheduler (transitions, checkpoint policy), then step its epoch-
        interval lr schedulers."""
        assert self.tuning_scheduler is not None
        self.tuning_scheduler.epoch_end(dict(metrics))
        self.tuning_scheduler.step_lr_schedulers("epoch", dict(metrics))
        self._install(ctx)
        self._sync_stop(ctx)

    def on_train_end(self, ctx: Any) -> None:
        """End training: the final restore and gates, when configured."""
        if self.tuning_scheduler is not None:
            self.tuning_scheduler.train_end()
