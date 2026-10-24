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
"""Phased fine-tuning on the core training loop: finetuning-scheduler's ``FTSController`` driven at the loop's
boundaries (#762).

:class:`FTSCoreCallback` is a core-loop callback (``SessionRunnerCfg.callbacks``). It builds the controller over the
session's module once training starts, so the controller's view is the module's attached ``fts_tunable_view`` (#758,
#759) and its physical names are the module's own. The callback places the controller's calls; phase hooks, payloads,
transition policies and gates (veto included), RNG forking and checkpoint policies stay finetuning-scheduler's.
Lightning compositions keep using the ``FinetuningScheduler`` callback.

The controller builds the optimizer (over phase 0's parameters, growing it at each phase) and the lr schedulers, so the
callback installs them as the module's: the loop reads the module's optimizer at every step, and the module's own lr
schedulers are dropped so that nothing is stepped twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from interpretune.utils import MisconfigurationException, instantiate_class, rank_zero_warn

# the step configuration keys finetuning-scheduler's `lr_scheduler_cfg` accepts, out of a module's `pl_lrs_cfg`
_LR_STEP_KEYS = ("interval", "frequency", "monitor")


class FTSCoreCallback:
    """Drive finetuning-scheduler's ``FTSController`` from the core training loop.

    Args:
        schedule: The fine-tuning schedule: a mapping, a path to a schedule YAML file, or ``None`` for the default
            schedule.
        controller_kwargs: Further ``FTSController`` keyword arguments (transition policies, a checkpoint policy,
            ``max_depth`` and so on). Without ``optimizer``, the controller's optimizer factory is the module's
            ``optimizer_init``; without ``lr_scheduler``, the module's ``lr_scheduler_init`` (stepped as its
            ``pl_lrs_cfg`` says), when it has one.
        controller_factory: Builds the controller from the module instead, for anything ``controller_kwargs`` cannot
            express. It must be rooted at the module it is given.

    Attributes:
        controller: The controller, once training has started.
    """

    def __init__(
        self,
        schedule: Any = None,
        *,
        controller_kwargs: Mapping[str, Any] | None = None,
        controller_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        if controller_factory is not None and (schedule is not None or controller_kwargs):
            raise MisconfigurationException(
                "Pass either `controller_factory` or `schedule`/`controller_kwargs` to FTSCoreCallback, not both."
            )
        self.schedule = schedule
        self.controller_kwargs = dict(controller_kwargs or {})
        self.controller_factory = controller_factory
        self.controller: Any | None = None

    def build_controller(self, module: Any) -> Any:
        """The controller over ``module``, from the factory or the schedule and keyword arguments."""
        if self.controller_factory is not None:
            return self.controller_factory(module)
        from finetuning_scheduler.core.controller import FTSController

        kwargs = dict(self.controller_kwargs)
        it_cfg = getattr(module, "it_cfg", None)
        if "optimizer" not in kwargs:
            optimizer_init = getattr(it_cfg, "optimizer_init", None)
            if not optimizer_init:
                raise MisconfigurationException(
                    "FTSCoreCallback needs an optimizer: set the module's `optimizer_init`, or pass `optimizer` in"
                    " `controller_kwargs`."
                )
            kwargs["optimizer"] = partial(instantiate_class, optimizer_init)
        if "lr_scheduler" not in kwargs and (lr_scheduler_init := getattr(it_cfg, "lr_scheduler_init", None)):
            kwargs["lr_scheduler"] = partial(instantiate_class, lr_scheduler_init)
            lrs_cfg = getattr(it_cfg, "pl_lrs_cfg", None) or {}
            kwargs.setdefault("lr_scheduler_cfg", {k: lrs_cfg[k] for k in _LR_STEP_KEYS if k in lrs_cfg})
        return FTSController(module, self.schedule, **kwargs)

    def _install(self, ctx: Any) -> None:
        """Make the controller's optimizer the module's (it can change at a phase transition), with no module-owned
        lr schedulers left for the loop to step."""
        assert self.controller is not None
        optimizers, lr_scheduler_configs = [self.controller.optimizer], []
        state = getattr(ctx.module, "_it_state", None)
        if state is not None:
            state._it_optimizers, state._it_lr_scheduler_configs = optimizers, lr_scheduler_configs
        else:
            ctx.module.optimizers, ctx.module.lr_scheduler_configs = optimizers, lr_scheduler_configs

    def _sync_stop(self, ctx: Any) -> None:
        assert self.controller is not None
        if self.controller.should_stop:
            ctx.should_stop = True

    def on_train_start(self, ctx: Any) -> None:
        """Build and set up the controller, then install its optimizer."""
        self.controller = controller = self.build_controller(ctx.module)
        controller.setup()
        self._install(ctx)
        final = controller.schedule[max(controller.schedule)]
        final_epochs = final.get("max_transition_epoch", -1)
        if ctx.max_epochs >= 0 and final_epochs is not None and final_epochs > ctx.max_epochs:
            rank_zero_warn(
                f"The schedule's final phase sets `max_transition_epoch` {final_epochs}, beyond the core loop's"
                f" `max_epochs` {ctx.max_epochs}. The core loop keeps its own budget (the Lightning callback would"
                " extend the run): raise `max_epochs` to run the whole schedule."
            )

    def on_train_epoch_start(self, ctx: Any) -> None:
        """Begin the controller's epoch, advancing a phase when one is due."""
        assert self.controller is not None
        self.controller.epoch_start()
        self._install(ctx)
        self._sync_stop(ctx)

    def on_train_step_end(self, ctx: Any, outputs: Any) -> None:
        """Step the controller's step-interval lr schedulers and evaluate step-granular transitions."""
        assert self.controller is not None
        self.controller.step_lr_schedulers("step")
        self.controller.step_end()
        self._install(ctx)
        self._sync_stop(ctx)

    def on_train_epoch_end(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        """Offer the epoch's metrics to the controller (transitions, checkpoint policy), then step its epoch-
        interval lr schedulers."""
        assert self.controller is not None
        self.controller.epoch_end(dict(metrics))
        self.controller.step_lr_schedulers("epoch", dict(metrics))
        self._install(ctx)
        self._sync_stop(ctx)

    def on_train_end(self, ctx: Any) -> None:
        """End training: the final restore and gates, when configured."""
        if self.controller is not None:
            self.controller.train_end()
