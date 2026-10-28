from __future__ import annotations
from typing import Any, TYPE_CHECKING
from collections.abc import Iterator
import torch

from interpretune.utils import MisconfigurationException, instantiate_class, rank_zero_warn
from interpretune.protocol import OptimizerLRScheduler, STEP_OUTPUT
from interpretune.tuning.views import VIEW_ATTR

if TYPE_CHECKING:
    from interpretune.config import ITConfig, ITState


class BaseITHooks:
    """IT protocol hooks implemented by ``BaseITModule``.

    Every hook here is COOPERATIVE: the base implementations do real work (datamodule wiring,
    optimizer instantiation from ``it_cfg``, session-completion bookkeeping), so an override that
    does not call ``super()`` silently drops that behavior rather than failing loudly.

    Step methods (``training_step``/``validation_step``/``test_step``/``predict_step``) are
    deliberately NOT defined here even though ``ITModuleProtocol`` mandates at least one: a default
    empty implementation would satisfy the structural check while giving the phase no execution
    semantics, hiding exactly the omission the protocol exists to surface. They are documented in
    ``docs/module_contract_and_limitations.md`` and must come from the user module, a mixin, or an
    adapter.
    """

    # if you override these in your module, ensure you cooperatively call super() if you want to retain
    # the relevant BaseITModule hook functionality
    # proper initialization of these variables should be done in the child class
    model: torch.nn.Module
    it_cfg: ITConfig
    session_complete: bool
    _it_state: ITState

    def setup(self, *args, **kwargs) -> None:
        """Attach the datamodule if provided and create the session's output directories.

        The datamodule arrives via kwargs rather than being assumed present: under some adapters it is
        reachable only through the trainer at this point, so setup must tolerate its absence here.
        """
        # for some adapters, datamodule access is not provided in setup and will be accessed via Trainer
        if datamodule := kwargs.get("datamodule", None):
            self._it_state._datamodule = datamodule
        self._make_setup_dirs()  # type: ignore[attr-defined]  # provided by mixin composition

    def optimizer_parameters(self) -> Iterator[torch.nn.Parameter]:
        """The parameters the default optimizer is built over.

        When a trainable finetuning-scheduler view is attached (``tunable_view``, #758), its root's parameters, less
        those whose role finetuning-scheduler never optimizes (``fitted``, ``instrument``). The root of every
        interpretune view is this module, so trainable modules registered beside ``self.model`` (interventions, spliced
        SAEs) are stepped with no adapter extending this (#761). The view is bound through finetuning-scheduler, so an
        attached object that is not a view raises finetuning-scheduler's ``TunableViewError`` naming what it lacks.
        Otherwise (no view, or one refusing the composition) the model's own parameters. An adapter that attaches
        trainable modules outside ``self.model`` (the SAEs ``add_saes_on_init`` splices into a ``TransformerBridge``,
        whose ``parameters()`` delegate to the wrapped HF model only) still extends this cooperatively, so those modules
        are stepped with or without a view.
        """
        view = getattr(self, VIEW_ATTR, None)
        if view is None:
            return self.model.parameters()
        from finetuning_scheduler.tunable_view import as_view
        from finetuning_scheduler.tunable_view.roles import FROZEN_ROLE_KINDS, role_kind

        bound = as_view(view)
        if not bound.support().trainable:
            return self.model.parameters()
        roles = bound.roles()
        return (
            p
            for n, p in bound.root.named_parameters()
            if n not in roles or role_kind(roles[n]) not in FROZEN_ROLE_KINDS
        )

    def configure_optimizers(self) -> OptimizerLRScheduler | None:
        """Optional because it is not mandatory in the context of core IT modules (required for some adapter
        modules).

        A module with nothing to optimize gets no optimizer (or lr scheduler), with a warning, rather than torch's
        empty-parameter-list error: an attached view whose every tensor has a role finetuning-scheduler never optimizes
        (a circuit-tracer composition's transcoders are all ``instrument`` until promoted) still serves sessions that
        do not train, analysis among them, whose configuration sets ``optimizer_init`` (#859).
        """
        self.set_input_require_grads()  # type: ignore[attr-defined]  # provided by mixin composition
        optimizer, scheduler = None, None
        if self.it_cfg.optimizer_init:  # in case this hook is manually invoked by the user
            if params := list(self.optimizer_parameters()):
                optimizer = instantiate_class(args=params, init=self.it_cfg.optimizer_init)
            else:
                why = (
                    "every parameter of its tunable view has a role finetuning-scheduler never optimizes"
                    " (`instrument` or `fitted`) until a schedule promotes it"
                    if getattr(self, VIEW_ATTR, None) is not None
                    else "it has no parameters"
                )
                rank_zero_warn(
                    f"`optimizer_init` is set but {type(self).__name__} has nothing to optimize: {why}. No optimizer"
                    " is configured; promote the tensors to tune, or leave `optimizer_init` unset for a session that"
                    " does not train."
                )
        if optimizer is not None and self.it_cfg.lr_scheduler_init:
            scheduler = {
                "scheduler": instantiate_class(args=optimizer, init=self.it_cfg.lr_scheduler_init),
                **self.it_cfg.pl_lrs_cfg,
            }
        if optimizer is None:
            return None
        # no lr scheduler is an empty list, not `[None]`: Lightning refuses a `None` scheduler, and the core loop would
        # otherwise carry an `LRSchedulerConfig` wrapping `None`
        return [optimizer], [scheduler] if scheduler is not None else []  # type: ignore[return-value]

    def on_session_end(self) -> Any | None:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if getattr(self, "memprofiler", None):
            self.memprofiler.dump_memory_stats()  # type: ignore[union-attr]  # validated by getattr guard
        self._it_state._session_complete = True

    # N.B. we call `on_session_end` at the end of train, test, analysis and predict session types only. This is because
    # `on_train_end` and `on_validation_end` are called with most training sessions (when running both a fit and
    # evaluation loop as is usually the case) but only `on_test_end` with the test stage.
    def on_train_end(self) -> Any | None:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def on_test_end(self) -> Any | None:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def on_predict_end(self) -> Any | None:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def forward(self, *args, **kwargs) -> STEP_OUTPUT:
        """Forward to the wrapped model, merging in any configured custom forward kwargs.

        A batch keyed by a primary input the model's forward cannot take is refused here, by name. Analysis paths
        resolve input keys through aliases, so such a config passes them all and would otherwise fail only here, in
        the first Trainer loop, with the model's own unrelated message.
        """
        from interpretune.utils.tokenization import forward_call_mismatch

        if not args and (problem := forward_call_mismatch(self.model, kwargs)) is not None:
            raise MisconfigurationException(f"{type(self).__name__}: {problem}")
        return self.model(*args, **kwargs, **self.it_cfg.cust_fwd_kwargs)
