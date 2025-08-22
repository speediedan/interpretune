import logging
from typing import Any, Union

from interpretune.base import ITDataModule, BaseITModule
from interpretune.utils import rank_zero_info
from interpretune.protocol import AllPhases, CorePhases

log = logging.getLogger(__name__)

HOOKABLE_ITMODULE = Union[ITDataModule, BaseITModule]

def it_init(module, datamodule, *args, **kwargs):
    _call_itmodule_hook(datamodule, hook_name="prepare_data", hook_msg="Preparing data", target_model=module.model)
    _call_itmodule_hook(datamodule, hook_name="setup", hook_msg="Setting up datamodule")
    _call_itmodule_hook(module, hook_name="setup", hook_msg="Setting up model", datamodule=datamodule)
    if module.it_cfg.optimizer_init:  # only attempt optimizer/scheduler initialization if a configuration is provided
        _call_itmodule_hook(module, hook_name="configure_optimizers", hook_msg="initializing optimizers and schedulers",
                            connect_output=True)

def it_session_end(module, datamodule, session_type: AllPhases = CorePhases.train, *args, **kwargs):
    # dispatch the appropriate stage-specific `end` hook upon completion of the session
    hook_name = f"on_{session_type.name}_end"
    _call_itmodule_hook(module, hook_name=hook_name, hook_msg="Running stage end hooks on IT module")
    _call_itmodule_hook(datamodule, hook_name=hook_name, hook_msg="Running stage end hooks on IT datamodule")

# TODO: consider adding IT teardown hooks (among others)

class _hookNameContextManager:
    """A context manager to change the default tensor type when tensors get created.

    See: :func:`torch.set_default_dtype`
    """

    def __init__(self, hookable_module: HOOKABLE_ITMODULE, hook_name: str) -> None:
        if not hasattr(hookable_module, '_current_fx_name'):
            setattr(hookable_module, '_current_fx_name', None)
        self._module_ref = hookable_module
        self._hook_name = hook_name
        self.previous_fx_name = None

    def __enter__(self) -> None:
        self.previous_fx_name = self._module_ref._current_fx_name
        self._module_ref._current_fx_name = self._hook_name  # type: ignore[assignment]

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._module_ref._current_fx_name = self.previous_fx_name  # type: ignore[assignment]

def _call_itmodule_hook(
    hookable_module: HOOKABLE_ITMODULE,
    hook_name: str,
    hook_msg: str | None = None,
    connect_output: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Any | None:

    hook_msg = hook_msg + ": " if hook_msg else ""

    fn = getattr(hookable_module, hook_name)

    with _hookNameContextManager(hookable_module, hook_name):
        rank_zero_info(f"{hook_msg}{hookable_module.__class__.__name__}")
        output = fn(*args, **kwargs)
    if connect_output:
        hookable_module._hook_output_handler(hook_name, output)
    else:
        return output
