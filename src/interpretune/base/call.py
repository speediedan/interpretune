# Copyright Lightning AI.
#
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
import logging
from contextlib import contextmanager
from typing import Any, Union, Optional

from interpretune.base.it_datamodule import ITDataModule
from interpretune.base.it_module import BaseITModule
from interpretune.utils.logging import rank_zero_info

log = logging.getLogger(__name__)

HOOKABLE_ITMODULE = Union[ITDataModule, BaseITModule]

def it_init(module, datamodule, *args, **kwargs):
    _call_itmodule_hook(datamodule, hook_name="prepare_data", hook_msg="Preparing data", target_model=module.model)
    _call_itmodule_hook(datamodule, hook_name="setup", hook_msg="Setting up datamodule")
    _call_itmodule_hook(module, hook_name="setup", hook_msg="Setting up model", datamodule=datamodule)
    if module.it_cfg.optimizer_init:  # only attempt optimizer/scheduler initialization if a configuration is provided
        _call_itmodule_hook(module, hook_name="configure_optimizers", hook_msg="initializing optimizers and schedulers",
                            connect_output=True)

def it_session_end(module, datamodule, *args, **kwargs):
    # N.B. currently associating session end with `on_train_end` hooks to ensure compatibility with Lightning
    _call_itmodule_hook(module, hook_name="on_train_end", hook_msg="Running session end hooks on IT module")
    _call_itmodule_hook(datamodule, hook_name="on_train_end", hook_msg="Running session end hooks on IT datamodule")

# TODO: consider adding IT teardown hooks (among others)

# TODO: consider removing this helper function as it is helpful to users in a fairly narrow context
@contextmanager
def model_call_redirect(wrapper_module: Any) -> None:
    """context manager making an arbitrary model-wrapping object callable, redirecting calls to an underlying
    model."""
    try:
        dunder_redirects = [("__call__", wrapper_module.model)]
        for method, target in dunder_redirects:
            if not hasattr(wrapper_module, method):
                redirect_target = getattr(target, method, None)
                assert redirect_target, f"Could not find redirect method `{method}` on target `{target}`."
                # N.B. set at class-level https://docs.python.org/3/reference/datamodel.html#emulating-callable-objects
                setattr(wrapper_module.__class__, method, redirect_target)
        yield
    finally:
        for method, target in dunder_redirects:
            if hasattr(wrapper_module, method):
                delattr(wrapper_module.__class__, method)


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
        self._module_ref._current_fx_name = self._hook_name

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._module_ref._current_fx_name = self.previous_fx_name

def _call_itmodule_hook(
    hookable_module: HOOKABLE_ITMODULE,
    hook_name: str,
    hook_msg: Optional[str] = None,
    connect_output: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Optional[Any]:

    hook_msg = hook_msg + ": " if hook_msg else ""

    fn = getattr(hookable_module, hook_name)
    if not callable(fn):
        return None

    with _hookNameContextManager(hookable_module, hook_name):
        rank_zero_info(f"{hook_msg}{hookable_module.__class__.__name__}")
        output = fn(*args, **kwargs)
    if connect_output:
        hookable_module._hook_output_handler(hook_name, output)
    else:
        return output
