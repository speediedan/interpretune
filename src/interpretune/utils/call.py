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
from typing import Any, Union

from interpretune.base.base_datamodule import ITDataModule
from interpretune.base.base_module import ITHookedModule

log = logging.getLogger(__name__)


class _hookNameContextManager:
    """A context manager to change the default tensor type when tensors get created.

    See: :func:`torch.set_default_dtype`
    """

    def __init__(self, it_module: Union["ITHookedModule", "ITDataModule"], hook_name: str) -> None:
        if not hasattr(it_module, '_current_fx_name'):
            setattr(it_module, '_current_fx_name', None)
        self._module_ref = it_module
        self._hook_name = hook_name
        self.previous_fx_name = None

    def __enter__(self) -> None:
        self.previous_fx_name = self._module_ref._current_fx_name
        self._module_ref._current_fx_name = self._hook_name

    def __exit__(self) -> None:
        self._module_ref._current_fx_name = self.previous_fx_name

def _call_itmodule_hook(
    it_module: "ITHookedModule",
    hook_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:

    fn = getattr(it_module, hook_name)
    if not callable(fn):
        return None

    with _hookNameContextManager(it_module, hook_name):
        output = fn(*args, **kwargs)

    return output
