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
"""Each module composition's finetuning-scheduler ``TunableView`` (#758) and its support record (#759).

A composition declares how it can be tuned by returning a view from ``build_tunable_view()``; the view's
``support()`` is the record. :func:`attach_tunable_view` attaches it to the module as ``tunable_view`` (the tunable view
protocol's ``VIEW_ATTR``) once the model is initialized, where finetuning-scheduler discovers it. Every view's root is
the interpretune module itself, so physical names are the module's own (``model.<...>``) and modules registered beside
the model (spliced SAEs, interventions) are reachable.

``BaseITModule.build_tunable_view()`` is itself a declaration (the identity view), so a composition that does not
override it is trainable; one whose ``build_tunable_view()`` returns ``None`` declares nothing and gets a view refusing
it, so it is never assumed trainable. Nothing is attached when the installed finetuning-scheduler predates views, and an
attached view is never replaced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

# `finetuning_scheduler.tunable_view.VIEW_ATTR`, kept here so attaching imports nothing from finetuning-scheduler
VIEW_ATTR = "tunable_view"


def fts_views_available() -> bool:
    """Whether the installed finetuning-scheduler provides the tunable view protocol."""
    try:
        import finetuning_scheduler.tunable_view  # noqa: F401
    except ImportError:
        return False
    return True


def peft_adapter_roles(root: nn.Module) -> dict[str, str]:
    """``adapter:<name>`` roles for every PEFT adapter tensor under ``root``, keyed by ``root``'s parameter names.

    Adapter tensors are found through PEFT's tuner-layer API (each layer's ``adapter_layer_names``), not by name, so
    every tuner whose layers hold their adapters in module or parameter dicts there (LoRA and its variants, IA3 and
    similar) is covered and a base weight PEFT renames (``base_layer.weight``) stays ``base``. Prompt-learning methods
    (prompt and prefix tuning), whose tensors live outside tuner layers, are not found and stay ``base``.
    """
    name_of = {id(p): n for n, p in root.named_parameters()}
    roles: dict[str, str] = {}
    for module in root.modules():
        for layer_name in getattr(module, "adapter_layer_names", ()) or ():
            container = getattr(module, layer_name, None)
            if not isinstance(container, (nn.ModuleDict, nn.ParameterDict)):
                continue
            for adapter, entry in container.items():
                params = entry.parameters() if isinstance(entry, nn.Module) else (entry,)
                for p in params:
                    if id(p) in name_of:
                        roles[name_of[id(p)]] = f"adapter:{adapter}"
    return roles


def identity_view(root: nn.Module, roles: Mapping[str, str] | None = None) -> Any:
    """Finetuning-scheduler's identity view over ``root``, with PEFT adapter roles and any further ``roles``."""
    from finetuning_scheduler.tunable_view import CanonicalView

    return CanonicalView(root, roles={**peft_adapter_roles(root), **(roles or {})})


class RefusingView:
    """A view declaring its module cannot be tuned, with the reason finetuning-scheduler reports when it refuses.

    Args:
        root: The module.
        reason: Why it cannot be tuned.
    """

    def __init__(self, root: nn.Module, reason: str) -> None:
        self.root = root
        self.reason = reason

    def support(self) -> Any:
        """Not trainable, with :attr:`reason`."""
        from finetuning_scheduler.tunable_view import TunableSupport

        return TunableSupport(trainable=False, reason=self.reason)

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Nothing resolves."""
        from finetuning_scheduler.tunable_view import Unresolvable

        raise Unresolvable(list(logical), self.reason)

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        """Names pass through unchanged."""
        return list(dict.fromkeys(physical))


def undeclared_reason(module: nn.Module) -> str:
    """The refusal reason for a composition that declares no tuning support."""
    return (
        f"{type(module).__name__} declares no tuning support (its `build_tunable_view()` returns None), so it is"
        " not assumed trainable."
    )


def attach_tunable_view(module: nn.Module) -> Any | None:
    """Attach ``module``'s view as ``tunable_view``, or a :class:`RefusingView` when it declares none.

    Does nothing when finetuning-scheduler predates views, or when a view is already attached (an explicit attachment,
    for example by a hub adapter's ``post_auto_model_init``, is never replaced).

    Returns:
        The attached view, or ``None`` when nothing was attached.
    """
    if not fts_views_available() or getattr(module, VIEW_ATTR, None) is not None:
        return None
    build = getattr(module, "build_tunable_view", None)
    view = build() if callable(build) else None
    if view is None:
        view = RefusingView(module, undeclared_reason(module))
    object.__setattr__(module, VIEW_ATTR, view)
    return view
