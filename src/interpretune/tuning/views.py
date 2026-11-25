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
the model (spliced SAEs, interventions) are reachable. Views come from venngram, so they attach whether or not
finetuning-scheduler is installed.

``BaseITModule.build_tunable_view()`` is itself a declaration: venngram's composed view (:func:`default_view`), which
places the model's tensors by layer and component wherever a published component map describes it and falls back to
the identity view where none does, so a composition that does not override it is trainable; one whose
``build_tunable_view()`` returns ``None`` declares nothing and gets a view refusing it, so it is never assumed
trainable. An attached view is never replaced.

A run can choose the view in its configuration instead (``it_cfg.tunable_view``, #841): a ``class_path`` and
``init_args``, instantiated with the module as the view's root by :func:`configured_view`. It takes precedence over
``build_tunable_view()``, so a schedule's naming (TransformerLens names or the module's own, the implicit norm thaw on
or off) is part of the run's configuration rather than of a Python subclass.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from interpretune.tuning.sites import peft_base_model
from interpretune.utils import rank_zero_info, rank_zero_warn

# `venngram.tuning.VIEW_ATTR`, kept here so attaching imports neither venngram nor finetuning-scheduler
VIEW_ATTR = "tunable_view"


def views_available() -> bool:
    """Whether venngram's tunable view protocol is importable: views attach whenever it is, whether or not
    finetuning-scheduler is installed (#874)."""
    try:
        import venngram.tuning  # noqa: F401
    except ImportError:
        return False
    return True


def fts_views_available() -> bool:
    """Whether the installed finetuning-scheduler schedules through tunable views.

    The view protocol itself lives in venngram (``venngram.tuning``, where finetuning-scheduler #162 moved it);
    finetuning-scheduler's ``core.views`` is the consumer that attaches and binds a view, so both must import.
    """
    try:
        import finetuning_scheduler.core.views  # noqa: F401
        import venngram.tuning  # noqa: F401
    except ImportError:
        return False
    return True


def module_address(*paths: str) -> str:
    """The venngram address selecting every tensor under the modules at ``paths`` (``module:<path>``, matched on
    whole segments), as a union."""
    return " | ".join(f"module:{path}" for path in paths)


def merge_roles(*declarations: Mapping[str, str]) -> dict[str, str]:
    """Merge role declarations as venngram views take them (``roles=``): tensor names mapped to roles, and roles
    mapped to addresses (speediedan/venngram#109).

    A later name entry replaces an earlier one, as a mapping update does. A role two declarations each map to an
    address selects the union of both, so cooperative ``tunable_roles()`` overrides add to ``super()``'s rather than
    replacing it. A key that parses as a role is taken as one here; the view still reads a key naming a tensor as a
    name entry (venngram's rule, :func:`venngram.tuning.views.expand_roles`).
    """
    from venngram.roles import parse_role

    def is_role(key: str) -> bool:
        try:
            parse_role(key)
        except ValueError:
            return False
        return True

    merged: dict[str, str] = {}
    for declared in declarations:
        for key, value in declared.items():
            if key in merged and is_role(key) and merged[key] != value:
                merged[key] = f"{merged[key]} | {value}"
            else:
                merged[key] = value
    return merged


def expand_declared_roles(root: nn.Module, roles: Mapping[str, str], structure: Any = None) -> dict[str, str]:
    """``roles`` (names, or roles mapped to addresses) as tensor names mapped to roles, for a view that takes names
    only: each address is resolved against ``root``'s parameters and buffers, placed by ``structure`` when given
    (venngram's :func:`~venngram.tuning.views.expand_roles`, which states what it refuses)."""
    from venngram.tuning.views import expand_roles

    names = [n for n, _ in root.named_parameters()] + [n for n, _ in root.named_buffers()]
    return expand_roles(roles, names, structure)


def adapter_roles(root: nn.Module) -> dict[str, str]:
    """The roles venngram's overlay maps give ``root``'s tensors: PEFT adapter tensors as ``adapter:<name>``, for
    every tuner family a seed overlay describes, and whatever else a known overlay covers.

    An unreadable registry gives no roles, with a warning: the view then still schedules by name.
    """
    from venngram.maps import overlay_roles, overlays_for
    from venngram.registry import RegistryError

    try:
        return overlay_roles(root, overlays_for(root))
    except RegistryError as err:
        rank_zero_warn(
            f"venngram's overlay maps could not be read ({err}), so the adapter and other library tensors of "
            f"{type(root).__name__} have no roles in its view. Warm venngram's registry cache (`VENNGRAM_CACHE`) or "
            "register the overlays in this process."
        )
        return {}


def identity_view(root: nn.Module, roles: Mapping[str, str] | None = None) -> Any:
    """The identity view over ``root`` (physical names, no layers or components), with the roles venngram's
    overlays give its tensors (:func:`adapter_roles`) and any further ``roles``."""
    from venngram.tuning import CanonicalView

    return CanonicalView(root, roles={**adapter_roles(root), **(roles or {})})


_REMINDED: set[str] = set()


def _peft_mount(root: nn.Module) -> str | None:
    """The path in ``root`` of the model a PEFT wrapper at ``root.model`` carries (``model.base_model.model``), or
    ``None`` when ``root.model`` is not a PEFT model.

    venngram mounts a base map at the outermost module whose class or ``config.architectures`` names a mapped
    architecture. A ``PeftModel`` reports its base model's ``config``, which records ``architectures`` once loaded from
    a checkpoint, so without a named mount the map would be placed at the wrapper, where none of its paths are (#1142).
    """
    base = peft_base_model(getattr(root, "model", None))
    if base is None:
        return None
    return next((path for path, module in root.named_modules() if module is base and path), None)


def default_view(
    root: nn.Module, roles: Mapping[str, str] | None = None, attached: Mapping[str, str] | None = None
) -> Any:
    """``root``'s view through venngram (``venngram.view``): the base map of the model it carries, mounted where
    the model sits (``model.``), the roles of every overlay that applies, and ``roles`` over them.

    ``attached`` places the modules spliced in beside the model (latent models and interventions, by the module each
    rewrites: ``BaseITModule.tunable_placements()``), so their tensors take the layer and component of their site and
    ``role:latent & layers:12:`` selects by depth. A placement the map cannot make (a site no map row places at or
    above) is left out, with a warning: that module's tensors locate at no layer, and still resolve by name and role.

    A model PEFT wraps is described where PEFT keeps it (``model.base_model.model``), not at the wrapper.

    Where no published component map describes the model, this is the identity view (:func:`identity_view`), with an
    info-level reminder that registering a map adds layers and components; where venngram's registry cannot be read,
    it is the identity view with a warning. Either way the composition stays trainable.
    """
    import venngram
    from venngram.registry import RegistryError
    from venngram.tuning import TunableViewError

    roles, attached, mount = dict(roles or {}), dict(attached or {}), _peft_mount(root)
    try:
        try:
            return venngram.view(root, roles=roles, attached=attached, mount=mount)
        except TunableViewError:
            if not attached:
                raise
        unattached = venngram.view(root, roles=roles, mount=mount)  # a refusal that is not a placement's is raised here
        # venngram refuses the view for any one placement it cannot make: keep those it can, one by one
        placeable, unplaced = {}, {}
        for key, at in attached.items():
            try:
                venngram.view(root, roles=roles, attached={key: at}, mount=mount)
                placeable[key] = at
            except TunableViewError as err:
                unplaced[key] = err
        reasons = "; ".join(map(str, unplaced.values()))
        rank_zero_warn(
            f"{', '.join(unplaced)} could not be placed at their sites in {type(root).__name__}'s view, so their"
            f" tensors locate at no layer or component: address them by name or role ({reasons})."
        )
        return venngram.view(root, roles=roles, attached=placeable, mount=mount) if placeable else unattached
    except KeyError as err:
        model = getattr(root, "model", None)
        named = f"{type(root).__name__}" + (f" ({type(model).__name__})" if isinstance(model, nn.Module) else "")
        if named not in _REMINDED:
            _REMINDED.add(named)
            rank_zero_info(
                f"No venngram component map describes {named}, so its tunable view addresses tensors by name and role "
                "only. Registering a map (`venngram.maps.register`) or publishing one to a registry adds layers and "
                f"components ({err.args[0] if err.args else err})."
            )
    except RegistryError as err:
        rank_zero_warn(
            f"venngram's registry could not be read ({err}), so {type(root).__name__} gets the identity view: no "
            "layers or components. Warm venngram's registry cache (`VENNGRAM_CACHE`) to address it by structure."
        )
    return identity_view(root, roles)


def configured_view(module: nn.Module) -> Any | None:
    """The view ``module.it_cfg.tunable_view`` configures (#841), or ``None`` when none is configured.

    The view class is instantiated with ``module`` as its root and the configured ``init_args``. A class taking
    ``roles`` receives the composition's roles (the overlays' roles, :func:`adapter_roles`, and the module's
    ``tunable_roles()``) unless ``init_args`` sets them, and a class taking ``attached`` (venngram's ``MappedView``)
    receives the placements of the modules spliced in beside the model (``tunable_placements()``) unless ``init_args``
    sets them.
    """
    cfg = getattr(getattr(module, "it_cfg", None), "tunable_view", None)
    if not cfg:
        return None
    from interpretune.utils import instantiate_class

    view_cls = instantiate_class({"class_path": cfg["class_path"]}, import_only=True)
    init_args = dict(cfg.get("init_args") or {})
    if "roles" not in init_args and "roles" in inspect.signature(view_cls).parameters:
        tunable_roles = getattr(module, "tunable_roles", None)
        declared = tunable_roles() if callable(tunable_roles) else None
        init_args["roles"] = {**adapter_roles(module), **(declared if isinstance(declared, Mapping) else {})}
    if "attached" not in init_args and "attached" in inspect.signature(view_cls).parameters:
        tunable_placements = getattr(module, "tunable_placements", None)
        placed = tunable_placements() if callable(tunable_placements) else None
        if isinstance(placed, Mapping) and placed:
            init_args["attached"] = dict(placed)
    return view_cls(module, **init_args)


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
        from venngram.tuning import TunableSupport

        return TunableSupport(trainable=False, reason=self.reason)

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Nothing resolves."""
        from venngram.tuning import Unresolvable

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


def check_books_of(view: Any, module: nn.Module) -> None:
    """Give ``view`` the ``check_book`` member finetuning-scheduler asks for (``book -> None``, raising venngram's
    ``PinMismatchError``), checking a schedule's pinned address books against the models ``module`` has loaded
    (:func:`~interpretune.tuning.pins.check_book_pins`).

    It is set on the instance, so it serves every view class a composition uses, venngram's own included. A view that
    already defines the member keeps its own, and one that takes no new attribute is left without it, so a pinned book
    stays refused through it.
    """
    if callable(getattr(view, "check_book", None)):
        return

    def check_book(book: Any) -> None:
        from interpretune.tuning.pins import check_book_pins

        check_book_pins(book, module)

    try:
        object.__setattr__(view, "check_book", check_book)
    except AttributeError:  # `__slots__`
        return


def attach_tunable_view(module: nn.Module) -> Any | None:
    """Attach ``module``'s view as ``tunable_view``: the one its configuration names (:func:`configured_view`),
    else the one ``build_tunable_view()`` declares, or a :class:`RefusingView` when it declares none.

    Does nothing when venngram's view protocol is unavailable, or when a view is already attached (an explicit
    attachment, for example by a hub adapter's ``post_auto_model_init``, is never replaced).

    The attached view checks the pins of a schedule's address books against the models ``module`` has loaded
    (:func:`check_books_of`).

    Returns:
        The attached view, or ``None`` when nothing was attached.
    """
    if not views_available() or getattr(module, VIEW_ATTR, None) is not None:
        return None
    view = configured_view(module)
    if view is None:
        build = getattr(module, "build_tunable_view", None)
        view = build() if callable(build) else None
    if view is None:
        view = RefusingView(module, undeclared_reason(module))
    check_books_of(view, module)
    object.__setattr__(module, VIEW_ATTR, view)
    return view
