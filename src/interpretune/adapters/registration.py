from __future__ import annotations
from typing import Any, Tuple, Callable, Type, Protocol, Set, runtime_checkable, Sequence, cast
from inspect import getmembers, isclass
from typing_extensions import override
from types import ModuleType
from pprint import pformat

from interpretune.utils import rank_zero_warn
from interpretune.protocol import Adapter


class CompositionRegistry(dict):
    """Maps an adapter combination to the classes composed for it, per component.

    Interpretune builds a datamodule/module by MRO composition rather than inheritance from a fixed
    base, so the registry is what answers "given adapters (core, sae_lens, transformer_lens), which
    classes compose, in what order". Adapters register into it at import time via
    :meth:`AdapterProtocol.register_adapter_ctx`; keys are canonicalized so combinations written in any
    order resolve identically.
    """

    # TODO: if this experimental compositional utility and protocol gains traction with external users:
    #         - change Adapter enum to a separate AdapterRegistry that can be loaded externally similar to extensions
    #           using the relevant entrypoint API config https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    def register(
        self,
        lead_adapter: Adapter,
        component_key: str,
        adapter_combination: tuple[Adapter | str],
        composition_classes: tuple[Callable[..., Any], ...],
        description: str | None = None,
    ) -> None:
        """Registers valid component + adapter compositions mapped to composition keys with required metadata.

        Args:
            lead_adapter: The adapter registering this set of valid compositions (e.g. LightningAdapter)
            component_key: The name of the component (e.g. "datamodule")
            adapter_combination: tuple identifying the valid adapter composition
            composition_classes: tuple[Callable, ...],
            description : composition description
        """
        supported_composition: dict[str | Adapter | tuple[Adapter | str], Any] = {}
        composition_key = (component_key,) + self.canonicalize_composition(adapter_combination)
        supported_composition[composition_key] = composition_classes
        supported_composition["lead_adapter"] = Adapter[lead_adapter] if isinstance(lead_adapter, str) else lead_adapter
        supported_composition["description"] = description if description is not None else ""
        self[composition_key] = supported_composition

    @staticmethod
    def resolve_adapter_filter(
        adapter_filter: Sequence[Adapter | str] | Adapter | str | None = None,
    ) -> list[Adapter]:
        """Normalize a filter (None, a single adapter, a string, or a sequence) to a list of adapters.

        None means "no filtering" and returns an empty list, which callers read as match-everything -- distinct from an
        empty filter matching nothing.
        """
        unresolved_filters = []
        if adapter_filter is None:
            return []
        if isinstance(adapter_filter, str):
            adapter_filter = [Adapter[adapter_filter]]
        elif isinstance(adapter_filter, Adapter):
            adapter_filter = [adapter_filter]
        for adapter in adapter_filter:
            try:
                adapter = CompositionRegistry.sanitize_adapter(adapter)
            except ValueError:
                unresolved_filters.append(adapter)
        if unresolved_filters:
            rank_zero_warn(
                "The following adapter names specified in `adapter_filter` could not be resolved: "
                f" {unresolved_filters}."
            )
        return [adapter for adapter in adapter_filter if isinstance(adapter, Adapter)]

    @staticmethod
    def sanitize_adapter(adapter: Adapter | str) -> Adapter:
        """Coerce an adapter name to its :class:`Adapter` member.

        Raises:
            ValueError: the string names no known adapter -- surfaced here, where the offending name is
                still in hand, rather than as a later composition miss.
        """
        if isinstance(adapter, str):
            try:
                adapter = Adapter[adapter]
            except KeyError:
                raise ValueError(f"Provided adapter string `{adapter}` could not be resolved.")
        return adapter

    def canonicalize_composition(self, adapter_ctx: Sequence[Adapter | str]) -> Tuple:
        """Reduce an adapter context to its canonical form: de-duplicated, coerced, and value-sorted.

        This is what makes composition keys order-independent -- ``(sae_lens, core)`` and
        ``(core, sae_lens)`` must name the same registered composition, or callers would have to know
        the registration order to look one up.
        """
        resolved_adapter_ctx: set[Adapter] = set()
        for adapter in adapter_ctx:
            resolved_adapter_ctx.add(CompositionRegistry.sanitize_adapter(adapter))
        # All items in resolved_adapter_ctx are now guaranteed to be Adapter objects
        adapter_list: list[Adapter] = list(resolved_adapter_ctx)
        adapter_ctx = tuple(sorted(adapter_list, key=lambda a: cast(Adapter, a).value))
        return adapter_ctx

    @override
    def get(self, composition_key: tuple[Adapter | str], default: Any = None) -> Any:
        if composition_key in self:
            supported_composition = self[composition_key]
            return supported_composition[composition_key]

        if default is not None:
            return default

        available_keys = pformat(self.keys()) or "none"
        err_msg = (
            f"The composition key `{composition_key}` was not found in the registry."
            f" Available valid compositions: {available_keys}"
        )
        raise KeyError(err_msg)

    def remove(self, composition_key: tuple[Adapter | str]) -> None:
        """Removes the registered adapter composition by name."""
        del self[composition_key]

    def available_compositions(self, adapter_filter: Sequence[Adapter | str] | Adapter | str | None = None) -> Set:
        """Returns a list of registered adapters, optionally filtering by the lead adapter that registered the
        valid composition."""
        if adapter_filter is not None:
            adapter_filter = CompositionRegistry.resolve_adapter_filter(adapter_filter)
            return {key for key in self.keys() for subkey in key if subkey in adapter_filter}
        return set(self.keys())

    def __str__(self) -> str:
        return f"Registered Adapter Compositions: {pformat(self.keys())}"


#: Adapter names added at runtime by hub components, mapped to the repo that added each one. Kept so a
#: second load of the same repo is idempotent while two repos claiming one name is an error rather than
#: a silent last-writer-wins.
_DYNAMIC_ADAPTERS: dict[str, str] = {}


class DynamicAdapterError(ValueError):
    """A hub component's declared adapter name cannot be added to :class:`~interpretune.protocol.Adapter`."""


def register_dynamic_adapter(name: str, *, source: str) -> Adapter:
    """Add ``name`` to the :class:`~interpretune.protocol.Adapter` enum on behalf of a hub component.

    Composition keys are built from ``Adapter`` members, so an adapter that arrives at runtime has to become
    one before it can register anything. ``Adapter`` is closed at class-creation time like any enum, so this
    installs the member through the enum's own member maps rather than rebuilding the class: rebuilding would
    orphan every ``Adapter`` reference already held by a live session.

    Idempotent per source: reloading the same component returns the member it added. A name already taken by
    a built-in adapter, or by a DIFFERENT component, raises -- two components silently sharing one name would
    make composition keys mean different things in different sessions.
    """
    if not name.isidentifier():
        raise DynamicAdapterError(f"{source}: adapter name {name!r} is not a valid Python identifier")
    existing = Adapter._member_map_.get(name)
    if existing is not None:
        owner = _DYNAMIC_ADAPTERS.get(name)
        if owner is None:
            raise DynamicAdapterError(
                f"{source}: adapter name {name!r} is a built-in interpretune adapter and cannot be redefined "
                "by a hub component."
            )
        if owner != source:
            raise DynamicAdapterError(
                f"{source}: adapter name {name!r} was already registered by {owner!r}. Two components cannot "
                "share one adapter name: composition keys are built from these members, so the same key would "
                "mean different things depending on load order."
            )
        return cast(Adapter, existing)
    member = object.__new__(Adapter)
    member._name_ = name
    member._value_ = name
    # `type.__setattr__` deliberately: EnumMeta refuses attribute assignment for names it already knows, and
    # the attribute has to exist before the member maps claim the name.
    type.__setattr__(Adapter, name, member)
    Adapter._member_map_[name] = member
    Adapter._member_names_.append(name)
    Adapter._value2member_map_[name] = member
    _DYNAMIC_ADAPTERS[name] = source
    return member


def dynamic_adapters() -> dict[str, str]:
    """Adapter names added at runtime, mapped to the component that added each."""
    return dict(_DYNAMIC_ADAPTERS)


@runtime_checkable
class AdapterProtocol(Protocol):
    """What a class must implement to participate in adapter composition."""

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        """Register this adapter's supported combinations and their composition classes."""
        ...


def _register_adapters(registry: Any, method: str, module: ModuleType, parent: Type[object]) -> None:
    for _, member in getmembers(module, isclass):
        if issubclass(member, parent) and member is not parent:  # and is_overridden(method, member, parent):
            register_fn = getattr(member, method)
            register_fn(registry)
