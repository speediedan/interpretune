from typing import Any, Dict, Optional, Tuple, Callable, Type, Protocol, Set, runtime_checkable, List, Sequence
from inspect import getmembers, isclass
from typing_extensions import override
from types import ModuleType
from pprint import pformat

from interpretune.utils import rank_zero_warn
from interpretune.protocol import Adapter

class CompositionRegistry(dict):
    # TODO: if this experimental compositional utility and protocol gains traction with external users:
    #         - change Adapter enum to a separate AdapterRegistry that can be loaded externally similar to extensions
    #           using the relevant entrypoint API config https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    def register(
        self,
        lead_adapter: Adapter,
        component_key: str,
        adapter_combination: Tuple[Adapter | str],
        composition_classes: Tuple[Callable[..., Any], ...],
        description: Optional[str] = None,
    ) -> None:
        """Registers valid component + adapter compositions mapped to composition keys with required metadata.

        Args:
            lead_adapter: The adapter registering this set of valid compositions (e.g. LightningAdapter)
            component_key: The name of the component (e.g. "datamodule")
            adapter_combination: tuple identifying the valid adapter composition
            composition_classes: Tuple[Callable, ...],
            description : composition description
        """
        supported_composition: Dict[str | Adapter | Tuple[Adapter | str], Tuple[Callable[..., Any], ...]] = {}
        composition_key = (component_key,) + self.canonicalize_composition(adapter_combination)
        supported_composition[composition_key] = composition_classes
        supported_composition['lead_adapter'] = Adapter[lead_adapter] if isinstance(lead_adapter, str) else lead_adapter
        supported_composition["description"] = description if description is not None else ""
        self[composition_key] = supported_composition

    @staticmethod
    def resolve_adapter_filter(adapter_filter: Optional[Sequence[Adapter| str]| Adapter | str] = None) -> List[Adapter]:
            unresolved_filters = []
            if isinstance(adapter_filter, str):
                adapter_filter = [Adapter[adapter_filter]]
            for adapter in adapter_filter:
                try:
                    adapter = CompositionRegistry.sanitize_adapter(adapter)
                except ValueError:
                    unresolved_filters.append(adapter)
            if unresolved_filters:
                rank_zero_warn("The following adapter names specified in `adapter_filter` could not be resolved: "
                               f" {unresolved_filters}.")
            return [adapter for adapter in adapter_filter if isinstance(adapter, Adapter)]

    @staticmethod
    def sanitize_adapter(adapter: Adapter | str) -> Adapter:
        if isinstance(adapter, str):
            try:
                adapter = Adapter[adapter]
            except KeyError:
                raise ValueError(f"Provided adapter string `{adapter}` could not be resolved.")
        return adapter

    def canonicalize_composition(self, adapter_ctx: Sequence[Adapter | str]) -> Tuple:
        resolved_adapter_ctx = set()
        for adapter in adapter_ctx:
            resolved_adapter_ctx.add(CompositionRegistry.sanitize_adapter(adapter))
        adapter_ctx = tuple(sorted(list(resolved_adapter_ctx), key=lambda a: a.value))
        return adapter_ctx

    @override
    def get(self, composition_key: Tuple[Adapter | str]) -> Any:
        if composition_key in self:
            supported_composition = self[composition_key]
            return supported_composition[composition_key]

        available_keys = pformat(self.keys()) or "none"
        err_msg = (f"The composition key `{composition_key}` was not found in the registry."
                   f" Available valid compositions: {available_keys}")
        raise KeyError(err_msg)

    def remove(self, composition_key: Tuple[Adapter | str]) -> None:
        """Removes the registered adapter composition by name."""
        del self[composition_key]

    def available_compositions(self, adapter_filter: Optional[Sequence[Adapter| str]| Adapter | str] = None) -> Set:
        """Returns a list of registered adapters, optionally filtering by the lead adapter that registered the
        valid composition."""
        if adapter_filter is not None:
            adapter_filter = CompositionRegistry.resolve_adapter_filter(adapter_filter)
            return {key for key in self.keys() for subkey in key if subkey in adapter_filter}
        return set(self.keys())

    def __str__(self) -> str:
        return f"Registered Adapter Compositions: {pformat(self.keys())}"

@runtime_checkable
class AdapterProtocol(Protocol):
    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None: ...

def _register_adapters(registry: Any, method: str, module: ModuleType, parent: Type[object]) -> None:
    for _, member in getmembers(module, isclass):
        if issubclass(member, parent) and member is not parent:  # and is_overridden(method, member, parent):
            register_fn = getattr(member, method)
            register_fn(registry)
