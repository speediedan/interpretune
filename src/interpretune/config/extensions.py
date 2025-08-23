from typing import Any, Dict, Callable, NamedTuple
from dataclasses import field, make_dataclass, dataclass

from interpretune.config import ITSerializableCfg
from interpretune.utils import MisconfigurationException, _import_class, _NEURONPEDIA_AVAILABLE


class ITExtension(NamedTuple):
    ext_attr: str
    ext_cls_fqn: str
    ext_cfg_fqn: str


@dataclass
class ExtensionsContext:
    """Context for managing extensions in Interpretune.

    This class provides a structured way to handle extensions, including their configuration and instantiation. It
    allows for easy integration of new extensions into the Interpretune framework.
    """

    SUPPORTED_EXTENSIONS: Dict[str, Callable] = field(default_factory=dict)
    SUPPORTED_EXTENSION_CFGS: Dict[str, Any] = field(default_factory=dict)
    BASE_EXTENSIONS: tuple = (
        ITExtension(
            "debug_lm",
            "interpretune.extensions.debug_generation.DebugGeneration",
            "interpretune.extensions.debug_generation.DebugLMConfig",
        ),
        ITExtension(
            "memprofiler",
            "interpretune.extensions.memprofiler.MemProfiler",
            "interpretune.extensions.memprofiler.MemProfilerCfg",
        ),
    )
    OPTIONAL_EXTENSIONS: list = field(default_factory=list)
    DEFAULT_EXTENSIONS: tuple = field(init=False)

    def __post_init__(self):
        if _NEURONPEDIA_AVAILABLE:
            self.OPTIONAL_EXTENSIONS.append(
                ITExtension(
                    "neuronpedia",
                    "interpretune.extensions.neuronpedia.NeuronpediaIntegration",
                    "interpretune.extensions.neuronpedia.NeuronpediaConfig",
                )
            )
        self.DEFAULT_EXTENSIONS = self.BASE_EXTENSIONS + tuple(self.OPTIONAL_EXTENSIONS)


class ITExtensionsConfigMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extensions_context = ExtensionsContext()

    def _detect_extensions(self):
        # TODO: update custom extensions to be read/added from it_cfg once interface stabilizes
        for it_ext in self.extensions_context.DEFAULT_EXTENSIONS:
            try:
                ext_class = _import_class(it_ext.ext_cls_fqn)
                ext_cfg_class = _import_class(it_ext.ext_cfg_fqn)
            except (ImportError, AttributeError) as e:
                err_msg = (
                    f"Unable to import and resolve specified extension and/or its configuration class from"
                    f" `ext_cls_fqn`: {it_ext.ext_cls_fqn}, `ext_cfg_fqn`: {it_ext.ext_cfg_fqn}, error: {e}"
                )
                raise MisconfigurationException(err_msg)
            self.extensions_context.SUPPORTED_EXTENSIONS[it_ext.ext_attr] = ext_class
            self.extensions_context.SUPPORTED_EXTENSION_CFGS[it_ext.ext_attr] = ext_cfg_class

    def _connect_extensions(self):
        self._detect_extensions()
        for ext_name, ext_class in self.extensions_context.SUPPORTED_EXTENSIONS.items():
            if getattr(self.it_cfg, f"{ext_name}_cfg").enabled:
                self._it_state._extensions[ext_name] = ext_class()
                getattr(self, ext_name).connect(self)
            else:
                self._it_state._extensions[ext_name] = None

    def __getattr__(self, name: str) -> Any:
        # we make extension handles available as direct root module attributes for convenience
        # filter only the supported extension attributes, ensuring `_it_state`` has been initialized
        if (
            self.extensions_context.SUPPORTED_EXTENSIONS.get(name)
            and (ext_attrs := self.__dict__.get("_it_state", None)) is not None
        ):
            return ext_attrs._extensions[name]
        return super().__getattr__(name)


# TODO: rather than load extensions from DEFAULT_EXTENSIONS, use a registry and the entry_point API to potentially load
#       external extensions if ever advertising such an API
supported_ext_cfgs = []
TMP_EXT_REGISTRY = ITExtensionsConfigMixin()
TMP_EXT_REGISTRY._detect_extensions()
for ext_attr, ext_cfg_cls in TMP_EXT_REGISTRY.extensions_context.SUPPORTED_EXTENSION_CFGS.items():
    supported_ext_cfgs.append((f"{ext_attr}_cfg", ext_cfg_cls, field(default_factory=ext_cfg_cls)))
ExtensionConf = make_dataclass("ExtensionsConf", supported_ext_cfgs, bases=(ITSerializableCfg,), kw_only=True)
