from typing import Any, Dict, Optional, Tuple, Callable, Mapping
from enum import auto
import importlib
from dataclasses import dataclass, field

from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig, ITSerializableCfg
from interpretune.base.modules import ITModule, BaseITModule
from interpretune.base.datamodules import ITDataModule
from interpretune.plugins.transformer_lens import BaseITLensModule, ITLensModule, TLensAttributeMixin
from interpretune.utils.warnings import unexpected_state_msg_suffix
from interpretune.utils.logging import rank_zero_warn
from interpretune.base.config.shared import AutoStrEnum
from interpretune.base.contract.protocol import (DataModuleInitable, ModuleSteppable, NamedWrapper, ITModuleProtocol,
                                                 ITDataModuleProtocol)
from interpretune.utils.import_utils import  _LIGHTNING_AVAILABLE

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import LightningDataModule, LightningModule
    from interpretune.frameworks.lightning import LightningAdapter
else:
    LightningDataModule = object
    LightningModule = object
    LightningAdapter = object

# NOTE [Interpretability Plugins]:
# `TransformerLens` is the first supported interpretability plugin but support for other plugins is expected

class Framework(AutoStrEnum):
    # CORE: The provided module and datamodule will be prepared for use with core PyTorch. The default
    #       trainer, a custom trainer or no trainer all can be used in combination with any supported and specified
    #       plugin.
    core = auto()
    # LIGHTNING: The provided module and datamodule will be prepared for use with the Lightning trainer and any
    #            supported and specified plugin.
    lightning = auto()


class Plugin(AutoStrEnum):
    # TRANSFORMER_LENS: The provided module and datamodule will be prepared for use with the TransformerLens plugin in
    #                  in combination with any supported and specified framework.
    transformer_lens = auto()


COMPOSITION_MAPPING = {
    # (component_key, framework_ctx, plugin_ctx)
    ('datamodule', Framework.core, None): (ITDataModule,),
    ('datamodule', Framework.lightning, None): (ITDataModule, LightningDataModule),
    ('datamodule', Framework.core, Plugin.transformer_lens): (ITDataModule,),
    ('datamodule', Framework.lightning, Plugin.transformer_lens): (ITDataModule, LightningDataModule),
    ('module', Framework.core, None): (ITModule,),
    ('module', Framework.lightning, None): (LightningAdapter, BaseITModule, LightningModule),
    ('module', Framework.core, Plugin.transformer_lens): ITLensModule,
    ('module', Framework.lightning, Plugin.transformer_lens): (TLensAttributeMixin, BaseITLensModule, LightningAdapter,
                                                     BaseITModule, LightningModule),
}


class ITMeta(type):
    def __new__(mcs, name, bases, classdict, **kwargs):
        component, input, ctx = mcs._validate_build_ctx(kwargs)
        # TODO: add runtime checks for adherence to IT protocol here?
        composition_classes = mcs._map_composition_target(component, ctx)
        if not isinstance(composition_classes, tuple):
            composition_classes = (composition_classes,)
        bases = (NamedWrapper, input, *composition_classes)
        built_class = super().__new__(mcs, name, bases, classdict)
        built_class._orig_module_name = input.__qualname__
        return built_class

    @staticmethod
    def _validate_build_ctx(kwargs: Dict) -> Tuple[str, Callable, Dict]:
        required_kwargs = ('component', 'input', 'ctx')
        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                raise ValueError(f"{kwarg} must be provided")
        if (component := kwargs.get('component')) not in ('module', 'datamodule', 'm', 'dm'):
            raise ValueError(f"Specified component was {component}, should be either 'module' or 'datamodule'")
        if not callable(input := kwargs.get('input')):
            raise ValueError(f"Specified input {input} is not a callable, it should be the class to be enriched.")
        if not isinstance(ctx := kwargs.get('ctx'), Dict):
            raise ValueError(f"Specified ctx {ctx} must be a dict specifying the desired class enrichment")
        return component, input, ctx

    @staticmethod
    def _map_composition_target(component, ctx):
        framework_ctx = ctx.get('framework_ctx', Framework.core)
        plugin_ctx = ctx.get('plugin_ctx', None)
        component_key = None
        match component:
            case 'datamodule' | 'dm':
                component_key = 'datamodule'
            case 'module' | 'm':
                component_key = 'module'
        assert component_key is not None, "invalid component, should be in: ('module', 'datamodule', 'm', 'dm')"
        return COMPOSITION_MAPPING[(component_key, framework_ctx, plugin_ctx)]

@dataclass(kw_only=True)
class UnencapsulatedArgs(ITSerializableCfg):
    # Most use cases will encapsulate datamodule/module config by subclassing the relevant dataclasses for a given
    # experiment/application but we also allow unencapsulated args/kwargs to be passed to the datamodule and module
    dm_args: Tuple = ()
    dm_kwargs: Dict[str, Any] = field(default_factory=dict)
    module_args: Tuple = ()
    module_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ITSessionConfig(UnencapsulatedArgs):
    datamodule: Optional[ITDataModuleProtocol] = None
    module: Optional[ITModuleProtocol] = None
    datamodule_cfg: ITDataModuleConfig
    datamodule_cls: str | DataModuleInitable
    module_cfg: ITConfig
    module_cls: str | ModuleSteppable
    framework_ctx: Framework | str = Framework.core
    plugin_ctx: Optional[Plugin | str] = None

    def __post_init__(self):
        if isinstance(self.framework_ctx, str):
            self.framework_ctx = Framework[self.framework_ctx]
        if isinstance(self.plugin_ctx, str):
            self.plugin_ctx = Plugin[self.plugin_ctx]
        # TODO: add checks validating supported framework and plugin combinations are provided
        # N.B. we must defer validating data/module_cls against their respective protocols until after composing
        # TODO: add warnings/errors here in case provided classes cannot be resolved or are invalid
        for attr_k in ["datamodule_cls", "module_cls"]:
            if isinstance(getattr(self, attr_k), str):
                module, module_cls = getattr(self, attr_k).rsplit(".", 1)
                mod = importlib.import_module(module)
                setattr(self, attr_k, getattr(mod, module_cls, None))


class ITSession(Mapping):

    FRAMEWORK_ATTRS = {Framework.core: ('datamodule', 'module'), Framework.lightning: ('datamodule', 'model')}
    READY_ATTRS = ["datamodule", "module"]
    COMPOSITION_TARGET_ATTRS = ["datamodule_cls", "module_cls"]
    READY_PROTOCOLS = [ITDataModuleProtocol, ITModuleProtocol]

    def __init__(self, session_cfg: ITSessionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datamodule = None
        self.module = None
        # to improve usability, run a datamodule cross-validation hook to allow auto-reconfiguration prior to
        # session instantiation
        session_cfg.datamodule_cfg._cross_validate(session_cfg.module_cfg)
        self.compose_interpretunable(session_cfg)
        self._ctx = self.FRAMEWORK_ATTRS[session_cfg.framework_ctx]


    def __getitem__(self, key):
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self):
        return len(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {self._ctx[0]: self.datamodule, self._ctx[1]: self.module}

    def _check_ready(self, session_cfg: ITSessionConfig) -> None:
        for ready, tocompose, ready_type in zip(self.READY_ATTRS, self.COMPOSITION_TARGET_ATTRS, self.READY_PROTOCOLS):
            if ready_mod := getattr(session_cfg, ready):  # if a module is ready, ensure we don't try to transform it
                if not isinstance(ready_mod, ready_type):
                    raise ValueError(f"{ready_mod} is not a {ready_type}, {ready_mod} should either be left unset (to"
                                     f" enable auto-composition of {getattr(session_cfg, tocompose)}) or be a"
                                     f" {ready_type}.")
                setattr(session_cfg, tocompose, None)

    def compose_interpretunable(self, session_cfg: ITSessionConfig) -> None:
        dm_cls, m_cls = None, None
        # 1. We first check to see if the datamodule and module classes are already defined and adhere to the relevant
        #    protocol.
        self._check_ready(session_cfg)
        # 1.1. If so, we can skip composition
        for ready, tocompose in zip(self.READY_ATTRS, self.COMPOSITION_TARGET_ATTRS):
            if getattr(session_cfg, tocompose) is not None:
                continue
            setattr(self, ready, getattr(session_cfg, ready))
        # 2. For the components (i.e. datamodule and module) that aren't ready, we compose the provided input component
        #    with the relevant classes based on the specified execution context and instantiate our enriched components!
        build_ctx = {'framework_ctx': session_cfg.framework_ctx, 'plugin_ctx': session_cfg.plugin_ctx}
        if session_cfg.datamodule_cls:  # 2.1 Compose datamodule if necessary
            dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm', input=session_cfg.datamodule_cls,
                            ctx=build_ctx)
            self.datamodule = dm_cls(itdm_cfg=session_cfg.datamodule_cfg, *session_cfg.dm_args, **session_cfg.dm_kwargs)
        self._set_dm_handles_for_instantiation(session_cfg)
        if session_cfg.module_cls:  # 2.2 Compose module if necessary
            m_cls = ITMeta('InterpretunableModule', (), {}, component='m', input=session_cfg.module_cls, ctx=build_ctx)
            self.module = m_cls(it_cfg=session_cfg.module_cfg, *session_cfg.module_args, **session_cfg.module_kwargs)
        self._set_model_handles_for_instantiation()
        self._validate_session(dm_cls, m_cls, session_cfg)

    def _set_dm_handles_for_instantiation(self, session_cfg: ITSessionConfig):
        # some datamodule handles may be required for module init, we update the module_cfg to provide them here
        supported_dm_handles_for_module = {"tokenizer": "tokenizer"}
        for m_attr, dm_handle in supported_dm_handles_for_module.items():
            # attach directly to module if it's ready, otherwise pass the handles to our module_cfg
            attr_target_obj = getattr(self, 'module', None) or session_cfg.module_cfg
            setattr(attr_target_obj, m_attr, getattr(self.datamodule, dm_handle))

    def _set_model_handles_for_instantiation(self):
        # having access to a model handle may be useful in some pre-setup hook steps (e.g. signature inspection in
        # `prepare_data`) we provide early access to the model handle for datamodule
        # TODO: for Lightning, since we're setting _module before trainer.model is set, we double check attr coherency
        # TODO: wrt Lightning compatibility, since we're providing access to the model handle before `setup_environment`
        #       or `configure_model` hooks are executed, need to evaluate the the validity/need to update this reference
        #       after subsequent model hooks are called
        supported_module_handles_for_datamodule = {"_module": self.module}
        for dm_attr, m_handle in supported_module_handles_for_datamodule.items():
            setattr(self.datamodule, dm_attr, m_handle)

    def _issue_warnings(self, dm_composed, m_composed, session_cfg, warn_datamodule, warn_module):
        if warn_datamodule:
            dm_warn_base = f"{self.datamodule} is not {ITDataModuleProtocol}"
            dm_warn_composed = (f" even after composing {session_cfg.datamodule_cls} with {dm_composed}."
                                f" {unexpected_state_msg_suffix}")
            not_ready_dm_warn = dm_warn_base + dm_warn_composed if dm_composed else dm_warn_base
            rank_zero_warn(not_ready_dm_warn)
        if warn_module:
            m_warn_base = f"{self.module} is not {ITModuleProtocol}"
            m_warn_composed = (f" even after composing {session_cfg.module_cls} with {m_composed}."
                            f" {unexpected_state_msg_suffix}")
            not_ready_m_warn = m_warn_base + m_warn_composed if m_composed else m_warn_base
            rank_zero_warn(not_ready_m_warn)

    def _validate_session(self, dm_composed, m_composed, session_cfg):
        warn_datamodule = not isinstance(self.datamodule, ITDataModuleProtocol)
        warn_module = not isinstance(self.module, ITModuleProtocol)
        if any([warn_datamodule, warn_module]):
            self._issue_warnings(dm_composed, m_composed, session_cfg, warn_datamodule, warn_module)
