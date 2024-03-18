from typing import Any, Dict, Optional, Tuple, Callable

import importlib
from dataclasses import dataclass, field

from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig, ITSerializableCfg
from interpretune.base.modules import ITModule, ITLightningModule
from interpretune.base.datamodules import ITLightningDataModule, ITDataModule
from interpretune.plugins.transformer_lens import ITLensLightningModule, ITLensModule
from interpretune.utils.warnings import unexpected_state_msg_suffix
from interpretune.utils.logging import rank_zero_warn
from interpretune.base.contract.protocol import (DataModuleInitable, ModuleSteppable, NamedWrapper, ITModuleProtocol,
                                                 ITDataModuleProtocol)


# NOTE [Interpretability Plugins]:
# `TransformerLens` is the first supported interpretability plugin but support for other plugins is expected

INTERPRETUNABLE_MODULE_MAPPING = {
    # (lightning, plugin_key)
    (False, None): ITModule,
    (True, None): ITLightningModule,
    (False, "transformerlens"): ITLensModule,
    (True, "transformerlens"): ITLensLightningModule,
}

class ITMeta(type):
    def __new__(mcs, name, bases, classdict, **kwargs):
        component, input, ctx = mcs._validate_build_ctx(kwargs)
        # TODO: add runtime checks for adherence to IT protocol here?
        composition_class = mcs._map_composition_target(component, ctx)
        bases = (NamedWrapper, input, composition_class)
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
        lightning = ctx.get('lightning', False)
        plugin = ctx.get('plugin', None)
        if component in ('module', 'm'):
            composition_class = INTERPRETUNABLE_MODULE_MAPPING[(lightning, plugin)]
        else:
            composition_class = ITDataModule if not lightning else ITLightningDataModule
        return composition_class


@dataclass(kw_only=True)
class UnencapsulatedArgs(ITSerializableCfg):
    # Most use cases will encapsulate datamodule/module config by subclassing the relevant dataclasses for a given
    # experiment/application but we also allow unencapsulated args/kwargs to be passed to the datamodule and module
    dm_args: Tuple = ()
    dm_kwargs: Dict[str, Any] = field(default_factory=dict)
    module_args: Tuple = ()
    module_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class InterpretunableSessionConfig(UnencapsulatedArgs):
    datamodule: Optional[ITDataModuleProtocol] = None
    module: Optional[ITModuleProtocol] = None
    datamodule_cfg: ITDataModuleConfig
    datamodule_cls: str | DataModuleInitable
    module_cfg: ITConfig
    module_cls: str | ModuleSteppable
    lightning: bool = False
    plugin: Optional[str] = None

    def __post_init__(self):
        # N.B. we must defer validating data/module_cls against their respective protocols until after composing
        # TODO: add warnings/errors here in case provided classes cannot be resolved or are invalid
        for attr_k in ["datamodule_cls", "module_cls"]:
            if isinstance(getattr(self, attr_k), str):
                module, module_cls = getattr(self, attr_k).rsplit(".", 1)
                mod = importlib.import_module(module)
                setattr(self, attr_k, getattr(mod, module_cls, None))
        # to improve usability, run a datamodule cross-validation hook to allow auto-reconfiguration prior to
        # session instantiation
        self.datamodule_cfg._cross_validate(self.module_cfg)
        self.compose_interpretunable()

    def _check_ready(self):
        ready_module_attrs = ["datamodule", "module"]
        composition_target_attrs = ["datamodule_cls", "module_cls"]
        ready_protocols = [ITDataModuleProtocol, ITModuleProtocol]
        for ready, tocompose, ready_type in zip(ready_module_attrs, composition_target_attrs, ready_protocols):
            if ready_mod := getattr(self, ready):  # if a module is ready, ensure we don't try to transform it
                if not isinstance(ready_mod, ready_type):
                    raise ValueError(f"{ready_mod} is not a {ready_type}, {ready_mod} should either be left unset (to "
                                     f"enable auto-composition of {getattr(self, tocompose)}) or be a {ready_type}.")
                setattr(self, tocompose, None)  # TODO: need to add tests for this functionality

    def compose_interpretunable(self) -> None:
        # 1. We first check to see if the datamodule and module classes are already defined and adhere to the relevant
        #    protocol.
        self._check_ready()
        # 2. For the components (i.e. datamodule and module) that aren't ready, we compose the provided input component
        #    with the relevant classes based on the specified execution context.
        build_ctx = {'lightning': self.lightning, 'plugin': self.plugin}
        dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm', input=self.datamodule_cls, ctx=build_ctx)
        m_cls = ITMeta('InterpretunableModule', (), {}, component='m', input=self.module_cls, ctx=build_ctx)
        # 3. We instantiate our Interpretunably enriched components!
        if dm_cls:
            self.datamodule = dm_cls(itdm_cfg=self.datamodule_cfg, *self.dm_args, **self.dm_kwargs)
        self._set_dm_handles_for_instantiation()
        if m_cls:
            self.module = m_cls(it_cfg=self.module_cfg, *self.module_args, **self.module_kwargs)
        self._set_model_handles_for_instantiation()
        self._validate_session(dm_cls, m_cls)

    def _set_dm_handles_for_instantiation(self):
        # some datamodule handles may be required for module init, we update the module_cfg to provide them here
        supported_dm_handles_for_module = {"tokenizer": "tokenizer"}
        for m_attr, dm_handle in supported_dm_handles_for_module.items():
            setattr(self.module_cfg, m_attr, getattr(self.datamodule, dm_handle))

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

    def _validate_session(self, dm_composed, m_composed):
        if not isinstance(self.datamodule, ITDataModuleProtocol):
            rank_zero_warn(f"{self.datamodule} is not {ITDataModuleProtocol} even after composing"
                           f"{self.datamodule_cls} with {dm_composed}. {unexpected_state_msg_suffix}")
        if not isinstance(self.module, ITModuleProtocol):
            rank_zero_warn(f"{self.module} is not {ITModuleProtocol} even after composing {self.module_cls} with"
                           f"{m_composed}. {unexpected_state_msg_suffix}")
