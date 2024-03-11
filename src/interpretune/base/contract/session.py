from typing import Any, Dict, Generic, Optional, Tuple

import importlib
from dataclasses import dataclass, field

from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig, ITSerializableCfg
from interpretune.base.modules import ITModule, ITLightningModule
from interpretune.base.datamodules import ITLightningDataModule, ITDataModule
from interpretune.plugins.transformer_lens import ITLensLightningModule, ITLensModule
from interpretune.utils.warnings import unexpected_state_msg_suffix
from interpretune.utils.logging import rank_zero_warn
from interpretune.base.contract.protocol import (DataModuleInitable, ModuleSteppable, T_dm, T_m, InterpretunableTuple,
                                                 NamedWrapper, InterpretunableModule, InterpretunableDataModule)


def interpretunable_factory(datamodule_wrapper: Generic[T_dm], module_wrapper: Generic[T_m],
                            datamodule: Optional[DataModuleInitable] = None, module: Optional[ModuleSteppable] = None) \
                                -> InterpretunableTuple:
    built_datamodule = None
    built_module = None
    if datamodule:
        class InterpretunableDataModule(NamedWrapper, datamodule, datamodule_wrapper):
            _orig_module_name = datamodule.__qualname__
        built_datamodule = InterpretunableDataModule
    if module:
        class InterpretunableModule(NamedWrapper, module, module_wrapper):
            _orig_module_name = module.__qualname__
        built_module = InterpretunableModule
    return InterpretunableTuple(datamodule=built_datamodule, module=built_module)

# NOTE [Interpretability Plugins]:
# `TransformerLens` is the first supported interpretability plugin but support for other plugins is expected

INTERPRETUNABLE_MODULE_MAPPING = {
    # (lightning, plugin_key)
    (False, None): ITModule,
    (True, None): ITLightningModule,
    (False, "transformerlens"): ITLensModule,
    (True, "transformerlens"): ITLensLightningModule,
}

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
    datamodule: Optional[InterpretunableDataModule] = None
    module: Optional[InterpretunableModule] = None
    datamodule_cfg: ITDataModuleConfig
    datamodule_cls: str | DataModuleInitable
    module_cfg: ITConfig
    module_cls: str | ModuleSteppable
    lightning: bool = False
    plugin: Optional[str] = None

    def __post_init__(self):
        # N.B. we must defer validating data/module_cls against their respective protocols until after wrapping
        # TODO: add warnings/errors here in case provided classes cannot be resolved or are invalid
        for attr_k in ["datamodule_cls", "module_cls"]:
            if isinstance(getattr(self, attr_k), str):
                module, module_cls = getattr(self, attr_k).rsplit(".", 1)
                mod = importlib.import_module(module)
                setattr(self, attr_k, getattr(mod, module_cls, None))
        # to improve usability, run a datamodule cross-validation hook to allow auto-reconfiguration prior to
        # session instantiation
        self.datamodule_cfg._cross_validate(self.module_cfg)
        self.wrap_interpretunable()

    def _check_ready(self):
        ready_module_attrs = ["datamodule", "module"]
        towrap_module_attrs = ["datamodule_cls", "module_cls"]
        ready_class_types = [InterpretunableDataModule, InterpretunableModule]
        for ready, towrap, ready_type in zip(ready_module_attrs, towrap_module_attrs, ready_class_types):
            if ready_mod := getattr(self, ready):  # if a module is ready, ensure we don't try to wrap it
                if not isinstance(ready_mod, ready_type):
                    raise ValueError(f"{ready_mod} is not a {ready_type}, {ready_mod} should either be left unset (to "
                                     f"enable auto-wrapping of {getattr(self, towrap)}) or be a {ready_type}.")
                setattr(self, towrap, None)  # TODO: need to add tests for this functionality

    def wrap_interpretunable(self) -> None:
        self._check_ready()
        dm_wrapper = ITDataModule if not self.lightning else ITLightningDataModule
        m_wrapper = INTERPRETUNABLE_MODULE_MAPPING[(self.lightning, self.plugin)]
        dm_cls, m_cls = interpretunable_factory(dm_wrapper, m_wrapper, self.datamodule_cls, self.module_cls)
        # instantiate the wrapped datamodule and/or module classes and attach them to the session
        if dm_cls:
            self.datamodule = dm_cls(itdm_cfg=self.datamodule_cfg, *self.dm_args, **self.dm_kwargs)
        self._set_dm_handles_for_instantiation()
        if m_cls:
            self.module = m_cls(it_cfg=self.module_cfg, *self.module_args, **self.module_kwargs)
        self._validate_session(dm_wrapper, m_wrapper)

    def _set_dm_handles_for_instantiation(self):
        # some datamodule handles may be required for module init, we update the module_cfg to provide them here
        supported_dm_handles_for_module = {"tokenizer": "tokenizer"}
        for m_attr, dm_handle in supported_dm_handles_for_module.items():
            setattr(self.module_cfg, m_attr, getattr(self.datamodule, dm_handle))

    def _validate_session(self, dm_wrapper, m_wrapper):
        if not isinstance(self.datamodule, InterpretunableDataModule):
            rank_zero_warn(f"{self.datamodule} is not {InterpretunableDataModule} even after wrapping"
                           f"{self.datamodule_cls} with {dm_wrapper}. {unexpected_state_msg_suffix}")
        if not isinstance(self.module, InterpretunableModule):
            rank_zero_warn(f"{self.module} is not {InterpretunableModule} even after wrapping {self.module_cls} with"
                           f"{m_wrapper}. {unexpected_state_msg_suffix}")
