from unittest.mock import patch, create_autospec
from dataclasses import dataclass
from itertools import product
import pytest
from copy import deepcopy
from enum import auto, IntEnum


from interpretune.base.modules import ITModule, BaseITModule
from interpretune.base.datamodules import ITDataModule
from interpretune.plugins.transformer_lens import ITLensModule
from tests.parity_acceptance.base.test_it_base import CoreCfg, ProfParityCfg, BaseCfg
from tests.parity_acceptance.plugins.transformer_lens.test_interpretune_tl import TLParityCfg, TLProfileCfg
from interpretune.base.call import _call_itmodule_hook
from tests.configuration import config_modules, get_it_cfg, get_itdm_cfg, config_session
from tests.modules import TestITDataModule, TestITModule
from tests.unit.cfg_aliases import (TLDebugCfg, LightningLlama2DebugCfg, CoreMemProfCfg, CoreGPT2PEFTCfg,
                                    CoreGPT2PEFTSeqCfg, CoreCfgForcePrepare)
from interpretune.utils.import_utils import _DOTENV_AVAILABLE

if _DOTENV_AVAILABLE:
    from dotenv import load_dotenv
else:
    load_dotenv = lambda: None



# NOTE [Datamodule/Module/Session Fixture Caching]:
# We note when instantiating module and datamodule manually (as is done in our datamodule/module fixture factories)
# before passing modules to ITSession, the module won't have a handle to the datamodule so any logic that relies
# on a datamodule attribute (like tokenizer) will fail. This prevents us from reusing any existing datamodule/
# module fixture for a full session without exploding the number of different fixtures we create/cache by
# creating some module fixtures with module definitions but not instantiation.
#
# TODO: If the resource parsimony is worth the added complexity, to address the above, we could patch the
# module fixtures with a mock tokenizer during instantiation until we can replace the mock with a real tokenizer
# during ITSession construction.


# TODO: switch to namedtuple if not subclassing this in the future
@dataclass(kw_only=True)
class FixtureCfg:
    test_cfg: BaseCfg = CoreCfg
    module_cls: BaseITModule = ITModule
    datamodule_cls: ITDataModule = ITDataModule

FIXTURE_CFGS = {
    "core_cust": FixtureCfg(),
    "core_cust_force_prepare": FixtureCfg(test_cfg=CoreCfgForcePrepare),
    "core_gpt2": FixtureCfg(test_cfg=ProfParityCfg),
    "core_gpt2_peft": FixtureCfg(test_cfg=CoreGPT2PEFTCfg),
    "core_gpt2_peft_seq": FixtureCfg(test_cfg=CoreGPT2PEFTSeqCfg),
    "core_cust_memprof": FixtureCfg(test_cfg=CoreMemProfCfg),
    "l_llama2_debug": FixtureCfg(test_cfg=LightningLlama2DebugCfg),
    "tl_cust": FixtureCfg(test_cfg=TLParityCfg, module_cls=ITLensModule),
    "tl_gpt2": FixtureCfg(test_cfg=TLProfileCfg, module_cls=ITLensModule),
    "tl_gpt2_debug": FixtureCfg(test_cfg=TLDebugCfg, module_cls=ITLensModule),
}

class FixturePhase(IntEnum):
    initonly: int = auto()
    prepare_data: int = auto()
    setup: int = auto()
    configure_optimizers: int = auto()

@pytest.fixture(scope="class")
def mock_dm():
    # this mock fixture is necessary because many tests will want a mock tokenizer but being a dynamic attribute,
    # the tokenizer isn't generated with autospec. We therefore attach a mock tokenizer to our mock datamodule here
    dm_cls = type('InterpretunableDataModule', (TestITDataModule, ITDataModule), {})
    with patch("transformers.PreTrainedTokenizer", autospec=True) as mock_tok:
        mock_uninit_dm = create_autospec(dm_cls)
        mock_uninit_dm.tokenizer = mock_tok
        yield mock_uninit_dm

@pytest.fixture(scope="class")
def make_it_datamodule():
    def __make_it_datamodule(datamodule_key):
        test_cfg = FIXTURE_CFGS[datamodule_key].test_cfg()
        dm_kwargs = {'force_prepare_data': test_cfg.force_prepare_data}
        itdm_cfg = get_itdm_cfg(test_cfg=test_cfg, dm_override_cfg=test_cfg.dm_override_cfg)
        dm_cls = type('InterpretunableDataModule', (TestITDataModule, FIXTURE_CFGS[datamodule_key].datamodule_cls), {})
        it_dm = dm_cls(itdm_cfg, **dm_kwargs)
        return it_dm
    yield __make_it_datamodule

def datamodule_fixture_factory(datamodule_key):
    @pytest.fixture(scope="class")
    def get_it_datamodule(make_it_datamodule):
        it_dm = make_it_datamodule(datamodule_key)
        if init_key in ("setup", "prepare_data"):
            with patch("tests.modules.TestITModule", autospec=True) as mock_m:
                _call_itmodule_hook(it_dm, hook_name="prepare_data", hook_msg="Preparing data",
                                    target_model=mock_m.model)
        if init_key == "setup":
            _call_itmodule_hook(it_dm, hook_name="setup", hook_msg="Setting up datamodule")
        yield it_dm
    return get_it_datamodule

@pytest.fixture(scope="class")
def make_it_module(tmp_path_factory):
    def __make_it_module(module_key, init_key):
        m_kwargs = {'test_alias': f"{module_key}_{init_key}_it_m_fixture", 'state_log_dir': None}
        it_cfg = get_it_cfg(test_cfg=FIXTURE_CFGS[module_key].test_cfg(),
                            core_log_dir=tmp_path_factory.mktemp(f"{module_key}_{init_key}_it_m_fixture"))
        m_cls = type('InterpretunableModule', (TestITModule, FIXTURE_CFGS[module_key].module_cls), {})
        it_m = m_cls(it_cfg, **m_kwargs)
        return it_m
    yield __make_it_module

def module_fixture_factory(module_key, init_key):
    @pytest.fixture(scope="class")
    def get_it_module(make_it_module, mock_dm):
        it_m = make_it_module(module_key, init_key)
        if init_key == "setup":
            _call_itmodule_hook(it_m, hook_name="setup", hook_msg="Setting up model", datamodule=mock_dm)
        yield it_m
    return get_it_module

def session_fixture_hook_exec(it_s, init_key: FixturePhase):
    if init_key.value > FixturePhase.initonly:  # call appropriate init phases if requested
        if init_key.value >= FixturePhase.prepare_data:
            _call_itmodule_hook(it_s.datamodule, hook_name="prepare_data", hook_msg="Preparing data",
                            target_model=it_s.module.model)
        if init_key.value >= FixturePhase.setup:
            _call_itmodule_hook(it_s.datamodule, hook_name="setup", hook_msg="Setting up datamodule",
                                module=it_s.module)
            _call_itmodule_hook(it_s.module, hook_name="setup", hook_msg="Setting up model",
                                datamodule=it_s.datamodule)
        if init_key.value >= FixturePhase.configure_optimizers:
            _call_itmodule_hook(it_s.module, hook_name="configure_optimizers",
                                hook_msg="initializing optimizers and schedulers", connect_output=True)
def session_fixture_factory(config_key, init_key):
    @pytest.fixture(scope="class")
    def get_it_session(tmp_path_factory):
        load_dotenv()  # load env vars from .env file # TODO: make a diff fixture?
        test_sess_config = FIXTURE_CFGS[config_key].test_cfg
        it_s = config_modules(test_sess_config(), f"{config_key}_{init_key}_it_session_fixture", {},
                              tmp_path_factory.mktemp(f"{config_key}_{init_key}_it_session_fixture"), {}, False)
        session_fixture_hook_exec(it_s, FixturePhase[init_key])
        setattr(it_s, 'fixt_test_cfg', deepcopy(test_sess_config))
        yield it_s
    return get_it_session

def configure_session_cfg_fixture(test_cfg_cls, tmp_factory):
    test_cfg = test_cfg_cls()
    itdm_cfg = get_itdm_cfg(test_cfg=test_cfg, dm_override_cfg=test_cfg.dm_override_cfg)
    it_cfg = get_it_cfg(test_cfg=test_cfg, core_log_dir=tmp_factory.mktemp(f"{test_cfg_cls.__name__}_sess_cfg_fixture"))
    TEST_CLS_MAPPING = {'datamodule_cls': 'tests.modules.TestITDataModule', 'module_cls': 'tests.modules.TestITModule'}
    core_cfg = {'datamodule_cfg': itdm_cfg, 'module_cfg': it_cfg, **TEST_CLS_MAPPING}
    return core_cfg, test_cfg

@pytest.fixture(scope="class")
def get_tl_it_session_cfg(tmp_path_factory):
    core_cfg, test_cfg = configure_session_cfg_fixture(TLParityCfg, tmp_path_factory)
    test_cfg.framework_ctx = 'core'
    test_cfg.plugin_ctx = 'transformer_lens'
    test_cfg.model_src_key = 'cust'
    yield config_session(core_cfg, test_cfg, 'it_session_cfg_tl_test', {}, None, {})

@pytest.fixture(scope="class")
def get_core_cust_it_session_cfg(tmp_path_factory):
    core_cfg, test_cfg = configure_session_cfg_fixture(CoreCfg, tmp_path_factory)
    test_cfg.framework_ctx = 'core'
    test_cfg.model_src_key = 'cust'
    yield config_session(core_cfg, test_cfg, 'it_session_cfg_core_test', {}, None, {})

for module_key, init_key in product(FIXTURE_CFGS.keys(), ["setup", "configure_optimizers"]):
    name = f"get_it_module__{module_key}__{init_key}"
    globals()[name] = module_fixture_factory(module_key, init_key)
    # overwrite just the name/qual name attributes for pytest
    globals()[name].__name__ = name
    globals()[name].__qualname__ = name

for datamodule_key, init_key in product(FIXTURE_CFGS.keys(), ["prepare_data", "setup"]):
    name = f"get_it_datamodule__{datamodule_key}__{init_key}"
    globals()[name] = datamodule_fixture_factory(datamodule_key)
    # overwrite just the name/qual name attributes for pytest
    globals()[name].__name__ = name
    globals()[name].__qualname__ = name

for session_key, init_key in product(FIXTURE_CFGS.keys(), ["initonly", "setup", "configure_optimizers"]):
    name = f"get_it_session__{session_key}__{init_key}"
    globals()[name] = session_fixture_factory(session_key, init_key)
    # overwrite just the name/qual name attributes for pytest
    globals()[name].__name__ = name
    globals()[name].__qualname__ = name
