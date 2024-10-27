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
# initially based on: https://bit.ly/3GDHDcI
import os
import threading
import random
from collections.abc import Iterable
from collections import defaultdict
from typing import Dict, Tuple
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

from unittest.mock import patch, create_autospec
from dataclasses import dataclass
from itertools import product
from copy import deepcopy
from enum import auto, IntEnum
import pytest
import yaml
import torch.distributed

from interpretune.adapters.registration import Adapter
from interpretune.base.call import _call_itmodule_hook
from interpretune.base.contract.session import ITMeta
from interpretune.base.contract.protocol import ModuleSteppable, DataModuleInitable
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.logging import rank_zero_only
from interpretune.utils.types import StrOrPath
from tests import _PATH_DATASETS, seed_everything, load_dotenv, FinetuningScheduler, get_fts, Trainer
from tests.configuration import (config_modules, get_it_cfg, get_itdm_cfg, config_session, TEST_IT_DATAMODULE,
                                 TEST_IT_MODULE)
from tests.modules import TestITDataModule
from tests.parity_acceptance.cfg_aliases import parity_cli_cfgs, CLI_EXP
from tests.parity_acceptance.test_it_cli import TEST_CONFIGS_CLI_PARITY
from tests.parity_acceptance.test_it_l import CoreCfg, ProfParityCfg, BaseCfg
from tests.parity_acceptance.test_it_tl import TLParityCfg, TLProfileCfg
from tests.unit.cfg_aliases import (TEST_CONFIGS_CLI_UNIT, unit_exp_cli_cfgs, TLDebugCfg,
    LightningLlama3DebugCfg, CoreMemProfCfg, CoreGPT2PEFTCfg, CoreGPT2PEFTSeqCfg,
    CoreCfgForcePrepare, LightningGPT2, LightningTLGPT2, TLMechInterpCfg)


test_cli_cfgs = deepcopy(parity_cli_cfgs)
test_cli_cfgs['exp_cfgs'].update(unit_exp_cli_cfgs)

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
    module_cls: ModuleSteppable = TEST_IT_MODULE
    datamodule_cls: DataModuleInitable = TEST_IT_DATAMODULE
    scope: str = "class"

FIXTURE_CFGS = {
    "core_cust": FixtureCfg(scope="session"),
    "core_cust_force_prepare": FixtureCfg(test_cfg=CoreCfgForcePrepare),
    "core_gpt2": FixtureCfg(test_cfg=ProfParityCfg),
    "core_gpt2_peft": FixtureCfg(test_cfg=CoreGPT2PEFTCfg),
    "core_gpt2_peft_seq": FixtureCfg(test_cfg=CoreGPT2PEFTSeqCfg),
    "core_cust_memprof": FixtureCfg(test_cfg=CoreMemProfCfg),
    "l_gpt2": FixtureCfg(test_cfg=LightningGPT2, scope="function"),
    "l_tl_gpt2": FixtureCfg(test_cfg=LightningTLGPT2, scope="function"),
    "l_llama3_debug": FixtureCfg(test_cfg=LightningLlama3DebugCfg),
    "tl_cust": FixtureCfg(test_cfg=TLParityCfg, scope="session"),
    "tl_cust_mi": FixtureCfg(test_cfg=TLMechInterpCfg, scope="function"),
    "tl_gpt2": FixtureCfg(test_cfg=TLProfileCfg),
    "tl_gpt2_debug": FixtureCfg(test_cfg=TLDebugCfg),
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
        dm_cls = ITMeta('InterpretunableDataModule', (), {}, component='dm',
                        input=FIXTURE_CFGS[datamodule_key].datamodule_cls, ctx=test_cfg.adapter_ctx)
        it_dm = dm_cls(itdm_cfg=itdm_cfg, **dm_kwargs)
        return it_dm
    yield __make_it_datamodule

# TODO: not currently used, may refactor and remove if not used in the near future
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
        test_cfg=FIXTURE_CFGS[module_key].test_cfg()
        core_log_dir = tmp_path_factory.mktemp(f"{module_key}_{init_key}_it_m_fixture")
        it_cfg = get_it_cfg(test_cfg=test_cfg, core_log_dir=core_log_dir)
        m_cls = ITMeta('InterpretunableModule', (), {}, component='m',
                       input=FIXTURE_CFGS[module_key].module_cls, ctx=test_cfg.adapter_ctx)
        it_m = m_cls(it_cfg=it_cfg, **m_kwargs)
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

def it_session_fixture_factory(config_key, init_key):
    @pytest.fixture(scope=FIXTURE_CFGS[config_key].scope)
    def get_it_session(tmp_path_factory):
        load_dotenv()  # load env vars from .env file # TODO: make a diff fixture?
        test_sess_config = FIXTURE_CFGS[config_key].test_cfg
        it_s = config_modules(test_sess_config(), f"{config_key}_{init_key}_it_session_fixture", {},
                              tmp_path_factory.mktemp(f"{config_key}_{init_key}_it_session_fixture"), {}, False)
        session_fixture_hook_exec(it_s, FixturePhase[init_key])
        setattr(it_s, 'fixt_test_cfg', deepcopy(test_sess_config))
        yield it_s
    return get_it_session

def configure_session_cfg(test_cfg_cls, tmp_path_or_factory):
    test_cfg = test_cfg_cls()
    if isinstance(tmp_path_or_factory, StrOrPath):
        tmp_log_dir = tmp_path_or_factory
    else:
        tmp_log_dir = tmp_path_or_factory.mktemp(f"{test_cfg_cls.__name__}_sess_cfg_fixture")
    itdm_cfg = get_itdm_cfg(test_cfg=test_cfg, dm_override_cfg=test_cfg.dm_override_cfg)
    it_cfg = get_it_cfg(test_cfg=test_cfg, core_log_dir=tmp_log_dir)
    TEST_CLS_MAPPING = {'datamodule_cls': 'tests.modules.TestITDataModule', 'module_cls': 'tests.modules.TestITModule'}
    core_cfg = {'datamodule_cfg': itdm_cfg, 'module_cfg': it_cfg, **TEST_CLS_MAPPING}
    return core_cfg, test_cfg

@pytest.fixture(scope="class")
def get_tl_it_session_cfg(tmp_path_factory):
    core_cfg, test_cfg = configure_session_cfg(TLParityCfg, tmp_path_factory)
    test_cfg.adapter_ctx = (Adapter.core, Adapter.transformer_lens)
    test_cfg.model_src_key = 'cust'
    yield config_session(core_cfg, test_cfg, 'it_session_cfg_tl_test', {}, None, {})

@pytest.fixture(scope="class")
def get_core_cust_it_session_cfg(tmp_path_factory):
    core_cfg, test_cfg = configure_session_cfg(CoreCfg, tmp_path_factory)
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
    globals()[name] = it_session_fixture_factory(session_key, init_key)
    # overwrite just the name/qual name attributes for pytest
    globals()[name].__name__ = name
    globals()[name].__qualname__ = name

# if additional CLI test configurations are needed, append them here and the `cli_test_configs` fixture will generate
# them sessionwise
GEN_CLI_CFGS = [TEST_CONFIGS_CLI_PARITY, TEST_CONFIGS_CLI_UNIT]

def gen_experiment_cfg_sets(test_keys: Iterable[Tuple[str, str, bool]], sess_paths: Tuple) -> Dict:
    EXPERIMENTS_BASE, BASE_DEBUG_CONFIG = sess_paths
    exp_cfg_sets = {}
    for exp, subexp, debug_mode in test_keys:
        subexp_cfg =  EXPERIMENTS_BASE / f"{subexp}.yaml"
        exp_cfg_sets[(exp, subexp, debug_mode)] = (subexp_cfg, BASE_DEBUG_CONFIG) if debug_mode else (subexp_cfg,)
    return exp_cfg_sets

@pytest.fixture(scope="function")
def fts_patch_env():
    os.environ["FTS_GEN_SCHEDULE_ALLOW_DUPLICATE"] = "True"
    yield
    for env_key in ("FTS_GEN_SCHEDULE_ALLOW_DUPLICATE",):
        if env_key in os.environ:
            del os.environ[env_key]

@pytest.fixture(scope="function")
def clean_cli_env():
    yield
    for env_key in ("IT_GLOBAL_SEED",):
        if env_key in os.environ:
            del os.environ[env_key]

@pytest.fixture(scope="session")
def cli_test_file_env(tmp_path_factory):
    os.environ["IT_CONFIG_BASE"] = str(tmp_path_factory.mktemp("test_cli_files"))
    sess_cfg_base = Path(os.environ["IT_CONFIG_BASE"])
    EXPERIMENTS_BASE = sess_cfg_base / "experiments" / CLI_EXP
    IT_CONFIG_DEFAULTS = sess_cfg_base / "global" / "defaults"
    os.environ["IT_CONFIG_DEFAULTS"] = str(IT_CONFIG_DEFAULTS)
    IT_CONFIG_GLOBAL = sess_cfg_base / "global"
    TEST_CLI_CONFIG_FILES = {
        "global_debug": (IT_CONFIG_GLOBAL, "base_debug.yaml", test_cli_cfgs["global_debug"]),
        "global_defaults": (IT_CONFIG_DEFAULTS, "default.yaml", test_cli_cfgs["global_defaults"]),
        "exp_cfgs": test_cli_cfgs["exp_cfgs"],
    }
    yield TEST_CLI_CONFIG_FILES, EXPERIMENTS_BASE
    for env_key in ("IT_CONFIG_BASE", "IT_CONFIG_DEFAULTS", "WANDB_API_KEY", "HF_GATED_PUBLIC_REPO_AUTH_KEY",
                    "IDE_PROJECT_ROOTS"):
        if env_key in os.environ:
            del os.environ[env_key]

def ensure_path(cfg_dir):
    cfg_dir = Path(cfg_dir)
    cfg_dir.mkdir(exist_ok=True, parents=True)
    return cfg_dir

def write_cfg_to_yaml_file(path, config):
    with open(path, "w") as f:
        f.write(yaml.dump(config))

def write_cli_config_files(test_cli_cfg_files, test_exp_model_dir):
    test_config_files = {}
    for k, v in test_cli_cfg_files.items():  # experiment-level files
        if k == "exp_cfgs":
            for k, v in v.items():
                cfg_dir = ensure_path(test_exp_model_dir)
                cfg_file_path = cfg_dir / f"{k.value}.yaml"
                write_cfg_to_yaml_file(cfg_file_path, v)
                test_config_files[k] = cfg_file_path
        else:
            cfg_dir = ensure_path(Path(v[0]))
            cfg_file_path = cfg_dir / v[1]
            write_cfg_to_yaml_file(cfg_file_path, v[2])
            test_config_files[k] = cfg_file_path
    return test_config_files

@pytest.fixture(scope="session")
def cli_test_configs(cli_test_file_env):
    # this fixture will collect all required cli test configuration files and dynamically generate them sessionwise
    test_cli_cfg_files, experiments_base = cli_test_file_env
    test_config_files = write_cli_config_files(test_cli_cfg_files, experiments_base)
    sess_paths = Path(experiments_base), Path(test_config_files["global_debug"])
    # we specify a set of CLI test configurations (GEN_CLI_CFGS) to drive our CLI configuration file generation
    test_keys = ((CLI_EXP, c.alias, c.cfg.debug_mode) for cfg in GEN_CLI_CFGS for c in cfg)
    EXPERIMENT_CFG_SETS = gen_experiment_cfg_sets(test_keys=test_keys, sess_paths=sess_paths)
    yield EXPERIMENT_CFG_SETS

def l_imp_to_exp(sched_dict: Dict) -> Dict:
    sched_dict[0]["params"] = [r"model.transformer.h.(9|1[0-1]).(mlp|attn|ln_(1|2)).(c_proj|c_fc|c_attn|weight|bias).*"]
    sched_dict[0]["max_transition_epoch"] = 2
    phase_1_pats = [r"model.transformer.h.([0-8](?!\d)).(mlp|attn|ln_(1|2)).(c_proj|c_fc|c_attn|weight|bias).*",
     r"model.transformer.(wpe|wte).weight", r"model.transformer.ln_f.*"]
    sched_dict[1]["params"] = phase_1_pats
    sched_dict[1]["lr"] = 1e-06
    sched_dict = {phase:phase_def for phase, phase_def in sched_dict.items() if phase in range(2)}
    return sched_dict

def tl_imp_to_exp(sched_dict: Dict) -> Dict:
    sched_dict[0]["params"] = [r"model.blocks.(9|1[0-1]).*"]
    sched_dict[0]["max_transition_epoch"] = 2
    phase_1_pats = [r"model.blocks.([0-8](?!\d)).*", r"model.(pos_embed|embed).*", "model.unembed.W_U",
                    "model.unembed.b_U"]
    sched_dict[1]["params"] = phase_1_pats
    sched_dict[1]["lr"] = 1e-06
    sched_dict = {phase:phase_def for phase, phase_def in sched_dict.items() if phase in range(2)}
    return sched_dict

@pytest.fixture(scope="function")
def gpt2_ft_schedules(tmpdir_factory, fts_patch_env, get_it_session__l_gpt2__setup,
                      get_it_session__l_tl_gpt2__setup) -> Tuple[Path, Dict]:
    """Generates a default fine-tuning schedule for 'implicit' testing, a modified one for 'explicit' mode and an
    epoch-driven transitions only one for epoch_transitions_only testing."""
    SCHED_TRANSFORMS = defaultdict(dict)
    SCHED_TRANSFORMS["l_gpt2"]["basic_explicit"] = l_imp_to_exp
    SCHED_TRANSFORMS["l_tl_gpt2"]["basic_explicit"] = tl_imp_to_exp
    seed_everything(42)
    callbacks = [FinetuningScheduler(gen_ft_sched_only=True)]
    # for simplicity, initially only running FTS non-distributed tests
    tmpdir = tmpdir_factory.mktemp("test_fts_schedules")
    rank = getattr(rank_zero_only, "rank", 0)
    models = {"l_gpt2": deepcopy(get_it_session__l_gpt2__setup.module),
              "l_tl_gpt2": deepcopy(get_it_session__l_tl_gpt2__setup.module)}
    test_schedules = defaultdict(dict)
    for i, (model_key, model) in enumerate(models.items()):
        trainer = Trainer(default_root_dir=tmpdir, callbacks=callbacks, devices=1)
        unmod_schedule_file = \
              tmpdir / "lightning_logs" / f"version_{i}" / f"{model.__class__.__name__}_ft_schedule.yaml"
        # N.B. Though we run this fixture for each rank to avoid adding special logic to each distributed client test,
        # we only generate a schedule on rank 0, linking to it on the other ranks.
        if rank == 0:
            with pytest.raises(SystemExit):
                trainer.fit(model)
        test_schedules[model_key]["implicit"] = get_fts(trainer).load_yaml_schedule(unmod_schedule_file)
        for transform_key, transform_fn in SCHED_TRANSFORMS[model_key].items():
            test_schedules[model_key][transform_key] = transform_fn(deepcopy(test_schedules[model_key]["implicit"]))
    return test_schedules



@pytest.fixture(scope="function")
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="function")
def make_deterministic(warn_only=False, fill_uninitialized_memory=True):
    # https://pytorch.org/docs/2.3/notes/randomness.html#reproducibility
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch._C._set_deterministic_fill_uninitialized_memory(fill_uninitialized_memory)
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


@pytest.fixture(scope="function", autouse=True)
def preserve_global_rank_variable():
    from interpretune.utils.logging import rank_zero_only
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


@pytest.fixture(scope="function", autouse=True)
def restore_env_variables():
    """Ensures that environment variables set during the test do not leak out."""
    okay_session_scope_keys = {"IT_CONFIG_BASE", "IT_CORE_SHARED", "IT_LIGHTNING_SHARED", "WANDB_API_KEY",
                               "HF_GATED_PUBLIC_REPO_AUTH_KEY", "IDE_PROJECT_ROOTS"}
    env_backup = os.environ.copy()
    yield
    leaked_vars = os.environ.keys() - env_backup.keys()
    # restore environment as it was before running the test
    os.environ.clear()
    os.environ.update(env_backup)
    # these are currently known leakers - ideally these would not be allowed
    allowlist = {
        "CUBLAS_WORKSPACE_CONFIG",  # enabled with deterministic flag
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "WANDB_MODE",
        "WANDB_REQUIRE_SERVICE",
        "WANDB_SERVICE",
        "HOROVOD_FUSION_THRESHOLD",  # set by HorovodStrategy # TODO: remove in v2.0.0
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
        "CUDA_MODULE_LOADING",  # leaked since PyTorch 1.13
        "KMP_INIT_AT_FORK",  # leaked since PyTorch 1.13
        "KMP_DUPLICATE_LIB_OK",  # leaked since PyTorch 1.13
        "CRC32C_SW_MODE",  # leaked by tensorboardX
        "TRITON_CACHE_DIR",  # leaked starting in PyTorch 2.0.0
        "OMP_NUM_THREADS",  # leaked by Lightning launchers,
        "TOKENIZERS_PARALLELISM",  # TODO: add a fixture that resets this currently leaked var
    }
    allowlist.update(okay_session_scope_keys)
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"

@pytest.fixture(scope="function", autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

@pytest.fixture
def tmpdir_server(tmpdir):
    Handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
    from http.server import ThreadingHTTPServer

    with ThreadingHTTPServer(("localhost", 0), Handler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        yield server.server_address
        server.shutdown()


# NOTE [Profiling and Standalone Marks]:
# - CI doesn't run all `profiling` marked tests by default, only the subset of profiling tests that are marked both
#   `profiling` and `profiling_ci`
# - The standalone marks run with CI by default and take precedence over profiling marks
# - To run all profiling tests, set `IT_RUN_PROFILING_TESTS` to `2`

def pytest_collection_modifyitems(items):
    # select special tests, all special tests run standalone
    # non-specific standalone tests and profiling_ci tests run in CI by default
    # all other special tests do not run in CI unless explicitly selected
    if os.getenv("IT_RUN_STANDALONE_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(standalone=True)`
            if marker.name == "skipif" and marker.kwargs.get("standalone")
        ]
    elif os.getenv("IT_RUN_PROFILING_TESTS", "0") == "2":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(profiling=True)`
            if marker.name == "skipif" and marker.kwargs.get("profiling")
        ]
    elif os.getenv("IT_RUN_PROFILING_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(profiling_ci=True)`
            if marker.name == "skipif" and marker.kwargs.get("profiling_ci")
        ]
    elif os.getenv("IT_RUN_OPTIONAL_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(optional=True)`
            if marker.name == "skipif" and marker.kwargs.get("optional")
        ]
