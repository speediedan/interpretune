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
from typing import Iterable,Dict, Tuple, Optional, List
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
import random

import pytest
import yaml
import torch.distributed
from tests import _PATH_DATASETS
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.base.components.cli import core_cli_main, compose_config
from interpretune.base.contract.session import Framework, Plugin
from tests.parity_acceptance.cli.cfg_aliases import cli_cfgs, CLI_EXP_MODEL, RUN_FN, TEST_CONFIGS_CLI_PARITY
from tests.unit.cfg_aliases import TEST_CONFIGS_CLI_UNIT

if _LIGHTNING_AVAILABLE:
    from interpretune.base.components.cli import l_cli_main
else:
    l_cli_main = None

# if additional CLI test configurations are needed, append them here and the `cli_test_configs` fixture will generate
# them sessionwise
GEN_CLI_CFGS = [TEST_CONFIGS_CLI_PARITY, TEST_CONFIGS_CLI_UNIT]

def gen_cli_args(run, framework_cli, compose_cfg, config_files, extra_args: List = None):
    cli_main_kwargs = {"run_command": run}
    cli_main_kwargs["args"] = extra_args if extra_args else None
    cli_args, cli_main =  [RUN_FN], core_cli_main  # defaults w/ no framework
    if framework_cli == Framework.lightning:
        cli_main = l_cli_main
        if run:
            cli_args += [run]  # Lightning uses a `jsonargparse` subcommand in sys.argv
    if compose_cfg:
        cli_args.extend(compose_config(config_files))
    else:
        for f in config_files:
            cli_args.extend(["--config", str(f)])
    return cli_main, cli_args, cli_main_kwargs


def gen_experiment_cfg_sets(test_keys: Iterable[Tuple[str, str, str, Optional[str], bool]], sess_paths: Tuple) -> Dict:
    EXPERIMENTS_BASE, BASE_TL_CONFIG, BASE_DEBUG_CONFIG = sess_paths
    exp_cfg_sets = {}
    for exp, model, subexp, plugin_ctx, debug_mode in test_keys:
        base_model_cfg =  EXPERIMENTS_BASE / f"{model}.yaml"
        base_cfg_set = (base_model_cfg,)
        if plugin_ctx:
            if plugin_ctx == Plugin.transformer_lens:
                exp_plugin_cfg = EXPERIMENTS_BASE /  f"{plugin_ctx.value}.yaml"
                base_cfg_set += (BASE_TL_CONFIG, exp_plugin_cfg,)
            else:
                raise ValueError(f"Unknown plugin type: {plugin_ctx}")
        subexp_cfg =  EXPERIMENTS_BASE / model / f"{subexp}.yaml"
        base_cfg_set += (subexp_cfg,)
        if debug_mode:
            base_cfg_set += (BASE_DEBUG_CONFIG,)
        exp_cfg_sets[(exp, model, subexp, plugin_ctx, debug_mode)] = base_cfg_set
    return exp_cfg_sets


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
    EXPERIMENTS_BASE = sess_cfg_base / "experiments" / CLI_EXP_MODEL[0]
    TEST_EXP_MODEL_DIR = EXPERIMENTS_BASE / CLI_EXP_MODEL[1]
    IT_CORE_SHARED = sess_cfg_base / "global" / "core"
    os.environ["IT_CORE_SHARED"] = str(IT_CORE_SHARED)
    IT_LIGHTNING_SHARED = sess_cfg_base / "global" / "lightning"
    os.environ["IT_LIGHTNING_SHARED"] = str(IT_LIGHTNING_SHARED)
    IT_CONFIG_GLOBAL = sess_cfg_base / "global"
    TEST_CLI_CONFIG_FILES = {
        "global_debug": (IT_CONFIG_GLOBAL, "base_debug.yaml", cli_cfgs["global_debug"]),
        "global_tl": (IT_CONFIG_GLOBAL, "base_transformer_lens.yaml", cli_cfgs["global_tl"]),
        "global_core": (IT_CORE_SHARED, "base_core.yaml", cli_cfgs["global_core"]),
        "global_lightning": (IT_LIGHTNING_SHARED, "base_lightning.yaml", cli_cfgs["global_lightning"]),
        "model_tl_cfg": (EXPERIMENTS_BASE, "transformer_lens.yaml", cli_cfgs["model_tl_cfg"]),
        "model_cfgs": (EXPERIMENTS_BASE, "cust.yaml", cli_cfgs["model_cfgs"]),
        "exp_cfgs": cli_cfgs["exp_cfgs"],
    }
    yield TEST_CLI_CONFIG_FILES, TEST_EXP_MODEL_DIR, EXPERIMENTS_BASE
    for env_key in ("IT_CONFIG_BASE", "IT_CORE_SHARED", "IT_LIGHTNING_SHARED", "WANDB_API_KEY", "LLAMA2_AUTH_KEY",
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
    for k, v in test_cli_cfg_files.items():
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
    test_cli_cfg_files, test_exp_model_dir, experiments_base = cli_test_file_env
    test_config_files = write_cli_config_files(test_cli_cfg_files, test_exp_model_dir)
    sess_paths = Path(experiments_base), Path(test_config_files["global_tl"]), Path(test_config_files["global_debug"])
    # we specify a set of CLI test configurations (GEN_CLI_CFGS) to drive our CLI configuration file generation
    test_keys = ((*CLI_EXP_MODEL, tc.alias, tc.cfg.plugin_ctx, tc.cfg.debug_mode) for cfg in GEN_CLI_CFGS for tc in cfg)
    EXPERIMENT_CFG_SETS = gen_experiment_cfg_sets(test_keys=test_keys, sess_paths=sess_paths)
    yield EXPERIMENT_CFG_SETS


@pytest.fixture(scope="function")
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="function")
def make_deterministic(warn_only=True, fill_uninitialized_memory=True):
    # https://pytorch.org/docs/2.2/notes/randomness.html#reproducibility
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
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    if _LIGHTNING_AVAILABLE:
        from lightning.fabric.utilities import rank_zero_only
    else:
        from interpretune.utils.logging import rank_zero_only  # type: ignore[no-redef]

    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


@pytest.fixture(scope="function", autouse=True)
def restore_env_variables():
    """Ensures that environment variables set during the test do not leak out."""
    okay_session_scope_keys = {"IT_CONFIG_BASE", "IT_CORE_SHARED", "IT_LIGHTNING_SHARED", "WANDB_API_KEY",
                               "LLAMA2_AUTH_KEY", "IDE_PROJECT_ROOTS"}
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
    elif os.getenv("IT_RUN_SLOW_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(slow=True)`
            if marker.name == "skipif" and marker.kwargs.get("slow")
        ]
