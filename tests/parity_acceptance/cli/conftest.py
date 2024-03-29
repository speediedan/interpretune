import os
from typing import Iterable,Dict, Tuple, Optional
from pathlib import Path

import yaml
import pytest

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.base.cli.core_cli import core_cli_main, compose_config
from interpretune.base.contract.session import Framework, Plugin
from tests.parity_acceptance.cli.cfg_aliases import cli_cfgs, CLI_EXP_MODEL, RUN_FN, TEST_CONFIGS_CLI

if _LIGHTNING_AVAILABLE:
    from interpretune.base.cli.lightning_cli import l_cli_main
else:
    l_cli_main = None


def gen_cli_args(run, framework_cli, compose_cfg, config_files):
    cli_main_kwargs, cli_args, cli_main = {"run_command": run}, [RUN_FN], core_cli_main  # defaults w/ no framework
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

def write_cli_config_file(path, config):
    with open(path, "w") as f:
        f.write(yaml.dump(config))

def write_cli_config_files(test_cli_cfg_files, test_exp_model_dir):
    test_config_files = {}
    for k, v in test_cli_cfg_files.items():
        if k == "exp_cfgs":
            for k, v in v.items():
                cfg_dir = ensure_path(test_exp_model_dir)
                cfg_file_path = cfg_dir / f"{k.value}.yaml"
                write_cli_config_file(cfg_file_path, v)
                test_config_files[k] = cfg_file_path
        else:
            cfg_dir = ensure_path(Path(v[0]))
            cfg_file_path = cfg_dir / v[1]
            write_cli_config_file(cfg_file_path, v[2])
            test_config_files[k] = cfg_file_path
    return test_config_files


@pytest.fixture(scope="session")
def cli_test_configs(cli_test_file_env):
    # this fixture will collect all required cli test configuration files and dynamically generate them sessionwise
    # if additional CLI test configurations are needed, append them here and the fixture will generate them sessionwise
    GEN_CLI_CFGS = [TEST_CONFIGS_CLI]
    test_cli_cfg_files, test_exp_model_dir, experiments_base = cli_test_file_env
    test_config_files = write_cli_config_files(test_cli_cfg_files, test_exp_model_dir)
    sess_paths = Path(experiments_base), Path(test_config_files["global_tl"]), Path(test_config_files["global_debug"])
    # we specify a set of CLI test configurations (GEN_CLI_CFGS) to drive our CLI configuration file generation
    test_keys = ((*CLI_EXP_MODEL, tc.alias, tc.cfg.plugin_ctx, tc.cfg.debug_mode) for cfg in GEN_CLI_CFGS for tc in cfg)
    EXPERIMENT_CFG_SETS = gen_experiment_cfg_sets(test_keys=test_keys, sess_paths=sess_paths)
    yield EXPERIMENT_CFG_SETS
