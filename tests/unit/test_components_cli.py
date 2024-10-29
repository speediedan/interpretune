import pytest
from unittest.mock import patch
import os
import shutil
from copy import deepcopy

from jsonargparse import Namespace, ArgumentError

from interpretune.base.components.cli import LightningCLIAdapter, bootstrap_cli
from tests.runif import RunIf
from tests.warns import unexpected_warns, CLI_EXPECTED_WARNS
from tests.conftest import make_deterministic  # noqa: F401
from tests.base_defaults import pytest_param_factory
from tests.parity_acceptance.cfg_aliases import RUN_FN, CLI_EXP
from tests.unit.cfg_aliases import TEST_CONFIGS_CLI_UNIT, EXPECTED_RESULTS_CLI_UNIT
from tests.parity_acceptance.test_it_cli import gen_cli_args


def collect_base_config():
    base_test = TEST_CONFIGS_CLI_UNIT[0]
    cli_cfg = base_test.cfg
    test_alias = base_test.alias
    return deepcopy(cli_cfg), test_alias

@pytest.mark.usefixtures("make_deterministic")
@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("test_alias, cli_cfg", pytest_param_factory(TEST_CONFIGS_CLI_UNIT, unpack=False))
def test_cli_unit_configs(recwarn, clean_cli_env, cli_test_configs, test_alias, cli_cfg):
    expected_warnings = CLI_EXPECTED_WARNS[(cli_cfg.cli_adapter, *cli_cfg.adapter_ctx)]
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    cli_main, cli_args, main_kwargs = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files,
                                                   cli_cfg.bootstrap_args, cli_cfg.extra_args)
    should_raise = (cli_cfg.extra_args)
    if should_raise:
        with pytest.raises(SystemExit):
            _ = cli_main(**main_kwargs)
    elif cli_cfg.env_seed:
        with patch.dict(os.environ, {"IT_GLOBAL_SEED": str(cli_cfg.env_seed)}), patch('sys.argv', cli_args):
            _ = cli_main(**main_kwargs)
            seed_result = os.environ.get("IT_GLOBAL_SEED")
    else:
        with patch('sys.argv', cli_args):
            _ = cli_main(**main_kwargs)
            seed_result = os.environ.get("IT_GLOBAL_SEED")
    if not should_raise:
        seed_test = EXPECTED_RESULTS_CLI_UNIT[test_alias].get('seed_test', None)
        if seed_test is not None:
            assert seed_test(seed_result)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

def test_enumerate_config(clean_cli_env, cli_test_configs):
    cli_cfg, test_alias = collect_base_config()
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    cli_main, cli_args, main_kwargs = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files,
                                                   cli_cfg.bootstrap_args, cli_cfg.extra_args)
    shared_dir_err_path = (cfg_files[1].parent / "defaults" / cfg_files[1].parts[-1]).with_suffix('.err')
    shutil.copy(cfg_files[1], shared_dir_err_path)
    cfg_files = cfg_files + (shared_dir_err_path,)
    with pytest.raises(ValueError, match="Non-YAML files found in directory"), patch('sys.argv', cli_args):
        _ = cli_main(**main_kwargs)
    os.remove(shared_dir_err_path)

def test_compose_config_absolute_exception(clean_cli_env, cli_test_configs):
    cli_cfg, test_alias = collect_base_config()
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    fnf_path = (cfg_files[1].parent / "defaults" / cfg_files[1].parts[-1]).with_suffix('.err')
    cfg_files = cfg_files + (fnf_path,)
    with pytest.raises(FileNotFoundError, match="Could not find configuration file path"):
        _ = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files, cli_cfg.bootstrap_args,
                         cli_cfg.extra_args)

@pytest.mark.parametrize("glob_search", [True, False], ids=["glob_search", "no_glob_search"])
@pytest.mark.parametrize("fnf_error", [True, False], ids=["fnf_error", "no_fnf_error"])
def test_compose_config_relative(clean_cli_env, cli_test_configs, fnf_error, glob_search):
    cli_cfg, test_alias = collect_base_config()
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    with patch('interpretune.base.components.cli.IT_CONFIG_BASE', os.environ.get('IT_CONFIG_BASE')):
        if fnf_error:
            if not glob_search:
                fnf_path = (cfg_files[1].parent / "defaults" / cfg_files[1].parts[-1]).with_suffix('.err')
                cfg_files = cfg_files + (fnf_path,)
            else:
                file_name_only_path = cfg_files[1].parts[-1].replace('.yaml', '.err')
                cfg_files = cfg_files[:-1] + (file_name_only_path,)
            with pytest.raises(FileNotFoundError, match="Could not find configuration file path"):
                _ = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files,
                                 cli_cfg.bootstrap_args, cli_cfg.extra_args)
        else:
            if not glob_search:
                explicit_relative_path = f"{cfg_files[1].parts[-2]}/{cfg_files[1].parts[-1]}"
                cfg_files = cfg_files[:-1] + (explicit_relative_path,)
                cli_main, *_ = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg,
                                                               cfg_files, cli_cfg.extra_args)
            elif glob_search:  # we always warn with glob_search
                file_name_only_path = cfg_files[1].parts[-1]
                cfg_files = cfg_files[:-1] + (file_name_only_path,)
                with pytest.warns(UserWarning, match="Glob search within"):
                    cli_main, *_ = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files,
                                               cli_cfg.extra_args)
            assert cli_main

@RunIf(lightning=True)
def test_lightning_adapter_attr_missing(clean_cli_env):
    lightning_cli_adapter = LightningCLIAdapter()
    mock_config =  Namespace({'it_session': Namespace({'datamodule': 'mock_dm'})})
    assert lightning_cli_adapter._it_session_cfg(mock_config, 'it_session.datamodule') == 'mock_dm'
    assert lightning_cli_adapter._it_session_cfg(mock_config, 'it_session.missing') is None

@pytest.mark.parametrize("run", [True, False], ids=["run", "norun"])
@pytest.mark.parametrize("l_cli", [True, False],ids=["l_cli", "core_cli"])
def test_bootstrap_cli(clean_cli_env, l_cli, run):
    # note this test targets the CLI bootstrap parsing logic versus full CLI execution
    # the CLI bootstrap options are also exercised in `test_cli_configs` but tested here as well in part to ensure the
    # lines covered by `test_cli_configs` via subprocesses are properly captured by coverage without the burden
    # of extra coverage-specific subprocess configuration
    from interpretune.base.components.cli import l_cli_main, core_cli_main
    cli_args = [RUN_FN]
    target_cli = l_cli_main if l_cli else core_cli_main
    if l_cli:
        cli_args.extend(["--lightning_cli", "--no_run" if not run else "test"])
    else:
        cli_args.extend(["--run_command", "test"] if run else [])
    cli_args.extend(["--config", "config.yaml"])
    # we expect to error out with an ArgumentError if this test is run independently since session-level fixture
    # `cli_test_configs` files won't be present. If run with that fixture's files available, we error with a SystemExit
    with pytest.raises((SystemExit, ArgumentError)) as err, patch('sys.argv', cli_args):
        _ = bootstrap_cli()
    assert err.traceback[1].locals['cli_main'] is target_cli
