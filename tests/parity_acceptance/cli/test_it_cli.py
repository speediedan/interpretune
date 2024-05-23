import pytest
from unittest import mock
import subprocess
import os
import sys

from tests.utils.runif import RunIf
from tests.utils.warns import unexpected_warns
from tests.conftest import make_deterministic  # noqa: F401
from tests.configuration import pytest_param_factory
from tests.conftest import gen_cli_args
from tests.parity_acceptance.cli.cfg_aliases import (TEST_CONFIGS_CLI_PARITY, CLI_EXPECTED_WARNS, EXPECTED_RESULTS_CLI,
                                                     CLI_EXP_MODEL, IT_HOME)


@pytest.mark.usefixtures("make_deterministic")
@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("test_alias, cli_cfg", pytest_param_factory(TEST_CONFIGS_CLI_PARITY, unpack=False))
def test_cli_configs(recwarn, clean_cli_env, cli_test_configs, test_alias, cli_cfg):
    expected_warnings = CLI_EXPECTED_WARNS[(cli_cfg.cli_adapter, *cli_cfg.adapter_ctx)]
    cfg_files = cli_test_configs[(*CLI_EXP_MODEL, test_alias, cli_cfg.adapter_ctx, cli_cfg.debug_mode)]
    cli_main, cli_args, main_kwargs = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files)
    if cli_cfg.use_harness:
        command = [sys.executable] + [os.path.join(IT_HOME, cli_args.pop(0))] + cli_args
        cp = subprocess.run(command)
        assert cp.returncode == 0
    else:
        with mock.patch('sys.argv', cli_args):
            cli = cli_main(**main_kwargs)
    if not cli_cfg.run:
        assert hasattr(cli.module, EXPECTED_RESULTS_CLI[test_alias]['hasattr'])
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
