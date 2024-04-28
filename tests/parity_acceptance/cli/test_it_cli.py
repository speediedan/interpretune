import pytest
from unittest import mock

from tests.utils.runif import RunIf
from tests.utils.warns import unexpected_warns
from tests.conftest import make_deterministic  # noqa: F401
from tests.configuration import pytest_param_factory
from tests.parity_acceptance.cli.conftest import gen_cli_args
from tests.parity_acceptance.cli.cfg_aliases import (TEST_CONFIGS_CLI, CLI_EXPECTED_WARNS, EXPECTED_RESULTS_CLI,
                                                     CLI_EXP_MODEL)


@pytest.mark.usefixtures("make_deterministic")
@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("test_alias, cli_cfg", pytest_param_factory(TEST_CONFIGS_CLI, unpack=False))
def test_cli_configs(recwarn, clean_cli_env, cli_test_configs, test_alias, cli_cfg):
    expected_warnings = CLI_EXPECTED_WARNS[(cli_cfg.framework_cli, cli_cfg.plugin_ctx)]
    cfg_files = cli_test_configs[(*CLI_EXP_MODEL, test_alias, cli_cfg.plugin_ctx, cli_cfg.debug_mode)]
    cli_main, cli_args, main_kwargs = gen_cli_args(cli_cfg.run, cli_cfg.framework_cli, cli_cfg.compose_cfg, cfg_files)
    with mock.patch('sys.argv', cli_args):
        cli = cli_main(**main_kwargs)
    if not cli_cfg.run:
        assert hasattr(cli.module, EXPECTED_RESULTS_CLI[test_alias]['hasattr'])
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
