import pytest
import subprocess
import os
import sys
from unittest import mock
from typing import Optional, List, Sequence
from dataclasses import dataclass

from interpretune.adapters import ADAPTER_REGISTRY
from interpretune.base import core_cli_main, compose_config, l_cli_main
from interpretune.utils import ArgsType, Adapter
from tests.base_defaults import pytest_factory, BaseAugTest
from tests.parity_acceptance.cfg_aliases import CLI_TESTS, IT_HOME, RUN_FN, CLI_EXP
from tests.runif import RunIf
from tests.warns import unexpected_warns, CLI_EXPECTED_WARNS


@dataclass(kw_only=True)
class CLICfg:
    cli_adapter: Adapter = Adapter.core
    run: Optional[str] = None
    env_seed: Optional[str] = None
    compose_cfg: bool = False
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core,)
    debug_mode: bool = False
    use_harness: bool = False
    bootstrap_args: Optional[List] = None
    extra_args: Optional[ArgsType] = None

    def __post_init__(self):
        self.adapter_ctx = ADAPTER_REGISTRY.canonicalize_composition(self.adapter_ctx)

TEST_CONFIGS_CLI_PARITY = (
    BaseAugTest(alias=CLI_TESTS.core_tl_test.value, cfg=CLICfg(compose_cfg=True, run="test", debug_mode=True,
                                                           adapter_ctx=(Adapter.core, Adapter.transformer_lens),
                                                           bootstrap_args=["--run_command=test"], use_harness=True)),
    BaseAugTest(alias=CLI_TESTS.core_tl_test_noharness.value, cfg=CLICfg(compose_cfg=True, run="test", debug_mode=True,
                                                           adapter_ctx=(Adapter.core, Adapter.transformer_lens))),
    BaseAugTest(alias=CLI_TESTS.core_tl_norun.value, cfg=CLICfg(adapter_ctx=(Adapter.core, Adapter.transformer_lens)),
            expected={'hasattr': 'tl_config_model_init'}),
    BaseAugTest(alias=CLI_TESTS.core_optim_train.value, cfg=CLICfg(compose_cfg=True, run="train", debug_mode=True)),
    BaseAugTest(alias=CLI_TESTS.l_tl_test.value, cfg=CLICfg(cli_adapter=Adapter.lightning, run="test",
                                                        adapter_ctx=(Adapter.lightning, Adapter.transformer_lens)),
                                                        marks="lightning"),
    BaseAugTest(alias=CLI_TESTS.l_tl_norun.value, cfg=CLICfg(cli_adapter=Adapter.lightning,
                                                         adapter_ctx=(Adapter.lightning, Adapter.transformer_lens),
                                                         bootstrap_args=["--lightning_cli", "--no_run"],
                                                         use_harness=True), marks="lightning",
                                                         expected={'hasattr': 'tl_config_model_init'}),
    BaseAugTest(alias=CLI_TESTS.l_tl_norun_noharness.value, cfg=CLICfg(cli_adapter=Adapter.lightning,
                                                        adapter_ctx=(Adapter.lightning, Adapter.transformer_lens)),
                                                        marks="lightning",
                                                        expected={'hasattr': 'tl_config_model_init'}),
    BaseAugTest(alias=CLI_TESTS.l_optim_fit.value, cfg=CLICfg(compose_cfg=True, run="fit", debug_mode=True,
                                                          cli_adapter=Adapter.lightning,
                                                          adapter_ctx=(Adapter.lightning,))),
)

EXPECTED_RESULTS_CLI = {cfg.alias: cfg.expected for cfg in TEST_CONFIGS_CLI_PARITY}


def gen_cli_args(run, cli_adapter, compose_cfg, config_files, bootstrap_args: Optional[ArgsType] = None,
                 extra_args: Optional[ArgsType] = None):
    cli_main_kwargs = {"run_mode": run} if run else {"run_mode": False}
    cli_main_kwargs["args"] = extra_args if extra_args else None
    cli_args, cli_main =  [RUN_FN], core_cli_main  # defaults w/ no adapter
    if bootstrap_args:
        cli_args.extend(bootstrap_args)
    if cli_adapter == Adapter.lightning:
        cli_main = l_cli_main
        if run:
            cli_args += [run]  # Lightning uses a `jsonargparse` subcommand in sys.argv
    if compose_cfg:  # TODO: consider deprecating compose_config for simplicity and subsequently removing this path
        cli_args.extend(compose_config(config_files))
    else:
        for f in config_files:
            cli_args.extend(["--config", str(f)])
    return cli_main, cli_args, cli_main_kwargs

@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("test_alias, cli_cfg", pytest_factory(TEST_CONFIGS_CLI_PARITY, unpack=False))
def test_cli_configs(recwarn, make_deterministic, clean_cli_env, cli_test_configs, test_alias, cli_cfg):
    expected_warnings = CLI_EXPECTED_WARNS[(cli_cfg.cli_adapter, *cli_cfg.adapter_ctx)]
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    cli_main, cli_args, main_kwargs = gen_cli_args(cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files,
                                                   bootstrap_args=cli_cfg.bootstrap_args, extra_args=cli_cfg.extra_args)
    if cli_cfg.use_harness:
        command = [sys.executable] + [os.path.join(IT_HOME, cli_args.pop(0))] + cli_args
        cp = subprocess.run(command)
        assert cp.returncode == 0
    else:
        with mock.patch('sys.argv', cli_args):
            cli = cli_main(**main_kwargs)
    if not cli_cfg.run and not cli_cfg.use_harness:
        assert hasattr(cli.module, EXPECTED_RESULTS_CLI[test_alias]['hasattr'])
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
