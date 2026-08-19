from unittest.mock import patch
import os
import shutil
from copy import deepcopy
from functools import partial

import pytest
import yaml
from jsonargparse import ArgumentError, ArgumentParser

from interpretune.base import LightningCLIAdapter, ITSessionMixin, bootstrap_cli
from interpretune.base.components.cli import ONE_DOOR_BODY_KEYS
from tests.core.loader_equivalence import cli_experiment_configs
from tests.runif import RunIf
from tests.warns import unexpected_warns, CLI_EXPECTED_WARNS
from tests.base_defaults import pytest_factory
from tests.parity_acceptance.cfg_aliases import RUN_FN, CLI_EXP
from tests.core.cfg_aliases import TEST_CONFIGS_CLI_UNIT, EXPECTED_RESULTS_CLI_UNIT
from tests.parity_acceptance.test_it_cli import gen_cli_args


def collect_base_config():
    base_test = TEST_CONFIGS_CLI_UNIT[0]
    cli_cfg = base_test.cfg
    test_alias = base_test.alias
    return deepcopy(cli_cfg), test_alias


@pytest.mark.usefixtures("make_deterministic")
@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("test_alias, cli_cfg", pytest_factory(TEST_CONFIGS_CLI_UNIT, unpack=False))
def test_cli_unit_configs(recwarn, clean_cli_env, cli_test_configs, test_alias, cli_cfg):
    expected_warnings = CLI_EXPECTED_WARNS[(cli_cfg.cli_adapter, *cli_cfg.adapter_ctx)]
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    cli_main, cli_args, main_kwargs = gen_cli_args(
        cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files, cli_cfg.bootstrap_args, cli_cfg.extra_args
    )
    should_raise = cli_cfg.extra_args
    if should_raise:
        with pytest.raises(SystemExit):
            _ = cli_main(**main_kwargs)
    elif cli_cfg.env_seed:
        with patch.dict(os.environ, {"IT_GLOBAL_SEED": str(cli_cfg.env_seed)}), patch("sys.argv", cli_args):
            _ = cli_main(**main_kwargs)
            seed_result = os.environ.get("IT_GLOBAL_SEED")
    else:
        with patch("sys.argv", cli_args):
            _ = cli_main(**main_kwargs)
            seed_result = os.environ.get("IT_GLOBAL_SEED")
    if not should_raise:
        seed_test = EXPECTED_RESULTS_CLI_UNIT[test_alias].get("seed_test", None)
        if seed_test is not None:
            assert seed_test(seed_result)
        unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
        assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


def test_enumerate_config(clean_cli_env, cli_test_configs):
    cli_cfg, test_alias = collect_base_config()
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    cli_main, cli_args, main_kwargs = gen_cli_args(
        cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files, cli_cfg.bootstrap_args, cli_cfg.extra_args
    )
    shared_dir_err_path = (cfg_files[1].parent / "defaults" / cfg_files[1].parts[-1]).with_suffix(".err")
    shutil.copy(cfg_files[1], shared_dir_err_path)
    cfg_files = cfg_files + (shared_dir_err_path,)
    with pytest.raises(ValueError, match="Non-YAML files found in directory"), patch("sys.argv", cli_args):
        _ = cli_main(**main_kwargs)
    os.remove(shared_dir_err_path)


def test_compose_config_absolute_exception(clean_cli_env, cli_test_configs):
    cli_cfg, test_alias = collect_base_config()
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    fnf_path = (cfg_files[1].parent / "defaults" / cfg_files[1].parts[-1]).with_suffix(".err")
    cfg_files = cfg_files + (fnf_path,)
    with pytest.raises(FileNotFoundError, match="Could not find configuration file path"):
        _ = gen_cli_args(
            cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files, cli_cfg.bootstrap_args, cli_cfg.extra_args
        )


@pytest.mark.parametrize("glob_search", [True, False], ids=["glob_search", "no_glob_search"])
@pytest.mark.parametrize("fnf_error", [True, False], ids=["fnf_error", "no_fnf_error"])
def test_compose_config_relative(clean_cli_env, cli_test_configs, fnf_error, glob_search):
    cli_cfg, test_alias = collect_base_config()
    cfg_files = cli_test_configs[(CLI_EXP, test_alias, cli_cfg.debug_mode)]
    with patch("interpretune.base.components.cli.IT_CONFIG_BASE", os.environ.get("IT_CONFIG_BASE")):
        if fnf_error:
            if not glob_search:
                fnf_path = (cfg_files[1].parent / "defaults" / cfg_files[1].parts[-1]).with_suffix(".err")
                cfg_files = cfg_files + (fnf_path,)
            else:
                file_name_only_path = cfg_files[1].parts[-1].replace(".yaml", ".err")
                cfg_files = cfg_files[:-1] + (file_name_only_path,)
            with pytest.raises(FileNotFoundError, match="Could not find configuration file path"):
                _ = gen_cli_args(
                    cli_cfg.run,
                    cli_cfg.cli_adapter,
                    cli_cfg.compose_cfg,
                    cfg_files,
                    cli_cfg.bootstrap_args,
                    cli_cfg.extra_args,
                )
        else:
            if not glob_search:
                explicit_relative_path = f"{cfg_files[1].parts[-2]}/{cfg_files[1].parts[-1]}"
                cfg_files = cfg_files[:-1] + (explicit_relative_path,)
                cli_main, *_ = gen_cli_args(
                    cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files, cli_cfg.extra_args
                )
            elif glob_search:  # we always warn with glob_search
                file_name_only_path = cfg_files[1].parts[-1]
                cfg_files = cfg_files[:-1] + (file_name_only_path,)
                with pytest.warns(UserWarning, match="Glob search within"):
                    cli_main, *_ = gen_cli_args(
                        cli_cfg.run, cli_cfg.cli_adapter, cli_cfg.compose_cfg, cfg_files, cli_cfg.extra_args
                    )
            assert cli_main


@RunIf(lightning=True)
def test_lightning_adapter_attr_missing(clean_cli_env):
    class _MockSession:
        datamodule = "mock_dm"

    lightning_cli_adapter = LightningCLIAdapter()
    lightning_cli_adapter.it_session = _MockSession()  # the session is loader-built and object-resolved now
    assert lightning_cli_adapter._it_session_object_attr("it_session.datamodule") == "mock_dm"
    assert lightning_cli_adapter._it_session_object_attr("it_session.missing") is None


@pytest.mark.parametrize("run", [True, False], ids=["run", "norun"])
@pytest.mark.parametrize("l_cli", [True, False], ids=["l_cli", "core_cli"])
def test_bootstrap_cli(clean_cli_env, l_cli, run):
    # note this test targets the CLI bootstrap parsing logic versus full CLI execution
    # the CLI bootstrap options are also exercised in `test_cli_configs` but tested here as well in part to ensure the
    # lines covered by `test_cli_configs` via subprocesses are properly captured by coverage without the burden
    # of extra coverage-specific subprocess configuration
    from interpretune.base import l_cli_main, core_cli_main

    cli_args = [RUN_FN]
    target_cli = l_cli_main if l_cli else core_cli_main
    if l_cli:
        cli_args.extend(["--lightning_cli", "--no_run" if not run else "test"])
    else:
        cli_args.extend(["--run_command", "test"] if run else [])
    cli_args.extend(["--config", "config.yaml"])
    # we expect to error out with an ArgumentError if this test is run independently since session-level fixture
    # `cli_test_configs` files won't be present. If run with that fixture's files available, we error with a SystemExit
    with pytest.raises((SystemExit, ArgumentError)) as err, patch("sys.argv", cli_args):
        _ = bootstrap_cli()
    assert err.traceback[1].locals["cli_main"] is target_cli


################################################################################
# Parse-surface pins for the shipped one-door configurations
################################################################################
# NOTE [Parse Surface vs Loader]: `tests/core/test_loader_equivalence.py` pins what the LOADER makes of
# these bodies; the tests below pin that the argv/config-file shim ACCEPTS them in the first place.
# Nothing pinned the second half before, and the gap shipped: the 4b migration (25ee43c) flattened every
# config in `experiments/cli/` to the one-door schema without adding those top-level keys to the parse
# surface, so all 15 failed `interpretune --config <cfg>` with rc=2. The loader harness stayed green
# (it bypasses the parser) and every CLI test config stayed in the legacy `session_cfg` dialect (see
# `tests/parity_acceptance/cfg_aliases.py`), so nothing failed, while the entire registered-benchmark
# lane, which drives the real CLI, was dark.

SHIPPED_CLI_CONFIGS = cli_experiment_configs()


def _is_lightning_config(config_path) -> bool:
    return "lightning" in (yaml.safe_load(config_path.read_text(encoding="utf-8")).get("adapter_ctx") or [])


CORE_CLI_CONFIGS = [p for p in SHIPPED_CLI_CONFIGS if not _is_lightning_config(p)]
LIGHTNING_CLI_CONFIGS = [p for p in SHIPPED_CLI_CONFIGS if _is_lightning_config(p)]


def _assert_parses(cli_main, args, config_path, surface):
    """`--print_config` dumps and exits DURING parsing, so the parse surface is pinned on CPU without
    loading a model: a rejected key exits 2, an accepted one exits 0."""
    with pytest.raises(SystemExit) as exc, patch("sys.argv", [RUN_FN]):
        cli_main(args=args)
    assert exc.value.code == 0, f"the {surface} parse surface rejected the shipped config {config_path.name}"


def test_shipped_cli_configs_discovered():
    """Both halves must stay populated; an empty half would make the pins below vacuously pass."""
    assert len(SHIPPED_CLI_CONFIGS) == 15, sorted(str(c) for c in SHIPPED_CLI_CONFIGS)
    assert CORE_CLI_CONFIGS and LIGHTNING_CLI_CONFIGS


@pytest.mark.parametrize("config_path", CORE_CLI_CONFIGS, ids=lambda p: p.stem)
def test_shipped_core_cli_config_parses(clean_cli_env, config_path):
    from interpretune.base import core_cli_main

    _assert_parses(
        partial(core_cli_main, run_mode=False), ["--config", str(config_path), "--print_config"], config_path, "core"
    )


@RunIf(lightning=True)
@pytest.mark.parametrize("config_path", LIGHTNING_CLI_CONFIGS, ids=lambda p: p.stem)
def test_shipped_lightning_cli_config_parses(clean_cli_env, config_path):
    from interpretune.base import l_cli_main

    _assert_parses(
        partial(l_cli_main, run_mode=True),
        ["test", "--config", str(config_path), "--print_config"],
        config_path,
        "lightning",
    )


def test_one_door_body_keys_are_on_the_parse_surface():
    """The shim's read set and its parse surface must not drift apart.

    `_session_mapping_from_sources` reads `ONE_DOOR_BODY_KEYS` out of a flattened body, but jsonargparse rejects the
    file before the shim ever sees it unless the parser also declares them. Pinning the two together means a future
    dialect key fails here, naming its cause, instead of failing opaquely in every shipped configuration at once.
    """
    parser = ArgumentParser()
    ITSessionMixin().add_arguments_to_parser(parser)
    declared = {action.dest for action in parser._actions}
    assert set(ONE_DOOR_BODY_KEYS) <= declared, f"read but not declared: {set(ONE_DOOR_BODY_KEYS) - declared}"


def test_shipped_configs_use_only_known_top_level_keys():
    """The other direction: a config may not introduce a top-level key the shim has no action for.

    `trainer` is Lightning's own argument group, so it is allowed only where Lightning is in `adapter_ctx`. A core
    configuration carrying one would parse against a parser that has no such group.
    """
    shim_keys = {"seed_everything", "session_cfg", *ONE_DOOR_BODY_KEYS}
    for config_path in SHIPPED_CLI_CONFIGS:
        top_level = set(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        allowed = shim_keys | {"trainer"} if _is_lightning_config(config_path) else shim_keys
        assert top_level <= allowed, f"{config_path.name} has unknown top-level keys: {sorted(top_level - allowed)}"
