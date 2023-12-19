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
import os.path
from unittest import mock
from typing import Iterable, Dict, Tuple
import pytest

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.cli.core_cli import compose_config, IT_CONFIG_BASE, IT_CONFIG_GLOBAL
from it_examples.experiments.rte_boolq.core import GPT2RTEBoolqITHookedModule
from tests.helpers.core.boring_model import (unexpected_warns, pytest_param_factory, TestCfg, RUN_FN, dummy_step,
                                             EXPECTED_WARNS)
from tests.helpers.runif import RunIf

if _LIGHTNING_AVAILABLE:
    from interpretune.base.it_lightning_modules import ITLightningModule
    from it_examples.experiments.rte_boolq.lightning import GPT2ITHookedLightningModule, Llama2ITLightningModule
else:
    ITLightningModule = None  # type: ignore[misc,assignment]
    GPT2ITHookedLightningModule = None  # type: ignore[misc,assignment]
    Llama2ITLightningModule = None  # type: ignore[misc,assignment]


EXPERIMENTS_BASE = IT_CONFIG_BASE / "experiments"
BASE_DEBUG_CONFIG = IT_CONFIG_GLOBAL / "base_debug.yaml"


def gen_experiment_cfg_sets(test_keys: Iterable[Tuple[str, str, str, str]]) -> Dict:
    exp_cfg_sets = {}
    for exp, model, subexp, debug_mode in test_keys:
        base_model_cfg =  EXPERIMENTS_BASE / exp / f"{model}.yaml"
        subexp_cfg =  EXPERIMENTS_BASE / exp / model / f"{subexp}.yaml"
        base_cfg_set = (base_model_cfg, subexp_cfg)
        if debug_mode == "debug":
            exp_cfg_sets[(exp, model, subexp, debug_mode)] = (*base_cfg_set, BASE_DEBUG_CONFIG)
        else:
            exp_cfg_sets[(exp, model, subexp, debug_mode)] = base_cfg_set
    return exp_cfg_sets

EXPERIMENT_CONFIG_SETS = gen_experiment_cfg_sets(
    (("rte_boolq", "gpt2", "rte_small_optimizer_scheduler_init", "debug"),
     ("rte_boolq", "gpt2", "hooked_rte_small_it_cli_test", "debug"),
     ("rte_boolq", "gpt2", "lightning_hooked_rte_small_noquant_test", "nodebug"),
     ("rte_boolq", "llama2", "lightning_rte_7b_qlora_zero_shot_test_only", "debug"))
)

# experiment config set aliases
gpt2_core = EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2", "rte_small_optimizer_scheduler_init", "debug")]
gpt2_core_hooked = EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2", "hooked_rte_small_it_cli_test", "debug")]
gpt2_l_hooked = EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2", "lightning_hooked_rte_small_noquant_test", "nodebug")]
llama2_l = EXPERIMENT_CONFIG_SETS[("rte_boolq", "llama2", "lightning_rte_7b_qlora_zero_shot_test_only", "debug")]

TEST_CONFIGS_EXAMPLES = (
    TestCfg("core_gpt2_hooked_compose_config", test_cfg=(False, None, False, gpt2_core_hooked, True)),
    TestCfg("core_gpt2_hooked", test_cfg=(False, None, False, gpt2_core_hooked, False)),
    TestCfg("core_gpt2_hooked_instantiate_only",test_cfg=(False, None, True, gpt2_core_hooked, False),
               expected_results={'class_type': GPT2RTEBoolqITHookedModule}),
    TestCfg("core_gpt2_optim_init", test_cfg=(False, None, False, gpt2_core, True)),
    TestCfg("lightning_gpt2_hooked", test_cfg=(True, "test", False, gpt2_l_hooked, False), marks="lightning"),
    TestCfg("lightning_gpt2_hooked_instantiate_only", test_cfg=(True, None, True, gpt2_l_hooked, False),
               marks="lightning", expected_results={'class_type': GPT2ITHookedLightningModule}),
    TestCfg("lightning_llama2", test_cfg=(True, "test", False, llama2_l, False), marks="lightning"),
)

EXPECTED_RESULTS_EXAMPLES = {cfg.test_alias: cfg.expected_results for cfg in TEST_CONFIGS_EXAMPLES}


def gen_cli_args(l_cli, subcommand, use_compose_config, config_files):
    if not l_cli:
        from interpretune.cli.core_cli import cli_main
        cli_args = [RUN_FN]
    else:
        from interpretune.cli.lightning_cli import cli_main
        cli_args = [RUN_FN, subcommand] if subcommand else [RUN_FN]
    if use_compose_config:
        cli_args.extend(compose_config(config_files))
    else:
        for f in config_files:
            cli_args.extend(["--config", os.path.join(IT_CONFIG_BASE, f)])
    return cli_main, cli_args

def invoke_cli(cli_main, cli_args, l_cli, instantiate_only):
    with mock.patch('sys.argv', cli_args):
        if l_cli and not instantiate_only:
            # TODO: once examples are real, remove this `dummy_step` patch and just run a small number of real steps
            with mock.patch.object(ITLightningModule, "test_step", dummy_step):
                cli = cli_main(instantiate_only=instantiate_only)
        else:
            cli = cli_main(instantiate_only=instantiate_only)
    return cli

@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("test_alias, l_cli, subcommand, instantiate_only, config_files, use_compose_config",
                         pytest_param_factory(TEST_CONFIGS_EXAMPLES))
def test_basic_examples(recwarn, test_alias, l_cli, subcommand, instantiate_only, config_files, use_compose_config):
    # TODO: refactor these conditional evaluations into functions resuable in other tests
    cli_main, cli_args = gen_cli_args(l_cli, subcommand, use_compose_config, config_files)
    cli = invoke_cli(cli_main, cli_args, l_cli, instantiate_only)
    if instantiate_only:
        assert isinstance(cli.model, EXPECTED_RESULTS_EXAMPLES[test_alias]['class_type'])
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=EXPECTED_WARNS)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
