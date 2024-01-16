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
from it_examples.experiments.rte_boolq.core import GPT2RTEBoolqITLensModule
from tests.configuration import pytest_param_factory, TestCfg
from tests.orchestration import dummy_step
from tests.utils.runif import RunIf
from tests.utils.warns import EXPECTED_WARNS, HF_EXPECTED_WARNS, unexpected_warns

if _LIGHTNING_AVAILABLE:
    from interpretune.base.modules import ITLightningModule
    from it_examples.experiments.rte_boolq.lightning import GPT2ITLensLightningModule
else:
    ITLightningModule = None  # type: ignore[misc,assignment]
    GPT2ITLensLightningModule = None  # type: ignore[misc,assignment]

EXAMPLE_WARNS = EXPECTED_WARNS + HF_EXPECTED_WARNS

RUN_FN = "run_experiment.py"
EXPERIMENTS_BASE = IT_CONFIG_BASE / "experiments"
BASE_DEBUG_CONFIG = IT_CONFIG_GLOBAL / "base_debug.yaml"
BASE_TL_CONFIG = IT_CONFIG_GLOBAL / "base_transformerlens.yaml"

def gen_experiment_cfg_sets(test_keys: Iterable[Tuple[str, str, str, bool, bool]]) -> Dict:
    exp_cfg_sets = {}
    for exp, model, subexp, use_tl, debug_mode in test_keys:
        base_model_cfg =  EXPERIMENTS_BASE / exp / f"{model}.yaml"
        subexp_cfg =  EXPERIMENTS_BASE / exp / model / f"{subexp}.yaml"
        base_cfg_set = (base_model_cfg, subexp_cfg)
        for cfg, enabled in zip((BASE_TL_CONFIG, BASE_DEBUG_CONFIG), (use_tl, debug_mode)):
            if enabled:
                base_cfg_set += (cfg,)
        exp_cfg_sets[(exp, model, subexp, use_tl, debug_mode)] = base_cfg_set
    return exp_cfg_sets

EXPERIMENT_CONFIG_SETS = gen_experiment_cfg_sets(
    (("rte_boolq", "gpt2", "rte_small_optimizer_scheduler_init", False, True),
     ("rte_boolq", "gpt2", "tl_rte_small_it_cli_test", True, True),
     ("rte_boolq", "gpt2", "lightning_tl_rte_small_noquant_test", True, False),
     ("rte_boolq", "llama2", "lightning_rte_7b_qlora_zero_shot_test_only", False, True))
)

# experiment config set aliases
gpt2_core = EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2", "rte_small_optimizer_scheduler_init", False, True)]
gpt2_core_tl = EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2", "tl_rte_small_it_cli_test", True, True)]
gpt2_l_tl = EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2", "lightning_tl_rte_small_noquant_test", True, False)]
llama2_l = EXPERIMENT_CONFIG_SETS[("rte_boolq", "llama2", "lightning_rte_7b_qlora_zero_shot_test_only", False, True)]

TEST_CONFIGS_EXAMPLES = (
    TestCfg(alias="core_gpt2_tl_compose_config", cfg=(False, None, False, gpt2_core_tl, True)),
    TestCfg(alias="core_gpt2_tl", cfg=(False, None, False, gpt2_core_tl, False)),
    TestCfg(alias="core_gpt2_tl_instantiate_only",cfg=(False, None, True, gpt2_core_tl, False),
               expected={'class_type': GPT2RTEBoolqITLensModule}),
    TestCfg(alias="core_gpt2_optim_init", cfg=(False, None, False, gpt2_core, True)),
    TestCfg(alias="lightning_gpt2_tl", cfg=(True, "test", False, gpt2_l_tl, False), marks="lightning"),
    TestCfg(alias="lightning_gpt2_tl_instantiate_only", cfg=(True, None, True, gpt2_l_tl, False),
               marks="lightning", expected={'class_type': GPT2ITLensLightningModule}),
    TestCfg(alias="lightning_llama2", cfg=(True, "test", False, llama2_l, False), marks="lightning"),
)

EXPECTED_RESULTS_EXAMPLES = {cfg.alias: cfg.expected for cfg in TEST_CONFIGS_EXAMPLES}


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

@pytest.mark.usefixtures("make_deterministic")
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
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=EXAMPLE_WARNS)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
