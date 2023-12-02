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

import pytest

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.utils.cli import compose_config, IT_CONFIG_BASE
from it_examples.experiments.rte_boolq.core import GPT2RTEBoolqITHookedModule
from tests.helpers.core.boring_model import (unexpected_warns, pytest_param_factory, TestConfig, RUN_FN,
                                             dummy_step, EXPECTED_WARNS,
                                             #EXPERIMENT_CONFIGS,
                                             EXPERIMENT_CONFIG_SETS)
from tests.helpers.runif import RunIf

if _LIGHTNING_AVAILABLE:
    from interpretune.base.it_lightning_modules import ITLightningModule
    from it_examples.experiments.rte_boolq.lightning import GPT2ITHookedLightningModule, Llama2ITLightningModule
else:
    ITLightningModule = None  # type: ignore[misc,assignment]
    GPT2ITHookedLightningModule = None  # type: ignore[misc,assignment]
    Llama2ITLightningModule = None  # type: ignore[misc,assignment]


RTEBOOLQ_TEST_CONFIGS = (
    TestConfig("core_gpt2_hooked_compose_config",
               test_config=(False, None, False, EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2_core_hooked")], True)),
    TestConfig("core_gpt2_hooked",
               test_config=(False, None, False, EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2_core_hooked")], False)),
    TestConfig("core_gpt2_hooked_instantiate_only",
               test_config=(False, None, True, EXPERIMENT_CONFIG_SETS[("rte_boolq", "gpt2_core_hooked")], False),
               expected_results=(GPT2RTEBoolqITHookedModule,)),
    TestConfig("lightning_gpt2_hooked",
               test_config=(True, "test", False, EXPERIMENT_CONFIG_SETS["rte_boolq", "gpt2_lightning_hooked"], False),
               marks="lightning"),
    TestConfig("lightning_gpt2_hooked_instantiate_only",
               test_config=(True, None, True, EXPERIMENT_CONFIG_SETS["rte_boolq", "gpt2_lightning_hooked"], False),
               marks="lightning",
               expected_results=(GPT2ITHookedLightningModule,)),
    TestConfig("lightning_llama2",
               test_config=(True, "test", False, EXPERIMENT_CONFIG_SETS["rte_boolq", "llama2_lightning"], False),
               marks="lightning"),
               #expected_results=(Llama2ITLightningModule,)),
)

EXPECTED_RESULTS_RTEBOOLQ = {cfg.test_alias: cfg.expected_results for cfg in RTEBOOLQ_TEST_CONFIGS}


@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("test_alias, lightning_cli, subcommand, instantiate_only, config_files, use_compose_config",
                         pytest_param_factory(RTEBOOLQ_TEST_CONFIGS))
def test_rteboolq(recwarn, test_alias, lightning_cli, subcommand, instantiate_only, config_files, use_compose_config):
    # TODO: refactor these conditional evaluations into functions resuable in other tests
    if not lightning_cli:
        from interpretune.utils.cli import cli_main
        cli_args = [RUN_FN]
    else:
        from interpretune.utils.lightning_cli import cli_main
        cli_args = [RUN_FN, subcommand] if subcommand else [RUN_FN]
    if use_compose_config:
        cli_args.extend(compose_config(config_files))
    else:
        for f in config_files:
            cli_args.extend(["--config", os.path.join(IT_CONFIG_BASE, f)])
    with mock.patch('sys.argv', cli_args):
        if lightning_cli and not instantiate_only:
            # TODO: once examples are real, remove this `dummy_step` patch and just run a small number of real steps
            with mock.patch.object(ITLightningModule, "test_step", dummy_step):
                cli = cli_main(instantiate_only=instantiate_only)
        else:
            cli = cli_main(instantiate_only=instantiate_only)
    if instantiate_only:
        assert isinstance(cli.model, EXPECTED_RESULTS_RTEBOOLQ[test_alias][0])
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=EXPECTED_WARNS)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
