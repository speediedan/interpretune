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
# TODO: re-enable one or more applied interpretune examples once they are ready
# import os.path
# from unittest import mock
#from collections.abc import Iterable
# from typing import Dict, Tuple, Optional
# import pytest

# from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
# from interpretune.base.cli.core_cli import compose_config, IT_CONFIG_BASE, IT_CONFIG_GLOBAL
# from interpretune.adapters.transformer_lens import ITLensModule
# from tests.configuration import pytest_param_factory, TestCfg
# from tests.utils import dummy_step
# from tests.runif import RunIf
# from tests.warns import EXPECTED_WARNS, HF_EXPECTED_WARNS, unexpected_warns, TL_EXPECTED_WARNS
# from tests.conftest import make_deterministic

# if _LIGHTNING_AVAILABLE:
#     from interpretune.base.modules import ITLightningModule
#     from interpretune.adapters.transformer_lens import ITLensLightningModule
# else:
#     ITLightningModule = None  # type: ignore[misc,assignment]
#     ITLensLightningModule = None  # type: ignore[misc,assignment]

# EXAMPLE_WARNS = EXPECTED_WARNS + HF_EXPECTED_WARNS + TL_EXPECTED_WARNS

# RUN_FN = "run_experiment.py"
# EXPERIMENTS_BASE = IT_CONFIG_BASE / "experiments"
# BASE_DEBUG_CONFIG = IT_CONFIG_GLOBAL / "base_debug.yaml"
# BASE_TL_CONFIG = IT_CONFIG_GLOBAL / "base_transformer_lens.yaml"

# def gen_experiment_cfg_sets(test_keys: Iterable[Tuple[str, str, str, Optional[str], bool]]) -> Dict:
#     exp_cfg_sets = {}
#     for exp, model, subexp, adapter_ctx, debug_mode in test_keys:
#         base_model_cfg =  EXPERIMENTS_BASE / exp / f"{model}.yaml"
#         base_cfg_set = (base_model_cfg,)
#         if adapter_ctx:
#             if adapter_ctx == "transformer_lens":
#                 exp_plugin_cfg = EXPERIMENTS_BASE / exp /  f"{adapter_ctx}.yaml"
#                 base_cfg_set += (BASE_TL_CONFIG, exp_plugin_cfg,)
#             else:
#                 raise ValueError(f"Unknown adapter type: {adapter_ctx}")
#         subexp_cfg =  EXPERIMENTS_BASE / exp / model / f"{subexp}.yaml"
#         base_cfg_set += (subexp_cfg,)
#         if debug_mode:
#             base_cfg_set += (BASE_DEBUG_CONFIG,)
#         exp_cfg_sets[(exp, model, subexp, adapter_ctx, debug_mode)] = base_cfg_set
#     return exp_cfg_sets

# EXPERIMENT_CFG_SETS = gen_experiment_cfg_sets(
#     # (exp, model, subexp, adapter_ctx, debug_mode)
#     (("rte_boolq", "gpt2", "rte_small_optimizer_scheduler_init", None, True),
#      ("rte_boolq", "gpt2", "tl_rte_small_it_cli_test", "transformer_lens", True),
#      ("rte_boolq", "gpt2", "lightning_tl_rte_small_noquant_test", "transformer_lens", False),
#      ("rte_boolq", "llama3", "lightning_rte_7b_qlora_zero_shot_test_only", None, True))
# )

# # experiment config set aliases
# gpt2_core = EXPERIMENT_CFG_SETS[("rte_boolq", "gpt2", "rte_small_optimizer_scheduler_init", None, True)]
# gpt2_core_tl = EXPERIMENT_CFG_SETS[("rte_boolq", "gpt2", "tl_rte_small_it_cli_test", "transformer_lens", True)]
# gpt2_l_tl = EXPERIMENT_CFG_SETS[("rte_boolq", "gpt2", "lightning_tl_rte_small_noquant_test", "transformer_lens",
#                                  False)]
# llama3_l = EXPERIMENT_CFG_SETS[("rte_boolq", "llama3", "lightning_rte_7b_qlora_zero_shot_test_only", None, True)]

# TEST_CONFIGS_EXAMPLES = (
#     TestCfg(alias="core_gpt2_tl_compose_config", cfg=(False, None, False, gpt2_core_tl, True)),
#     TestCfg(alias="core_gpt2_tl", cfg=(False, None, False, gpt2_core_tl, False), marks="optional"),
#     TestCfg(alias="core_gpt2_tl_instantiate_only",cfg=(False, None, True, gpt2_core_tl, False),
#                expected={'class_type': ITLensModule}),
#     TestCfg(alias="core_gpt2_optim_init", cfg=(False, None, False, gpt2_core, True)),
#     TestCfg(alias="lightning_gpt2_tl", cfg=(True, "test", False, gpt2_l_tl, False), marks="lightning"),
#     TestCfg(alias="lightning_gpt2_tl_instantiate_only", cfg=(True, None, True, gpt2_l_tl, False),
#                marks="l_optional", expected={'class_type': ITLensLightningModule}),
#     TestCfg(alias="lightning_llama3", cfg=(True, "test", False, llama3_l, False), marks="lightning"),
# )

# EXPECTED_RESULTS_EXAMPLES = {cfg.alias: cfg.expected for cfg in TEST_CONFIGS_EXAMPLES}


# def gen_cli_args(l_cli, subcommand, compose_cfg, config_files):
#     if not l_cli:
#         from interpretune.base.cli.core_cli import core_cli_main
#         cli_args = [RUN_FN]
#     else:
#         from interpretune.base.cli.lightning_cli import l_cli_main
#         cli_args = [RUN_FN, subcommand] if subcommand else [RUN_FN]
#     if compose_cfg:
#         cli_args.extend(compose_config(config_files))
#     else:
#         for f in config_files:
#             cli_args.extend(["--config", os.path.join(IT_CONFIG_BASE, f)])
#     return core_cli_main, cli_args

# def invoke_cli(cli_main, cli_args, l_cli, instantiate_only):
#     with mock.patch('sys.argv', cli_args):
#         if l_cli and not instantiate_only:
#             # TODO: once examples are real, remove this `dummy_step` patch and just run a small number of real steps
#             with mock.patch.object(ITLightningModule, "test_step", dummy_step):
#                 cli = cli_main(instantiate_only=instantiate_only)
#         else:
#             cli = cli_main(instantiate_only=instantiate_only)
#     return cli

# @pytest.mark.usefixtures("make_deterministic")
# @RunIf(min_cuda_gpus=1, skip_windows=True)
# @pytest.mark.parametrize("test_alias, l_cli, subcommand, instantiate_only, config_files, use_compose_config",
#                          pytest_param_factory(TEST_CONFIGS_EXAMPLES))
# def test_basic_examples(recwarn, test_alias, l_cli, subcommand, instantiate_only, config_files, use_compose_config):
#     # TODO: refactor these conditional evaluations into functions resuable in other tests
#     cli_main, cli_args = gen_cli_args(l_cli, subcommand, use_compose_config, config_files)
#     cli = invoke_cli(cli_main, cli_args, l_cli, instantiate_only)
#     if instantiate_only:
#         assert isinstance(cli.module, EXPECTED_RESULTS_EXAMPLES[test_alias]['class_type'])
#     # ensure no unexpected warnings detected
#     unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=EXAMPLE_WARNS)
#     assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
