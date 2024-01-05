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
# TODO: fill in this placeholder with actual core tests
from typing import Optional, Callable, NamedTuple, Dict
from dataclasses import dataclass
from functools import partial

import pytest

from tests.helpers.warns import unexpected_warns, CORE_CONTEXT_WARNS, LIGHTING_EXPECTED_WARNS
from tests.helpers.core import (TestCfg, TestResult, config_modules, def_results, collect_results, pytest_param_factory,
                                run_it, run_lightning)
from tests.helpers.cfg_aliases import (w_lit, cuda, cuda_bf16, bf16, cuda_act, test_bs1_mem, cuda_bf16_l,
                                       bs1_nowarm_mem, bs1_nowarm_hk_mem, bs1_warm_mem, MemProfResult)


class ParityCfg(NamedTuple):
    loop_type: str = "train"
    device_type: str = "cpu"
    precision: str | int = 32
    full_dataset: bool = True
    act_ckpt: bool = False
    lightning: bool = False
    train_steps: Optional[int] = 1
    val_steps: Optional[int] = 1
    test_steps: Optional[int] = 1
    dm_override_cfg: Optional[Dict] = None
    memprofiling_cfg: Optional[Dict] = None
    from_pretrained_cfg: Optional[Dict] = None
    cust_fwd_kwargs: Optional[Dict] = None


# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these core results for all supported parity test suffixes (e.g. '_l')
parity_results = {
    "train_cpu_32": TestResult(exact_results=def_results("cpu", 32), close_results=((0, 'loss', 9.4690589),)),
    "train_cpu_32_debug": TestResult(exact_results=def_results("cpu", 32), close_results=((0, 'loss', 9.4690589),)),
    "train_cuda_32": TestResult(exact_results=def_results("cuda", 32), close_results=((0, 'loss', 7.3083372),)),
    "train_cuda_bf16": TestResult(exact_results=def_results("cuda", "bf16"), close_results=((0, 'loss', 5.3750343),)),
    "train_cpu_bf16": TestResult(exact_results=def_results("cpu", "bf16")),
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32)),
    "test_cuda_32": TestResult(exact_results=def_results("cuda", 32)),
    "test_cuda_bf16": TestResult(exact_results=def_results("cuda", "bf16")),
    "test_cpu_32_prof": TestResult(exact_results=def_results("cpu", 32), mem_results=("test", "hooks", (391647232,)),
                                   tolerance_map={"rss_diff": (0.05, 1e08)}),  # lightning ver requires a bit more
    "test_cuda_32_prof": TestResult(exact_results=def_results("cuda", 32),
                                    mem_results=("test", "cuda", (544714240, 666343424, 731906048))),
    "test_cuda_bf16_prof": TestResult(exact_results=def_results("cuda", "bf16"),
                                      mem_results=("test", "cuda", (301460992, 345495552, 362807296))),
    "train_cpu_32_prof": TestResult(exact_results=def_results("cpu", 32),
                                    mem_results=("train", "hooks", (513175552,))),
    "train_cpu_32_prof_act": TestResult(exact_results=def_results("cpu", 32),
                                        mem_results=("train", "hooks", (385560576,))),
    "train_cuda_32_prof": TestResult(exact_results=def_results("cuda", 32),
                                     mem_results=("train", "cuda", (1939940352, 2579284992, 2862612480))),
    "train_cuda_32_prof_act": TestResult(exact_results=def_results("cuda", 32),
                                         mem_results=("train", "cuda", (1587114496, 2577159168, 2791309312))),
    "train_cuda_bf16_prof": TestResult(exact_results=def_results("cuda", "bf16"),
                                       mem_results=("train", "cuda", (1132208128, 1363834880, 1491075072)),
                                       tolerance_map={k: (0.1, 1e08) for k in MemProfResult.cuda_mem_keys}),
}

@dataclass
class ParityTest(TestCfg):
    result_gen: Optional[Callable] = partial(collect_results, parity_results)


# we use a single set of results but separate tests for core/lightning parity tests since Lightning is not a required
# dependency for Interpretune and we want to mark at the test-level for greater clarity and flexibility (we want to
# signal clearly when either diverges from the expected benchmark so aren't testing relative values only)
# we are sensibly sampling the configuration space rather than exhaustively testing all framework configuration
# combinations due to resource constraints
# note that while we could access test_alias using the request fixture (request.node.callspec.id), this approach
# allows us to flexibly define test ids, configurations, marks and expected outputs together
PARITY_SINGLE_DEVICE_CONFIGS = (
    ParityTest(alias="train_cpu_32", cfg=ParityCfg()),
    ParityTest(alias="train_cpu_32_l", cfg=ParityCfg(**w_lit), marks="lightning"),
    #ParityTest(alias="train_cpu_32_debug", cfg=ParityCfg(**debug_hidden)),
    ParityTest(alias="train_cuda_32", cfg=ParityCfg(**cuda), marks="cuda"),
    ParityTest(alias="train_cuda_32_l", cfg=ParityCfg(**cuda, **w_lit), marks="cuda_l"),
    ParityTest(alias="train_cuda_bf16", cfg=ParityCfg(**cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cuda_bf16_l", cfg=ParityCfg(**cuda_bf16_l), marks="bf16_cuda_l"),
    ParityTest(alias="test_cpu_32", cfg=ParityCfg("test")),
    ParityTest(alias="test_cpu_32_l", cfg=ParityCfg("test", **w_lit), marks="lightning"),
    ParityTest(alias="test_cuda_32", cfg=ParityCfg("test", **cuda), marks="cuda"),
    ParityTest(alias="test_cuda_bf16", cfg=ParityCfg("test", **cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cpu_bf16", cfg=ParityCfg(**bf16), marks="skip_win_slow"),
    ParityTest(alias="test_cpu_32_prof", cfg=ParityCfg(**test_bs1_mem), marks="prof"),
    ParityTest(alias="test_cpu_32_prof_l", cfg=ParityCfg(**w_lit, **test_bs1_mem), marks="prof_l"),
    ParityTest(alias="test_cuda_32_prof", cfg=ParityCfg(**cuda, **test_bs1_mem), marks="cuda_prof"),
    ParityTest(alias="test_cuda_32_prof_l", cfg=ParityCfg(**cuda, **w_lit, **test_bs1_mem), marks="cuda_prof_l"),
    ParityTest(alias="test_cuda_bf16_prof", cfg=ParityCfg(**cuda_bf16, **test_bs1_mem), marks="bf16_cuda_prof"),
    ParityTest(alias="train_cpu_32_prof", cfg=ParityCfg(**bs1_nowarm_hk_mem), marks="prof"),
    ParityTest(alias="train_cpu_32_prof_act", cfg=ParityCfg(act_ckpt=True, **bs1_nowarm_mem), marks="prof",),
    ParityTest(alias="train_cuda_32_prof", cfg=ParityCfg(**cuda, **bs1_warm_mem), marks="cuda_prof"),
    ParityTest(alias="train_cuda_32_prof_act", cfg=ParityCfg(**cuda_act, **bs1_warm_mem), marks="cuda_prof"),
    ParityTest(alias="train_cuda_bf16_prof", cfg=ParityCfg(**cuda_bf16, **bs1_warm_mem), marks="bf16_cuda_prof"),
    ParityTest(alias="train_cuda_bf16_prof_l", cfg=ParityCfg(**cuda_bf16_l, **bs1_warm_mem), marks="bf16_cuda_prof_l"),
)

EXPECTED_RESULTS_PARITY_SINGLE_DEVICE = {cfg.alias: cfg.expected for cfg in PARITY_SINGLE_DEVICE_CONFIGS}

@pytest.mark.usefixtures("reset_deterministic_algorithm")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PARITY_SINGLE_DEVICE_CONFIGS, unpack=False))
def test_parity_single_device(recwarn, tmp_path, test_alias, test_cfg):
    expected_results = EXPECTED_RESULTS_PARITY_SINGLE_DEVICE[test_alias] or {}
    expected_warnings = LIGHTING_EXPECTED_WARNS if test_cfg.lightning else CORE_CONTEXT_WARNS
    datamodule, module = config_modules(test_cfg, expected_results, tmp_path)
    if test_cfg.lightning:
        _ = run_lightning(module, datamodule, test_cfg, tmp_path)
    else:
        run_it(module, datamodule, test_cfg)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
