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
from typing import Optional, Callable, Dict
from dataclasses import dataclass, field
from functools import partial

import pytest

from tests.utils.warns import unexpected_warns, CORE_CONTEXT_WARNS, LIGHTING_CONTEXT_WARNS
from tests.orchestration import parity_test
from tests.base.expected import basic_parity_results, profiling_parity_results
from tests.configuration import TestCfg, ParityCfg, pytest_param_factory, collect_results
from tests.base.cfg_aliases import (w_lit, cuda, cuda_bf16, bf16, cuda_act, test_bs1_mem, cuda_bf16_l, debug_hidden,
                                    test_bs1_mem_nosavedt, bs1_nowarm_mem, bs1_nowarm_hk_mem, bs1_warm_mem)


@dataclass
class ParityTest(TestCfg):
    result_gen: Optional[Callable] = partial(collect_results, basic_parity_results)


PARITY_BASIC_CONFIGS = (
    ParityTest(alias="train_cpu_32", cfg=ParityCfg()),
    ParityTest(alias="train_cpu_32_l", cfg=ParityCfg(**w_lit), marks="lightning"),
    ParityTest(alias="train_cpu_32_debug", cfg=ParityCfg(**debug_hidden), marks="optional"),
    ParityTest(alias="train_cuda_32", cfg=ParityCfg(**cuda), marks="cuda"),
    ParityTest(alias="train_cuda_32_l", cfg=ParityCfg(**cuda, **w_lit), marks="cuda_l"),
    ParityTest(alias="train_cuda_bf16", cfg=ParityCfg(**cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cuda_bf16_l", cfg=ParityCfg(**cuda_bf16_l), marks="bf16_cuda_l"),
    ParityTest(alias="test_cpu_32", cfg=ParityCfg("test")),
    ParityTest(alias="test_cpu_32_l", cfg=ParityCfg("test", **w_lit), marks="lightning"),
    ParityTest(alias="test_cuda_32", cfg=ParityCfg("test", **cuda), marks="cuda"),
    ParityTest(alias="test_cuda_bf16", cfg=ParityCfg("test", **cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cpu_bf16", cfg=ParityCfg(**bf16), marks="skip_win_slow"),
)

EXPECTED_PARITY_BASIC = {cfg.alias: cfg.expected for cfg in PARITY_BASIC_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PARITY_BASIC_CONFIGS, unpack=False))
def test_parity_basic(recwarn, tmp_path, test_alias, test_cfg):
    expected_results = EXPECTED_PARITY_BASIC[test_alias] or {}
    expected_warnings = LIGHTING_CONTEXT_WARNS if test_cfg.lightning else CORE_CONTEXT_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=False)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@dataclass
class ProfilingTest(TestCfg):
    result_gen: Optional[Callable] = partial(collect_results, profiling_parity_results)
    # See NOTE [Profiling and Standalone Marks]
    function_marks: Dict = field(default_factory=lambda: {'profiling': True})

PROFILING_PARITY_CONFIGS = (
    ProfilingTest(alias="test_cpu_32", cfg=ParityCfg(**test_bs1_mem_nosavedt), marks="standalone"),
    ProfilingTest(alias="test_cpu_32_l", cfg=ParityCfg(**w_lit, **test_bs1_mem_nosavedt), marks="lightning"),
    ProfilingTest(alias="test_cuda_32", cfg=ParityCfg(**cuda, **test_bs1_mem), marks="cuda"),
    ProfilingTest(alias="test_cuda_32_l", cfg=ParityCfg(**cuda, **w_lit, **test_bs1_mem), marks="cuda_l_alone"),
    ProfilingTest(alias="test_cuda_bf16", cfg=ParityCfg(**cuda_bf16, **test_bs1_mem), marks="bf16_cuda"),
    ProfilingTest(alias="train_cpu_32", cfg=ParityCfg(**bs1_nowarm_hk_mem)),
    ProfilingTest(alias="train_cpu_32_act", cfg=ParityCfg(act_ckpt=True, **bs1_nowarm_mem)),
    ProfilingTest(alias="train_cuda_32", cfg=ParityCfg(**cuda, **bs1_warm_mem), marks="cuda"),
    ProfilingTest(alias="train_cuda_32_act", cfg=ParityCfg(**cuda_act, **bs1_warm_mem), marks="cuda_alone"),
    ProfilingTest(alias="train_cuda_bf16", cfg=ParityCfg(**cuda_bf16, **bs1_warm_mem), marks="bf16_cuda_alone"),
    ProfilingTest(alias="train_cuda_bf16_l", cfg=ParityCfg(**cuda_bf16_l, **bs1_warm_mem), marks="bf16_cuda_l"),
)

EXPECTED_PROFILING_PARITY = {cfg.alias: cfg.expected for cfg in PROFILING_PARITY_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PROFILING_PARITY_CONFIGS, unpack=False))
def test_parity_profiling(recwarn, tmp_path, test_alias, test_cfg):
    expected_results = EXPECTED_PROFILING_PARITY[test_alias] or {}
    expected_warnings = LIGHTING_CONTEXT_WARNS if test_cfg.lightning else CORE_CONTEXT_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=False)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
