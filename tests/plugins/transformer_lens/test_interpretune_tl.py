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

from tests.utils.warns import unexpected_warns, TL_CONTEXT_WARNS, TL_LIGHTNING_CONTEXT_WARNS
from tests.configuration import TestCfg, collect_results, ParityCfg, pytest_param_factory
from tests.orchestration import parity_test
from tests.plugins.transformer_lens.expected import tl_parity_results, tl_profiling_parity_results
from tests.base.cfg_aliases import (w_lit, cuda,test_bs1_mem, test_bs1_mem_nosavedt, bs1_nowarm_hk_mem, bs1_warm_mem,
                                    debug_hidden)


# TODO: add tl and tl_profiling bf16 tests if/when support vetted
# TODO: add tl activation checkpointing tests if/when support vetted

class TLParityCfg(ParityCfg):
    plugin: Optional[str] = "transformerlens"

@dataclass
class TLParityTest(TestCfg):
    result_gen: Optional[Callable] = partial(collect_results, tl_parity_results)


PARITY_TL_CONFIGS = (
    TLParityTest(alias="test_cpu_32", cfg=TLParityCfg("test")),
    TLParityTest(alias="test_cpu_32_l", cfg=TLParityCfg("test", **w_lit), marks="lightning"),
    TLParityTest(alias="test_cuda_32", cfg=TLParityCfg("test", **cuda), marks="cuda"),
    TLParityTest(alias="test_cuda_32_l", cfg=TLParityCfg("test", **cuda, **w_lit), marks="cuda_l"),
    TLParityTest(alias="train_cpu_32", cfg=TLParityCfg()),
    TLParityTest(alias="train_cpu_32_l", cfg=TLParityCfg(**w_lit), marks="lightning"),
    TLParityTest(alias="train_cpu_32_debug", cfg=ParityCfg(**debug_hidden), marks="optional"),
    TLParityTest(alias="train_cuda_32", cfg=TLParityCfg(**cuda), marks="cuda"),
    TLParityTest(alias="train_cuda_32_l", cfg=TLParityCfg(**cuda, **w_lit), marks="cuda_l"),
)

EXPECTED_PARITY_TL = {cfg.alias: cfg.expected for cfg in PARITY_TL_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PARITY_TL_CONFIGS, unpack=False))
def test_parity_tl(recwarn, tmp_path, test_alias, test_cfg):
    expected_results = EXPECTED_PARITY_TL[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CONTEXT_WARNS if test_cfg.lightning else TL_CONTEXT_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=False)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@dataclass
class ProfilingTest(TestCfg):
    result_gen: Optional[Callable] = partial(collect_results, tl_profiling_parity_results)
    # See NOTE [Profiling and Standalone Marks]
    function_marks: Dict = field(default_factory=lambda: {'profiling': False})

PARITY_TL_PROFILING_CONFIGS = (
    ProfilingTest(alias="test_cpu_32", cfg=TLParityCfg(**test_bs1_mem_nosavedt), marks="standalone"),
    ProfilingTest(alias="test_cpu_32_l", cfg=TLParityCfg(**w_lit, **test_bs1_mem_nosavedt), marks="lightning"),
    ProfilingTest(alias="test_cuda_32", cfg=TLParityCfg(**cuda, **test_bs1_mem), marks="cuda"),
    ProfilingTest(alias="test_cuda_32_l", cfg=TLParityCfg(**cuda, **w_lit, **test_bs1_mem), marks="cuda_l"),
    ProfilingTest(alias="train_cpu_32", cfg=TLParityCfg(**bs1_nowarm_hk_mem), marks="standalone"),
    ProfilingTest(alias="train_cuda_32", cfg=TLParityCfg(**cuda, **bs1_warm_mem), marks="cuda"),
    ProfilingTest(alias="train_cuda_32_l", cfg=TLParityCfg(**cuda, **w_lit, **bs1_warm_mem), marks="cuda_l_alone"),
)

EXPECTED_PARITY_TL_PROFILING = {cfg.alias: cfg.expected for cfg in PARITY_TL_PROFILING_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PARITY_TL_PROFILING_CONFIGS, unpack=False))
def test_parity_tl_profiling(recwarn, tmp_path, test_alias, test_cfg):
    expected_results = EXPECTED_PARITY_TL_PROFILING[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CONTEXT_WARNS if test_cfg.lightning else TL_CONTEXT_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=True)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
