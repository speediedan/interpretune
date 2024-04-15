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
from typing import Optional, Callable, Dict
from dataclasses import dataclass, field
from functools import partial

import pytest

from tests.utils.warns import unexpected_warns, CORE_CTX_WARNS, LIGHTING_CTX_WARNS
from tests.orchestration import parity_test
from interpretune.base.contract.session import Framework
from tests.parity_acceptance.base.expected import basic_parity_results, profiling_parity_results
from tests.configuration import BaseAugTest, BaseCfg, pytest_param_factory, collect_results, IT_GLOBAL_STATE_LOG_MODE
from tests.parity_acceptance.base.cfg_aliases import (w_lit, cuda, cuda_bf16, bf16, cuda_act, test_bs1_mem, cuda_bf16_l,
                                                      debug_hidden, test_bs1_mem_nosavedt, bs1_nowarm_mem,
                                                      bs1_nowarm_hk_mem, bs1_warm_mem)


@dataclass(kw_only=True)
class CoreCfg(BaseCfg):
    model_src_key: Optional[str] = "cust"

@dataclass
class ParityTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, basic_parity_results)


PARITY_BASIC_CONFIGS = (
    ParityTest(alias="train_cpu_32", cfg=CoreCfg()),
    ParityTest(alias="train_cpu_32_l", cfg=CoreCfg(**w_lit), marks="lightning"),
    ParityTest(alias="train_cpu_32_debug", cfg=CoreCfg(**debug_hidden), marks="optional"),
    ParityTest(alias="train_cuda_32", cfg=CoreCfg(**cuda), marks="cuda"),
    ParityTest(alias="train_cuda_32_l", cfg=CoreCfg(**cuda, **w_lit), marks="cuda_l"),
    ParityTest(alias="train_cuda_bf16", cfg=CoreCfg(**cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cuda_bf16_l", cfg=CoreCfg(**cuda_bf16_l), marks="bf16_cuda_l"),
    ParityTest(alias="test_cpu_32", cfg=CoreCfg(phase="test")),
    ParityTest(alias="test_cpu_32_l", cfg=CoreCfg(phase="test", **w_lit), marks="lightning"),
    ParityTest(alias="test_cuda_32", cfg=CoreCfg(phase="test", model_src_key="pretrained", **cuda), marks="cuda"),
    ParityTest(alias="test_cuda_bf16", cfg=CoreCfg(phase="test", **cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cpu_bf16", cfg=CoreCfg(**bf16), marks="skip_win_slow"),
)

EXPECTED_PARITY_BASIC = {cfg.alias: cfg.expected for cfg in PARITY_BASIC_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PARITY_BASIC_CONFIGS, unpack=False))
def test_parity_basic(recwarn, tmp_path, test_alias, test_cfg):
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_BASIC[test_alias] or {}
    expected_warnings = LIGHTING_CTX_WARNS if test_cfg.framework_ctx == Framework.lightning else CORE_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@dataclass(kw_only=True)
class ProfParityCfg(BaseCfg):
    model_src_key: Optional[str] = "pretrained"

@dataclass
class ProfilingTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, profiling_parity_results)
    # See NOTE [Profiling and Standalone Marks]
    function_marks: Dict = field(default_factory=lambda: {'profiling': True})

PROFILING_PARITY_CONFIGS = (
    ProfilingTest(alias="test_cpu_32", cfg=ProfParityCfg(**test_bs1_mem_nosavedt), marks="profiling_ci"),
    ProfilingTest(alias="test_cpu_32_l", cfg=ProfParityCfg(**w_lit, **test_bs1_mem_nosavedt), marks="lightning"),
    ProfilingTest(alias="test_cuda_32", cfg=ProfParityCfg(**cuda, **test_bs1_mem), marks="cuda"),
    ProfilingTest(alias="test_cuda_32_l", cfg=ProfParityCfg(**cuda, **w_lit, **test_bs1_mem), marks="cuda_l_profci"),
    ProfilingTest(alias="test_cuda_bf16", cfg=ProfParityCfg(**cuda_bf16, **test_bs1_mem), marks="bf16_cuda"),
    ProfilingTest(alias="train_cpu_32", cfg=ProfParityCfg(**bs1_nowarm_hk_mem), marks="optional"),
    ProfilingTest(alias="train_cpu_32_act", cfg=ProfParityCfg(act_ckpt=True, **bs1_nowarm_mem)),
    ProfilingTest(alias="train_cuda_32", cfg=ProfParityCfg(**cuda, **bs1_warm_mem), marks="cuda"),
    ProfilingTest(alias="train_cuda_32_act", cfg=ProfParityCfg(**cuda_act, **bs1_warm_mem), marks="cuda_profci"),
    ProfilingTest(alias="train_cuda_bf16", cfg=ProfParityCfg(**cuda_bf16, **bs1_warm_mem), marks="bf16_cuda_profci"),
    ProfilingTest(alias="train_cuda_bf16_l", cfg=ProfParityCfg(**cuda_bf16_l, **bs1_warm_mem), marks="bf16_cuda_l"),
)

EXPECTED_PROFILING_PARITY = {cfg.alias: cfg.expected for cfg in PROFILING_PARITY_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PROFILING_PARITY_CONFIGS, unpack=False))
def test_parity_profiling(recwarn, tmp_path, test_alias, test_cfg):
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PROFILING_PARITY[test_alias] or {}
    expected_warnings = LIGHTING_CTX_WARNS if test_cfg.framework_ctx == Framework.lightning else CORE_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
