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
from typing import Optional, Callable
from dataclasses import dataclass
from functools import partial

import pytest

from interpretune.protocol import Adapter
from tests.base_defaults import BaseAugTest, BaseCfg, pytest_factory
from tests.configuration import IT_GLOBAL_STATE_LOG_MODE
from tests.results import collect_results
from tests.orchestration import parity_test
from tests.parity_acceptance.expected import l_parity_results, profiling_results
from tests.parity_acceptance.cfg_aliases import (
    w_lit,
    req_det,
    req_det_l,
    req_det_cuda,
    req_det_cuda_l,
    req_det_cuda_bf16,
    req_det_cuda_bf16_l,
    cuda,
    cuda_bf16,
    bf16,
    cuda_act,
    test_bs1_mem,
    cuda_bf16_l,
    test_bs1_mem_nosavedt,
    bs1_nowarm_mem,
    act_ckpt,
    bs1_nowarm_hk_mem,
    bs1_warm_mem,
)
from tests.warns import unexpected_warns, CORE_CTX_WARNS, LIGHTING_CTX_WARNS


@dataclass(kw_only=True)
class CoreCfg(BaseCfg):
    model_src_key: Optional[str] = "cust"


@dataclass
class ParityTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, l_parity_results)


PARITY_BASIC_CONFIGS = (
    ParityTest(alias="train_cpu_32", cfg=CoreCfg(**req_det)),
    ParityTest(alias="train_cpu_32_l", cfg=CoreCfg(**req_det_l), marks="lightning"),
    ParityTest(alias="train_cuda_32", cfg=CoreCfg(**req_det_cuda), marks="cuda"),
    ParityTest(alias="train_cuda_32_l", cfg=CoreCfg(**req_det_cuda_l), marks="cuda_l"),
    ParityTest(alias="train_cuda_bf16", cfg=CoreCfg(**req_det_cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cuda_bf16_l", cfg=CoreCfg(**req_det_cuda_bf16_l), marks="bf16_cuda_l"),
    ParityTest(alias="test_cpu_32", cfg=CoreCfg(phase="test")),
    ParityTest(alias="test_cpu_32_l", cfg=CoreCfg(phase="test", **w_lit), marks="lightning"),
    ParityTest(alias="predict_cpu_32_l", cfg=CoreCfg(phase="predict", **w_lit), marks="lightning"),
    ParityTest(alias="test_cuda_32", cfg=CoreCfg(phase="test", model_src_key="gpt2", **cuda), marks="cuda"),
    ParityTest(alias="test_cuda_bf16", cfg=CoreCfg(phase="test", **cuda_bf16), marks="bf16_cuda"),
    ParityTest(alias="train_cpu_bf16", cfg=CoreCfg(**bf16), marks="skip_win_optional"),
)

EXPECTED_PARITY_BASIC = {cfg.alias: cfg.expected for cfg in PARITY_BASIC_CONFIGS}


@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(PARITY_BASIC_CONFIGS, unpack=False))
def test_parity_l(recwarn, tmp_path, request, test_alias, test_cfg):
    if test_cfg.req_deterministic:
        request.getfixturevalue("make_deterministic")
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_BASIC[test_alias] or {}
    expected_warnings = LIGHTING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else CORE_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@dataclass(kw_only=True)
class ProfParityCfg(BaseCfg):
    model_src_key: Optional[str] = "gpt2"
    force_prepare_data: bool = True  # force data preparation for profiling and CI runner cache reproduction


@dataclass
class ProfilingTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, profiling_results)


L_PROFILING_CONFIGS = (
    ProfilingTest(
        alias="test_l_profiling.test_cpu_32", cfg=ProfParityCfg(**test_bs1_mem_nosavedt), marks="profiling_ci"
    ),
    ProfilingTest(
        alias="test_l_profiling.test_cpu_32_l",
        cfg=ProfParityCfg(**w_lit, **test_bs1_mem_nosavedt),
        marks="lightning_prof",
    ),
    ProfilingTest(alias="test_l_profiling.test_cuda_32", cfg=ProfParityCfg(**cuda, **test_bs1_mem), marks="cuda_prof"),
    ProfilingTest(
        alias="test_l_profiling.test_cuda_32_l",
        cfg=ProfParityCfg(**cuda, **w_lit, **test_bs1_mem),
        marks="cuda_l_profci",
    ),
    ProfilingTest(
        alias="test_l_profiling.test_cuda_bf16", cfg=ProfParityCfg(**cuda_bf16, **test_bs1_mem), marks="bf16_cuda_prof"
    ),
    ProfilingTest(alias="test_l_profiling.train_cpu_32", cfg=ProfParityCfg(**bs1_nowarm_hk_mem), marks="optional"),
    ProfilingTest(
        alias="test_l_profiling.train_cpu_32_act", cfg=ProfParityCfg(**act_ckpt, **bs1_nowarm_mem), marks="prof"
    ),
    ProfilingTest(alias="test_l_profiling.train_cuda_32", cfg=ProfParityCfg(**cuda, **bs1_warm_mem), marks="cuda_prof"),
    ProfilingTest(
        alias="test_l_profiling.train_cuda_32_act", cfg=ProfParityCfg(**cuda_act, **bs1_warm_mem), marks="cuda_profci"
    ),
    ProfilingTest(
        alias="test_l_profiling.train_cuda_bf16",
        cfg=ProfParityCfg(**cuda_bf16, **bs1_warm_mem),
        marks="bf16_cuda_profci",
    ),
    ProfilingTest(
        alias="test_l_profiling.train_cuda_bf16_l",
        cfg=ProfParityCfg(**cuda_bf16_l, **bs1_warm_mem),
        marks="bf16_cuda_l_prof",
    ),
)

EXPECTED_PROFILING_PARITY = {cfg.alias: cfg.expected for cfg in L_PROFILING_CONFIGS}


@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(L_PROFILING_CONFIGS, unpack=False, fq_alias=True))
def test_l_profiling(recwarn, tmp_path, test_alias, test_cfg):
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PROFILING_PARITY[test_alias] or {}
    expected_warnings = LIGHTING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else CORE_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
