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
from typing import Optional, Callable, Sequence
from dataclasses import dataclass
from functools import partial

import pytest

from interpretune.protocol import Adapter
from tests.base_defaults import BaseAugTest, BaseCfg, pytest_factory
from tests.configuration import IT_GLOBAL_STATE_LOG_MODE
from tests.orchestration import parity_test
from tests.parity_acceptance.cfg_aliases import (
    cuda,
    req_det_cuda,
    test_bs1_mem,
    test_bs1_mem_nosavedt,
    bs1_warm_mem,
    w_l_tl,
)
from tests.parity_acceptance.expected import tl_parity_results, profiling_results
from tests.results import collect_results
from tests.warns import unexpected_warns, TL_CTX_WARNS, TL_LIGHTNING_CTX_WARNS

# TODO: add tl and tl_profiling bf16 tests if/when support vetted
# TODO: add tl activation checkpointing tests if/when support vetted


@dataclass(kw_only=True)
class TLParityCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.transformer_lens)
    model_src_key: Optional[str] = "cust"


@dataclass
class TLParityTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, tl_parity_results)


PARITY_TL_CONFIGS = (
    TLParityTest(alias="test_cpu_32", cfg=TLParityCfg(phase="test", model_src_key="gpt2")),
    TLParityTest(
        alias="test_cpu_32_l",
        cfg=TLParityCfg(
            phase="test",
            **w_l_tl,
        ),
        marks="lightning",
    ),
    TLParityTest(alias="test_cuda_32", cfg=TLParityCfg(phase="test", **req_det_cuda), marks="cuda"),
    TLParityTest(alias="test_cuda_32_l", cfg=TLParityCfg(phase="test", **req_det_cuda, **w_l_tl), marks="cuda_l"),
    TLParityTest(alias="train_cpu_32", cfg=TLParityCfg()),
    TLParityTest(alias="train_cpu_32_l", cfg=TLParityCfg(**w_l_tl), marks="lightning"),
    TLParityTest(alias="train_cuda_32", cfg=TLParityCfg(**req_det_cuda), marks="cuda"),
    TLParityTest(alias="train_cuda_32_l", cfg=TLParityCfg(**req_det_cuda, **w_l_tl), marks="cuda_l"),
)

EXPECTED_PARITY_TL = {cfg.alias: cfg.expected for cfg in PARITY_TL_CONFIGS}


@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(PARITY_TL_CONFIGS, unpack=False))
def test_parity_tl(recwarn, tmp_path, request, test_alias, test_cfg):
    if test_cfg.req_deterministic:
        request.getfixturevalue("make_deterministic")
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_TL[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else TL_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@dataclass
class ProfilingTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, profiling_results)


@dataclass(kw_only=True)
class TLProfileCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.transformer_lens)
    model_src_key: Optional[str] = "gpt2"


TL_PROFILING_CONFIGS = (
    ProfilingTest(alias="test_tl_profiling.test_cpu_32", cfg=TLProfileCfg(**test_bs1_mem_nosavedt), marks="optional"),
    ProfilingTest(
        alias="test_tl_profiling.test_cpu_32_l", cfg=TLProfileCfg(**w_l_tl, **test_bs1_mem_nosavedt), marks="l_optional"
    ),
    ProfilingTest(alias="test_tl_profiling.test_cuda_32", cfg=TLProfileCfg(**cuda, **test_bs1_mem), marks="cuda_prof"),
    ProfilingTest(
        alias="test_tl_profiling.test_cuda_32_l",
        cfg=TLProfileCfg(**cuda, **w_l_tl, **test_bs1_mem),
        marks="cuda_l_prof",
    ),
    # See NOTE [Transformer Lens Profiling Parity Differences], temporarily disabled
    # ProfilingTest(alias="test_tl_profiling.train_cpu_32", cfg=TLProfileCfg(**bs1_nowarm_hk_mem), marks="optional"),
    ProfilingTest(
        alias="test_tl_profiling.train_cuda_32", cfg=TLProfileCfg(**cuda, **bs1_warm_mem), marks="cuda_profci"
    ),
    # See NOTE [Transformer Lens Profiling Parity Differences], temporarily disabled
    # ProfilingTest(alias="test_tl_profiling.train_cuda_32_l",
    #               cfg=TLProfileCfg(**cuda, **w_l_tl, **bs1_warm_mem), marks="cuda_l_optional"),
)

EXPECTED_PARITY_TL_PROFILING = {cfg.alias: cfg.expected for cfg in TL_PROFILING_CONFIGS}


@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(TL_PROFILING_CONFIGS, unpack=False, fq_alias=True))
def test_tl_profiling(recwarn, tmp_path, test_alias, test_cfg):
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_TL_PROFILING[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else TL_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
