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
from typing import Optional, Callable
from dataclasses import dataclass
from functools import partial

import pytest

from tests.utils.warns import unexpected_warns, TL_CONTEXT_WARNS, TL_LIGHTNING_CONTEXT_WARNS
from tests.configuration import (TestCfg, TestResult, config_modules, def_results, collect_results, ParityCfg,
                                 pytest_param_factory)
from tests.orchestration import run_it, run_lightning
from tests.base.cfg_aliases import (w_lit, cuda)


tl_results = partial(def_results, dataset_type="tl")
# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these core results for all supported parity test suffixes (e.g. '_l')
tl_parity_results = {
    "test_cpu_32": TestResult(exact_results=tl_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=tl_results("cuda", 32, ds_cfg="test")),
    "train_cpu_32": TestResult(exact_results=tl_results("cpu", 32, ds_cfg="train")),
    "train_cuda_32": TestResult(exact_results=tl_results("cuda", 32, ds_cfg="train")),
}

class TLParityCfg(ParityCfg):
    transformerlens: bool = True

@dataclass
class TLParityTest(TestCfg):
    result_gen: Optional[Callable] = partial(collect_results, tl_parity_results)

# we use a single set of results but separate tests for core/lightning parity tests since Lightning is not a required
# dependency for Interpretune and we want to mark at the test-level for greater clarity and flexibility (we want to
# signal clearly when either diverges from the expected benchmark so aren't testing relative values only)
# we are sensibly sampling the configuration space rather than exhaustively testing all framework configuration
# combinations due to resource constraints
# note that while we could access test_alias using the request fixture (`request.node.callspec.id`), this approach
# allows us to flexibly define test ids, configurations, marks and expected outputs together
TL_PARITY_CONFIGS = (
    TLParityTest(alias="test_cpu_32", cfg=TLParityCfg("test")),
    TLParityTest(alias="test_cpu_32_l", cfg=TLParityCfg("test", **w_lit), marks="lightning"),
    TLParityTest(alias="test_cuda_32", cfg=TLParityCfg("test", **cuda), marks="cuda"),
    TLParityTest(alias="test_cuda_32_l", cfg=TLParityCfg("test", **cuda, **w_lit), marks="cuda_l"),
    TLParityTest(alias="train_cpu_32", cfg=TLParityCfg()),
    TLParityTest(alias="train_cpu_32_l", cfg=TLParityCfg(**w_lit), marks="lightning"),
    #TLParityTest(alias="train_cpu_32_debug", cfg=ParityCfg(**debug_hidden)),
    TLParityTest(alias="train_cuda_32", cfg=TLParityCfg(**cuda), marks="cuda"),
    TLParityTest(alias="train_cuda_32_l", cfg=TLParityCfg(**cuda, **w_lit), marks="cuda_l"),
    # ParityTest(alias="test_cuda_bf16", cfg=ParityCfg("test", **cuda_bf16), marks="bf16_cuda"),
    # ParityTest(alias="train_cpu_bf16", cfg=ParityCfg(**bf16), marks="skip_win_slow"),
    # ParityTest(alias="test_cpu_32_prof", cfg=ParityCfg(**test_bs1_mem_nosavedt), marks="prof"),
    # ParityTest(alias="test_cpu_32_prof_l", cfg=ParityCfg(**w_lit, **test_bs1_mem_nosavedt), marks="prof_l"),
    # ParityTest(alias="test_cuda_32_prof", cfg=ParityCfg(**cuda, **test_bs1_mem), marks="cuda_prof"),
    # ParityTest(alias="test_cuda_32_prof_l", cfg=ParityCfg(**cuda, **w_lit, **test_bs1_mem), marks="cuda_prof_l"),
    # ParityTest(alias="test_cuda_bf16_prof", cfg=ParityCfg(**cuda_bf16, **test_bs1_mem), marks="bf16_cuda_prof"),
    # ParityTest(alias="train_cpu_32_prof", cfg=ParityCfg(**bs1_nowarm_hk_mem), marks="prof"),
    # ParityTest(alias="train_cpu_32_prof_act", cfg=ParityCfg(act_ckpt=True, **bs1_nowarm_mem), marks="prof",),
    # ParityTest(alias="train_cuda_32_prof", cfg=ParityCfg(**cuda, **bs1_warm_mem), marks="cuda_prof"),
    # ParityTest(alias="train_cuda_32_prof_act", cfg=ParityCfg(**cuda_act, **bs1_warm_mem), marks="cuda_prof"),
    # ParityTest(alias="train_cuda_bf16_prof", cfg=ParityCfg(**cuda_bf16, **bs1_warm_mem), marks="bf16_cuda_prof"),
)

EXPECTED_TL_PARITY = {cfg.alias: cfg.expected for cfg in TL_PARITY_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(TL_PARITY_CONFIGS, unpack=False))
def test_tl_parity(recwarn, tmp_path, test_alias, test_cfg):
    expected_results = EXPECTED_TL_PARITY[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CONTEXT_WARNS if test_cfg.lightning else TL_CONTEXT_WARNS
    # set `state_log_mode=True` manually below during development to generate/dump state logs for a given test rather
    # than testing the relevant assertions
    datamodule, module = config_modules(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=False)
    if test_cfg.lightning:
        _ = run_lightning(module, datamodule, test_cfg, tmp_path)
    else:
        run_it(module, datamodule, test_cfg)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
