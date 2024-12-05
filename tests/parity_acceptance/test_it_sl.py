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
from typing import Optional, Callable, Sequence
from dataclasses import dataclass
from functools import partial

import pytest

from interpretune.adapters.registration import Adapter

from tests.base_defaults import BaseAugTest, BaseCfg, pytest_factory
from tests.configuration import IT_GLOBAL_STATE_LOG_MODE
from tests.orchestration import parity_test
from tests.parity_acceptance.cfg_aliases import (cuda, w_l_sl, cust_no_sae_grad)
from tests.parity_acceptance.expected import sl_parity_results
from tests.results import collect_results
from tests.warns import unexpected_warns, SL_CTX_WARNS, SL_LIGHTNING_CTX_WARNS

# TODO: add tl and tl_profiling bf16 tests if/when support vetted
# TODO: add tl activation checkpointing tests if/when support vetted

@dataclass(kw_only=True)
class SLParityCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens)
    model_src_key: Optional[str] = "cust"
    add_saes_on_init: bool = True

@dataclass
class SLParityTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, sl_parity_results)


PARITY_SL_CONFIGS = (
    SLParityTest(alias="test_cpu_32", cfg=SLParityCfg(phase="test")),
    SLParityTest(alias="test_cpu_32_l", cfg=SLParityCfg(phase="test", model_src_key="gpt2", **w_l_sl),
                 marks="lightning"),
    SLParityTest(alias="train_cpu_32_l", cfg=SLParityCfg(**cust_no_sae_grad, **w_l_sl), marks="lightning"),
    SLParityTest(alias="train_cuda_32", cfg=SLParityCfg(**cuda), marks="cuda"),
)

EXPECTED_PARITY_SL = {cfg.alias: cfg.expected for cfg in PARITY_SL_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(PARITY_SL_CONFIGS, unpack=False))
def test_parity_sl(recwarn, tmp_path, test_alias, test_cfg):
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_SL[test_alias] or {}
    expected_warnings = SL_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else SL_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
