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
from typing import Optional, Callable, Dict, Any
from copy import copy
from dataclasses import dataclass, field
from functools import partial
from collections.abc import Iterable

import pytest

from interpretune.adapters.registration import Adapter
from tests.configuration import BaseAugTest, BaseCfg, pytest_param_factory, IT_GLOBAL_STATE_LOG_MODE
from tests.orchestration import parity_test
from tests.parity_acceptance.cfg_aliases import cuda, l_gpt2_fts, l_tl_gpt2_fts, TestFTS
from tests.parity_acceptance.expected import fts_parity_results
from tests.results import collect_results
from tests.warns import unexpected_warns, TL_CTX_WARNS, TL_LIGHTNING_CTX_WARNS, FTS_CTX_WARNS
from tests.runif import RunIf


@dataclass(kw_only=True)
class FTSParityCfg(BaseCfg):
    adapter_ctx: Iterable[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    model_src_key: Optional[str] = "gpt2"
    callback_cfgs: Optional[Dict[Any, Dict]] = field(default_factory=lambda: {})
    limit_train_batches: Optional[int] = 2
    limit_val_batches: Optional[int] = 2
    limit_test_batches: Optional[int] = 2
    max_epochs: Optional[int] = 4
    fts_schedule_key: Optional[tuple] = None
    model_cfg: Optional[dict] = field(default_factory=lambda: {})
    max_steps: Optional[int] = -1
    save_checkpoints: bool = True


@dataclass
class FTSParityTest(BaseAugTest):
    result_gen: Optional[Callable] = partial(collect_results, fts_parity_results, normalize=False)


PARITY_FTS_CONFIGS = (
    FTSParityTest(alias="train_cpu_32_l_fts", cfg=FTSParityCfg(**l_gpt2_fts)),
    FTSParityTest(alias="train_cpu_32_l_tl_fts", cfg=FTSParityCfg(**l_tl_gpt2_fts), marks="l_optional"),
    FTSParityTest(alias="train_cuda_32_l_fts", cfg=FTSParityCfg(**l_gpt2_fts, **cuda), marks="cuda_l_optional"),
    FTSParityTest(alias="train_cuda_32_l_tl_fts", cfg=FTSParityCfg(**l_tl_gpt2_fts, **cuda), marks="cuda"),
)

EXPECTED_PARITY_FTS = {cfg.alias: cfg.expected for cfg in PARITY_FTS_CONFIGS}

@pytest.mark.usefixtures("make_deterministic")
@RunIf(lightning=True, finetuning_scheduler=True)
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_param_factory(PARITY_FTS_CONFIGS, unpack=False))
def test_parity_fts(gpt2_ft_schedules, recwarn, tmp_path, test_alias, test_cfg):
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_FTS[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else TL_CTX_WARNS
    expected_warnings = copy(expected_warnings) + FTS_CTX_WARNS
    if test_cfg.fts_schedule_key:
        mod, type = test_cfg.fts_schedule_key
        test_cfg.callback_cfgs[TestFTS]["ft_schedule"] = gpt2_ft_schedules[mod][type]
        test_cfg.callback_cfgs[TestFTS]["expected_exact"] = expected_results.pop('callback_results', {})
        test_cfg.callback_cfgs[TestFTS]["state_log_dir"] = tmp_path if state_log_mode else None
        test_cfg.callback_cfgs[TestFTS]["test_alias"] = test_alias
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
