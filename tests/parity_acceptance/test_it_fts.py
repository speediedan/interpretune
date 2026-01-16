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
from typing import Callable, Dict, Any, Sequence
from copy import copy
from dataclasses import dataclass, field
from functools import partial

import pytest

from interpretune.protocol import Adapter
from tests.base_defaults import BaseAugTest, BaseCfg, pytest_factory
from tests.configuration import IT_GLOBAL_STATE_LOG_MODE
from tests.orchestration import parity_test
from tests.parity_acceptance.cfg_aliases import (
    cuda,
    l_gpt2_fts,
    l_tl_ht_gpt2_fts_multiphase,
    l_tl_bridge_gpt2_fts_multiphase,
    l_tl_bridge_gpt2_tl_names_fts,
    TestFTS,
)
from tests.parity_acceptance.expected import fts_parity_results
from tests.results import collect_results
from tests.warns import unexpected_warns, TL_CTX_WARNS, TL_LIGHTNING_CTX_WARNS, FTS_CTX_WARNS
from tests.runif import RunIf


@dataclass(kw_only=True)
class FTSParityCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    model_src_key: str | None = "gpt2"
    callback_cfgs: dict[Any, Dict] | None = field(default_factory=lambda: {})
    limit_train_batches: int | None = 2
    limit_val_batches: int | None = 2
    limit_test_batches: int | None = 2
    max_epochs: int | None = 4
    fts_schedule_key: tuple | None = None
    model_cfg: dict | None = field(default_factory=lambda: {})
    max_steps: int | None = -1
    save_checkpoints: bool = True


@dataclass
class FTSParityTest(BaseAugTest):
    result_gen: Callable | None = partial(collect_results, fts_parity_results, normalize=False)


PARITY_FTS_CONFIGS = (
    # NOTE: train_cpu_32_l_fts marked as standalone due to disk space constraints on GitHub runners.
    # This test creates large checkpoint files and may cause out-of-space errors when run alongside other tests.
    # Consider switching to a trivial custom model if this becomes problematic.
    FTSParityTest(alias="train_cpu_32_l_fts", cfg=FTSParityCfg(**l_gpt2_fts), marks="standalone"),
    FTSParityTest(alias="train_cuda_32_l_fts", cfg=FTSParityCfg(**l_gpt2_fts, **cuda), marks="cuda_l_optional"),
    FTSParityTest(
        alias="train_cpu_32_l_tl_ht_fts", cfg=FTSParityCfg(**l_tl_ht_gpt2_fts_multiphase), marks="l_optional"
    ),
    FTSParityTest(
        alias="train_cuda_32_l_tl_ht_fts", cfg=FTSParityCfg(**l_tl_ht_gpt2_fts_multiphase, **cuda), marks="cuda_alone"
    ),
    FTSParityTest(
        alias="train_cuda_32_l_tl_bridge_fts",
        cfg=FTSParityCfg(**l_tl_bridge_gpt2_fts_multiphase, **cuda),
        marks="cuda_alone",
    ),
    FTSParityTest(
        alias="train_cuda_32_l_tl_bridge_tl_names_fts",
        cfg=FTSParityCfg(**l_tl_bridge_gpt2_tl_names_fts, **cuda),
        marks="cuda_alone",
    ),
)

EXPECTED_PARITY_FTS = {cfg.alias: cfg.expected for cfg in PARITY_FTS_CONFIGS}


@pytest.mark.usefixtures("make_deterministic")
@RunIf(lightning=True, finetuning_scheduler=True)
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(PARITY_FTS_CONFIGS, unpack=False))
def test_parity_fts(recwarn, tmp_path, test_alias, test_cfg, request):
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_FTS[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else TL_CTX_WARNS
    expected_warnings = copy(expected_warnings) + FTS_CTX_WARNS
    if test_cfg.fts_schedule_key:
        fixture_key, schedule_type = test_cfg.fts_schedule_key
        # Request the fixture dynamically using the fixture_key from schedule config
        fixture_name = f"get_ft_schedule__{fixture_key}__setup"
        schedules = request.getfixturevalue(fixture_name)
        test_cfg.callback_cfgs[TestFTS]["ft_schedule"] = schedules[schedule_type]
        test_cfg.callback_cfgs[TestFTS]["expected_exact"] = expected_results.pop("callback_results", {})
        test_cfg.callback_cfgs[TestFTS]["state_log_dir"] = tmp_path if state_log_mode else None
        test_cfg.callback_cfgs[TestFTS]["test_alias"] = test_alias
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
