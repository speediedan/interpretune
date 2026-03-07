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
"""NNsight adapter parity tests.

Tests NNsight model initialization, inference, and training across core and Lightning
adapter compositions using the GPT-2 model.

NNsight wraps HuggingFace models directly without weight conversion, providing:
- Tracing-based activation access via context managers
- Native HuggingFace ecosystem compatibility
"""

from typing import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

import pytest

from interpretune.protocol import Adapter
from interpretune.config import NNsightConfig
from tests.base_defaults import BaseAugTest, BaseCfg, pytest_factory
from tests.configuration import IT_GLOBAL_STATE_LOG_MODE
from tests.orchestration import parity_test
from tests.parity_acceptance.cfg_aliases import (
    cuda,
    req_det_cuda,
    test_bs1_mem,
    test_bs1_mem_nosavedt,
    bs1_warm_mem,
    w_l_ns,
)
from tests.parity_acceptance.expected import ns_parity_results, profiling_results
from tests.results import collect_results
from tests.warns import unexpected_warns, NS_CTX_WARNS, NS_LIGHTNING_CTX_WARNS


@dataclass(kw_only=True)
class NSParityCfg(BaseCfg):
    """Configuration for NNsight parity tests.

    Uses GPT-2 (openai-community/gpt2) as the default model for testing. NNsight wraps HuggingFace models directly
    without weight conversion.
    """

    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight)
    model_src_key: str | None = "gpt2"  # Use GPT-2 for NNsight tests
    # NNsight configuration - model_name will sync with model_name_or_path
    nnsight_cfg: NNsightConfig | None = field(default_factory=lambda: NNsightConfig(model_name="openai-community/gpt2"))


@dataclass
class NSParityTest(BaseAugTest):
    """Test configuration class for NNsight parity tests."""

    result_gen: Callable | None = partial(collect_results, ns_parity_results)


# Core + NNsight parity tests
# All CPU NNsight parity tests are standalone: cumulative NNsight model loads OOM CI runners
# and prevent subsequent TransformerLens parity tests from completing.
PARITY_NS_CONFIGS = (
    NSParityTest(alias="test_cpu_32", cfg=NSParityCfg(phase="test"), marks="standalone"),
    NSParityTest(
        alias="test_cpu_32_l",
        cfg=NSParityCfg(
            phase="test",
            **w_l_ns,
        ),
        marks="l_standalone",
    ),
    NSParityTest(alias="test_cuda_32", cfg=NSParityCfg(phase="test", **req_det_cuda), marks="cuda"),
    NSParityTest(alias="test_cuda_32_l", cfg=NSParityCfg(phase="test", **req_det_cuda, **w_l_ns), marks="cuda_l"),
    NSParityTest(alias="train_cpu_32", cfg=NSParityCfg(), marks="standalone"),
    NSParityTest(alias="train_cpu_32_l", cfg=NSParityCfg(**w_l_ns), marks="l_standalone"),
    NSParityTest(alias="train_cuda_32", cfg=NSParityCfg(**req_det_cuda), marks="cuda"),
    NSParityTest(alias="train_cuda_32_l", cfg=NSParityCfg(**req_det_cuda, **w_l_ns), marks="cuda_l"),
)

EXPECTED_PARITY_NS = {cfg.alias: cfg.expected for cfg in PARITY_NS_CONFIGS}


@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(PARITY_NS_CONFIGS, unpack=False))
def test_parity_ns(recwarn, tmp_path, request, test_alias, test_cfg):
    """Test NNsight adapter parity across core and Lightning contexts.

    Validates that NNsight models produce consistent results for inference and training operations.
    """
    if test_cfg.req_deterministic:
        request.getfixturevalue("make_deterministic")
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_NS[test_alias] or {}
    expected_warnings = NS_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else NS_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@dataclass
class NSProfilingTest(BaseAugTest):
    """Test configuration class for NNsight profiling tests."""

    result_gen: Callable | None = partial(collect_results, profiling_results)


@dataclass(kw_only=True)
class NSProfileCfg(BaseCfg):
    """Configuration for NNsight profiling tests.

    Uses GPT-2 for consistent profiling measurements.
    """

    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight)
    model_src_key: str | None = "gpt2"
    nnsight_cfg: NNsightConfig | None = field(default_factory=lambda: NNsightConfig(model_name="openai-community/gpt2"))


# NOTE: Lightning variants (_l) are temporarily skipped pending investigation of ~2x memory footprint difference
# compared to core variants. This suggests possible duplicate model references or other memory overhead.
# See TODO in expected.py for more context.
NS_PROFILING_CONFIGS = (
    NSProfilingTest(alias="test_ns_profiling.test_cpu_32", cfg=NSProfileCfg(**test_bs1_mem_nosavedt), marks="optional"),
    NSProfilingTest(
        alias="test_ns_profiling.test_cpu_32_l",
        cfg=NSProfileCfg(**w_l_ns, **test_bs1_mem_nosavedt),
        marks="l_optional",
        function_marks={"skip": "NNsight+Lightning profiling: ~2x memory vs core - needs investigation"},
    ),
    NSProfilingTest(
        alias="test_ns_profiling.test_cuda_32", cfg=NSProfileCfg(**cuda, **test_bs1_mem), marks="cuda_prof"
    ),
    NSProfilingTest(
        alias="test_ns_profiling.test_cuda_32_l",
        cfg=NSProfileCfg(**cuda, **w_l_ns, **test_bs1_mem),
        marks="cuda_l_prof",
        function_marks={"skip": "NNsight+Lightning profiling: ~2x memory vs core - needs investigation"},
    ),
    NSProfilingTest(
        alias="test_ns_profiling.train_cuda_32", cfg=NSProfileCfg(**cuda, **bs1_warm_mem), marks="cuda_profci"
    ),
)

EXPECTED_PARITY_NS_PROFILING = {cfg.alias: cfg.expected for cfg in NS_PROFILING_CONFIGS}


@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(NS_PROFILING_CONFIGS, unpack=False, fq_alias=True))
def test_ns_profiling(recwarn, tmp_path, test_alias, test_cfg):
    """Test NNsight profiling for memory footprint analysis.

    Validates memory usage patterns for NNsight models during inference and training.
    """
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_NS_PROFILING[test_alias] or {}
    expected_warnings = NS_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else NS_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
