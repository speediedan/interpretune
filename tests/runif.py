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
# Initially based on https://bit.ly/3J5oOk4
import os
import re
import sys
from typing import Optional

import pytest
import torch
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE, _BNB_AVAILABLE, _FTS_AVAILABLE
from packaging.version import Version
from pkg_resources import get_distribution

EXTENDED_VER_PAT = re.compile(r"([0-9]+\.){2}[0-9]+")

# runif components
cuda_mark = {'min_cuda_gpus': 1}
bf16_cuda_mark = {'bf16_cuda': True}
profiling_mark = {'profiling': True}
profiling_ci_mark = {'profiling_ci': True}
standalone_mark = {'standalone': True}
optional_mark = {'optional': True}
lightning_mark = {"lightning": True}
fts_mark = {'finetuning_scheduler': True}
bitsandbytes_mark = {"bitsandbytes": True}
skip_win_mark = {'skip_windows': True}

# RunIf aliases
RUNIF_ALIASES = {
    "lightning": lightning_mark,
    "bitsandbytes": bitsandbytes_mark,
    "fts": fts_mark,
    "optional": optional_mark,
    "prof": profiling_mark,
    "profiling_ci": profiling_ci_mark,
    "standalone": standalone_mark,
    "l_fts": {**lightning_mark, **fts_mark},
    "cuda": cuda_mark,
    "cuda_profci": {**cuda_mark, **profiling_ci_mark},
    "cuda_l": {**cuda_mark, **lightning_mark},
    "cuda_l_fts": {**cuda_mark, **lightning_mark, **fts_mark},
    "cuda_l_fts_profci": {**cuda_mark, **lightning_mark, **fts_mark, **profiling_ci_mark},
    "cuda_l_profci": {**cuda_mark, **lightning_mark, **profiling_ci_mark},
    "cuda_l_optional": {**cuda_mark, **lightning_mark, **optional_mark},
    "bf16_cuda": bf16_cuda_mark,
    "bf16_cuda_profci": {**bf16_cuda_mark, **profiling_ci_mark},
    "bf16_cuda_l": {**bf16_cuda_mark, **lightning_mark},
    "l_optional": {**lightning_mark, **optional_mark},
    "skip_win_optional": {**skip_win_mark, **optional_mark},
}

class RunIf:
    """RunIf wrapper for simple marking specific cases, basically a `pytest.mark.skipif` decorator factory:

    @RunIf(min_torch="0.0")
    @pytest.mark.parametrize("arg1", [1, 2.0])
    def test_wrapper(arg1):
        assert arg1 > 0.0
    """

    def __new__(
        self,
        *args,
        min_cuda_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        max_python: Optional[str] = None,
        bf16_cuda: bool = False,
        skip_windows: bool = False,
        skip_mac_os: bool = False,
        standalone: bool = False,
        profiling: bool = False,
        profiling_ci: bool = False,
        optional: bool = False,
        lightning: bool = False,
        finetuning_scheduler: bool = False,
        bitsandbytes: bool = False,
        **kwargs,
    ):
        """
        Args:
            *args: Any :class:`pytest.mark.skipif` arguments.
            min_cuda_gpus: Require this number of gpus and that the ``PL_RUN_CUDA_TESTS=1`` environment variable is set.
            min_torch: Require that PyTorch is greater or equal to this version.
            max_torch: Require that PyTorch is less than or equal to this version.
            min_python: Require that Python is greater or equal to this version.
            max_python: Require that Python is less than this version.
            bf16_cuda: Require that CUDA device supports bf16.
            skip_windows: Skip for Windows platform.
            skip_mac_os: Skip Mac OS platform.
            standalone: Mark the test as standalone, our CI will run it in a separate process.
                This requires that the ``IT_RUN_STANDALONE_TESTS=1`` environment variable is set.
            profiling: Mark the test as profiling. It will run as a separate process and only be included in CI
                in limited cases. This requires that the ``IT_RUN_PROFILING_TESTS=2`` environment variable is set.
            profiling_ci: Mark the test as a profiling test intended to run with normal ci. It will run as a separate
                process and usually included in CI. This requires that the ``IT_RUN_PROFILING_TESTS=1`` environment
                variable is set.
            optional: Mark the test as for optional/extended testing. It will run as a separate process and only be
                included in CI in limited cases. This requires that the ``IT_RUN_OPTIONAL_TESTS=1`` environment variable
                is set.
            lightning: Require that lightning is installed.
            finetuning_scheduler: Require that finetuning_scheduler is installed.
            bitsandbytes: Require that bitsandbytes is installed.
            **kwargs: Any :class:`pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_cuda_gpus:
            conditions.append(torch.cuda.device_count() < min_cuda_gpus)
            reasons.append(f"GPUs>={min_cuda_gpus}")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["min_cuda_gpus"] = True

        if min_torch:
            torch_version = get_distribution("torch").version
            extended_torch_ver = EXTENDED_VER_PAT.match(torch_version).group() or torch_version
            conditions.append(Version(extended_torch_ver) < Version(min_torch))
            reasons.append(f"torch>={min_torch}, {extended_torch_ver} installed.")

        if max_torch:
            torch_version = get_distribution("torch").version
            extended_torch_ver = EXTENDED_VER_PAT.match(torch_version).group() or torch_version
            conditions.append(Version(extended_torch_ver) > Version(max_torch))
            reasons.append(f"torch<={max_torch}, {extended_torch_ver} installed.")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if max_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) >= Version(max_python))
            reasons.append(f"python<{max_python}, {py_version} installed.")

        if bf16_cuda:
            try:
                cond = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            except (AssertionError, RuntimeError) as e:
                # AssertionError: Torch not compiled with CUDA enabled
                # RuntimeError: Found no NVIDIA driver on your system.
                is_unrelated = "Found no NVIDIA driver" not in str(e) or "Torch not compiled with CUDA" not in str(e)
                if is_unrelated:
                    raise e
                cond = True

            conditions.append(cond)
            reasons.append("CUDA device bf16")

        if skip_windows:
            conditions.append(sys.platform == "win32")
            reasons.append("unimplemented on Windows")

        if skip_mac_os:
            conditions.append(sys.platform == "darwin")
            reasons.append("unimplemented or temporarily bypassing these tests for MacOS")

        if standalone:
            env_flag = os.getenv("IT_RUN_STANDALONE_TESTS", "0")
            conditions.append(env_flag != "1")
            reasons.append("Standalone execution")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["standalone"] = True

        if profiling:
            env_flag = os.getenv("IT_RUN_PROFILING_TESTS", "0")
            conditions.append(env_flag != "2")
            reasons.append("Profiling All execution")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["profiling"] = True

        if profiling_ci:
            env_flag = os.getenv("IT_RUN_PROFILING_TESTS", "0")
            conditions.append(env_flag not in ["1", "2"])
            reasons.append("Profiling CI execution")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["profiling_ci"] = True

        if optional:
            env_flag = os.getenv("IT_RUN_OPTIONAL_TESTS", "0")
            conditions.append(env_flag != "1")
            reasons.append("Optional/extended test execution")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["optional"] = True

        if lightning:
            conditions.append(not _LIGHTNING_AVAILABLE)
            reasons.append("Lightning")

        if finetuning_scheduler:
            conditions.append(not _FTS_AVAILABLE)
            reasons.append("Finetuning Scheduler")

        if bitsandbytes:
            conditions.append(not _BNB_AVAILABLE)
            reasons.append("BitsandBytes")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args, condition=any(conditions), reason=f"Requires: [{' + '.join(reasons)}]", **kwargs
        )


@RunIf(min_torch="99")
def test_always_skip():
    exit(1)


@pytest.mark.parametrize("arg1", [0.5, 1.0, 2.0])
@RunIf(min_torch="0.0")
def test_wrapper(arg1: float):
    assert arg1 > 0.0
