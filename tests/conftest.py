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
# initially based on: https://bit.ly/3GDHDcI
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
import random

import pytest
import torch.distributed

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE

from tests import _PATH_DATASETS


@pytest.fixture(scope="function")
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="function")
def make_deterministic(warn_only=True, fill_uninitialized_memory=True):
    # https://pytorch.org/docs/2.2/notes/randomness.html#reproducibility
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch._C._set_deterministic_fill_uninitialized_memory(fill_uninitialized_memory)
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


@pytest.fixture(scope="function", autouse=True)
def preserve_global_rank_variable():
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    if _LIGHTNING_AVAILABLE:
        from lightning.fabric.utilities import rank_zero_only
    else:
        from interpretune.utils.logging import rank_zero_only  # type: ignore[no-redef]

    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


@pytest.fixture(scope="function", autouse=True)
def restore_env_variables():
    """Ensures that environment variables set during the test do not leak out."""
    env_backup = os.environ.copy()
    yield
    leaked_vars = os.environ.keys() - env_backup.keys()
    # restore environment as it was before running the test
    os.environ.clear()
    os.environ.update(env_backup)
    # these are currently known leakers - ideally these would not be allowed
    allowlist = {
        "CUBLAS_WORKSPACE_CONFIG",  # enabled with deterministic flag
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "WANDB_MODE",
        "WANDB_REQUIRE_SERVICE",
        "WANDB_SERVICE",
        "HOROVOD_FUSION_THRESHOLD",  # set by HorovodStrategy # TODO: remove in v2.0.0
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
        "CUDA_MODULE_LOADING",  # leaked since PyTorch 1.13
        "KMP_INIT_AT_FORK",  # leaked since PyTorch 1.13
        "KMP_DUPLICATE_LIB_OK",  # leaked since PyTorch 1.13
        "CRC32C_SW_MODE",  # leaked by tensorboardX
        "TRITON_CACHE_DIR",  # leaked starting in PyTorch 2.0.0
        "OMP_NUM_THREADS",  # leaked by Lightning launchers,
        "TOKENIZERS_PARALLELISM",  # TODO: add a fixture that resets this currently leaked var
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"

@pytest.fixture(scope="function", autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

@pytest.fixture
def tmpdir_server(tmpdir):
    Handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
    from http.server import ThreadingHTTPServer

    with ThreadingHTTPServer(("localhost", 0), Handler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        yield server.server_address
        server.shutdown()


# NOTE [Profiling and Standalone Marks]:
# - CI doesn't run all `profiling` marked tests by default, only the subset of profiling tests that are marked both
#   `profiling` and `profiling_ci`
# - The standalone marks run with CI by default and take precedence over profiling marks
# - To run all profiling tests, set `IT_RUN_PROFILING_TESTS` to `2`

def pytest_collection_modifyitems(items):
    # select special tests
    if os.getenv("IT_RUN_STANDALONE_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(standalone=True)`
            if marker.name == "skipif" and marker.kwargs.get("standalone")
        ]
    elif os.getenv("IT_RUN_PROFILING_TESTS", "0") == "2":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(profiling=True)`
            if marker.name == "skipif" and marker.kwargs.get("profiling")
        ]
    elif os.getenv("IT_RUN_PROFILING_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(profiling_ci=True)`
            if marker.name == "skipif" and marker.kwargs.get("profiling_ci")
        ]
    elif os.getenv("IT_RUN_OPTIONAL_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(optional=True)`
            if marker.name == "skipif" and marker.kwargs.get("optional")
        ]
    elif os.getenv("IT_RUN_SLOW_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(slow=True)`
            if marker.name == "skipif" and marker.kwargs.get("slow")
        ]
