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
import logging
import os

import numpy as np

from interpretune.utils import _LIGHTNING_AVAILABLE, _FTS_AVAILABLE, _DOTENV_AVAILABLE

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import seed_everything
    from lightning.pytorch import Trainer, Callback
    from lightning.pytorch.callbacks import ModelCheckpoint
    from interpretune.base import l_cli_main
else:
    seed_everything = object
    l_cli_main = None
    Trainer = object
    Callback = object
    ModelCheckpoint = object


if _FTS_AVAILABLE:
    from finetuning_scheduler import FinetuningScheduler, CallbackResolverMixin

    fts_resolver = CallbackResolverMixin()

    def get_fts(trainer: Trainer) -> Callback:
        fts_resolver.connect_callback(trainer, reconnect=True)
        return fts_resolver.finetuningscheduler_callback

else:
    FinetuningScheduler = object
    get_fts = lambda: None

if _DOTENV_AVAILABLE:
    from dotenv import load_dotenv
else:
    load_dotenv = lambda: None


_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
_TEMP_PATH = os.path.join(_PROJECT_ROOT, "test_temp")
_PATH_DATASETS = os.path.join(_PROJECT_ROOT, "Datasets")

# TODO: this setting `PYTHONPATH` may not be used by other envs like Conda for import packages
if _PROJECT_ROOT not in os.getenv("PYTHONPATH", ""):
    splitter = ":" if os.environ.get("PYTHONPATH", "") else ""
    os.environ["PYTHONPATH"] = f'{_PROJECT_ROOT}{splitter}{os.environ.get("PYTHONPATH", "")}'

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))

if not os.path.isdir(_TEMP_PATH):
    os.mkdir(_TEMP_PATH)

logging.basicConfig(level=logging.ERROR)
