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
# Initially based on https://bit.ly/3oQ8Vqf
from pathlib import Path
from typing import Tuple
from collections import defaultdict
from unittest import mock

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.utils.basic_trainer import BasicTrainer, BasicTrainerCfg
from interpretune.base.contract.session import Framework, ITSessionConfig
from configuration import config_modules


if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
else:
    Trainer = object
    ModelCheckpoint = object


########################################################################################################################
# Core Train/Test Orchestration
########################################################################################################################

########################################################################################################################
# NOTE: [Parity Testing Approach]
# - We use a single set of results but separate tests for framework parity tests since we want to minimize framework
#   dependencies for Interpretune and we want to mark at the test-level for greater clarity and flexibility (we want to
#   signal clearly when either diverges from the expected benchmark so aren't testing relative values only)
# - The configuration space for parity tests is sampled rather than exhaustively testing all framework configuration
#   combinations due to resource constraints
# - Note that while we could access test_alias using the request fixture (`request.node.callspec.id`), this approach
#   using dataclass encapsulation allows us to flexibly define test ids, configurations, marks and expected outputs
#   together
# - We always check for basic exact match on device type, precision and dataset state
# - Our result mapping function uses these shared results for all supported parity test suffixes (e.g. '_l')
# - Set `state_log_mode=True` by setting the environmental variable `IT_GLOBAL_STATE_LOG_MODE` to `1` during development
#   to generate/dump state logs for tests rather than testing the relevant assertions (this can be manually overridden
#   on a test-by-test basis as well)
########################################################################################################################

def parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode: bool = False):
    it_session = config_modules(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    if test_cfg.framework_ctx == Framework.lightning:
        _ = run_lightning(it_session, test_cfg, tmp_path)
    else:
        run_it(it_session, test_cfg)

def run_it(it_session: ITSessionConfig, test_cfg: Tuple):
    trainer_config = BasicTrainerCfg(it_session=it_session, max_epochs=test_cfg.max_epochs)
    trainer = BasicTrainer(trainer_cfg=trainer_config)
    if test_cfg.phase == "test":
        trainer.test()
    elif test_cfg.phase == "train":
        trainer.train()
    else:
        raise ValueError("Unsupported phase type, phase must be 'test' or 'train'")



################################################################################
# Lightning Framework Train/Test Orchestration
################################################################################

def run_lightning(it_session: ITSessionConfig, test_cfg: Tuple, tmp_path: Path) -> Trainer:
    accelerator = "cpu" if test_cfg.device_type == "cpu" else "gpu"
    trainer_steps = {"limit_train_batches": test_cfg.limit_train_batches,
                     "limit_val_batches": test_cfg.limit_val_batches, "limit_test_batches": test_cfg.limit_test_batches,
                     "limit_predict_batches": 1, "max_steps": test_cfg.limit_train_batches}
    trainer = Trainer(default_root_dir=tmp_path, devices=1, deterministic=True, accelerator=accelerator,
                      max_epochs=test_cfg.max_epochs, precision=lightning_prec_alias(test_cfg.precision),
                      num_sanity_val_steps=0, **trainer_steps)
    lightning_func = trainer.fit if test_cfg.phase == "train" else trainer.test
    with mock.patch.object(ModelCheckpoint, "_save_checkpoint"):  # do not save checkpoints for lightning tests
        lightning_func(**it_session)
    return trainer

def lightning_prec_alias(precision: str):
    # TODO: update this and get_model_input_dtype() to use a shared set of supported alias mappings once more
    # types are tested
    return "bf16-true" if precision == "bf16" else "32-true"

################################################################################
# Simple Utility Functions
################################################################################

def dummy_step(*args, **kwargs) -> None:
    ...

def nones(num_n) -> Tuple:  # to help dedup config
    return (None,) * num_n

def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)
