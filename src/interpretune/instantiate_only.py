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
"""TODO, add transformerlens basic llama2 example to cli-driven config in ipynb for rapid experimentation."""

import sys
from typing import List
import logging

from interpretune.utils.cli import cli_main, compose_config
from interpretune.utils.call import _call_itmodule_hook
from interpretune.utils.logging import rank_zero_info


log = logging.getLogger(__name__)

def run_direct(config_files: List[str]):
    cli = cli_main(**compose_config(config_files))
    # after getting cli returning, should add setup steps etc to ensure env is prepped for iterative experimentation
    # hook
    rank_zero_info(f"{cli.__class__.__name__}: preparing data")
    _call_itmodule_hook(cli.datamodule, "prepare_data", target_model=cli.model.model)
    _call_itmodule_hook(cli.datamodule, "setup")
    _call_itmodule_hook(cli.model, "setup", datamodule=cli.datamodule)
    pass
    # TODO: add in a lightweight optimizer wrapper configuration parser
    # (based on lightning/pytorch/core/optimizer.py)
    #  to parse configure_optimizers output (while still requiring user to execute the loop)
    # optim_conf = call._call_lightning_module_hook(model.trainer, "configure_optimizers", pl_module=model)
    #cli.trainer.test(cli.model, datamodule=cli.datamodule)
    #cli.trainer.predict(cli.model, datamodule=cli.datamodule)
    #cli.trainer.predict(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    run_direct(sys.argv[1:])
