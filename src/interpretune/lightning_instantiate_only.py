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
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.utils.cli import compose_config
from interpretune.utils.lightning_cli import cli_main


def run_direct(config_files: List[str]):
    assert _LIGHTNING_AVAILABLE, "Lightning not installed so cannot invoke Lightning CLI."
    cli = cli_main(**compose_config(config_files))
    cli.trainer.test(cli.model, datamodule=cli.datamodule)
    cli.trainer.predict(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    run_direct(sys.argv[1:])
