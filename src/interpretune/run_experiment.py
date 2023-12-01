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
from typing import Callable

def bootstrap_cli() -> Callable:
    # TODO: consider adding an env var option to control CLI selection
    if "--lightning_cli" in sys.argv[1:]:
        lightning_cli = True
        sys.argv.remove("--lightning_cli")
    else:
        lightning_cli = False
    if lightning_cli:
        from interpretune.utils.lightning_cli import cli_main
    else:
        from interpretune.utils.cli import cli_main  # type: ignore[no-redef]
    return cli_main()

if __name__ == "__main__":
    bootstrap_cli()
