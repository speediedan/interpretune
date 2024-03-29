# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """TODO, decompose parity functions ported from arena."""
# ### N.B. this is a stupid monolithic copy of a transformer_lens mech_interp tutorial to vet some basic
# ### functional parity while exploring approaches to integrating it, will be adding specific tests after initial explore

# import sys
# from pathlib import Path

# from interpretune.base.cli.core_cli import env_setup

# # ensure test sub-packages are in the path
# tests_dir = Path(__file__).parent.parent.path
# if str(tests_dir) not in sys.path: sys.path.append(str(tests_dir))

# # from ioi_parity import test_ioi_parity


# # if __name__ == "__main__":
# #     env_setup()
# #     test_ioi_parity()
#     #cli_main() # testing without a config for now
