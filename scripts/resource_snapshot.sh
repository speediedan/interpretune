#!/bin/bash
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
# GPU_EPSILON_EVENT_CMD for `tests/special_tests.sh --resource-debug`: the runner calls it with a boundary label
# (special-tests:collect:start, special-tests:test:end:<node id>, ...) and logs what it prints.
source "$(dirname "$0")/infra_utils.sh"
log_shell_resource_snapshot "$1"
