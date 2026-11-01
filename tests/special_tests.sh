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

# Run the special tests (those that need a process of their own) through gpu-epsilon's shared runner, configured by
# [tool.gpu-epsilon.special-tests] in pyproject.toml: one process per node id, each outcome read from per-rank records.
# `python -m gpu_epsilon.special_tests --help` lists the options. Examples, from the repository root:
#   ./tests/special_tests.sh                                   # every standalone test (--mark_type defaults to standalone)
#   ./tests/special_tests.sh --mark_type=profile_ci --log_file=/tmp/parent_session.log
#   ./tests/special_tests.sh --mark_type=standalone --collect_dir='src/it_examples/tests' --filter_pattern='notebook'
#   ./tests/special_tests.sh --mark_type=standalone --no-reruns --allow-failures   # every test, once (still exits 1)
#   ./tests/special_tests.sh --mark_type=standalone --resource-debug   # resource snapshots at every boundary
# --mark_type=exp_patch is refused: no test in this repository selects on an experimental patch.

set -eo pipefail

root="$(dirname "$0")/.."
config="${root}/pyproject.toml"
_gpu_lease_wrap="${root}/scripts/gpu_lease_wrap.sh"
# shellcheck source=../scripts/gpu_lease_wrap.sh
[[ -r "${_gpu_lease_wrap}" ]] && source "${_gpu_lease_wrap}"
type gpu_lease_reexec &>/dev/null || gpu_lease_reexec() { :; }

if ! python -c "import gpu_epsilon.special_tests" 2>/dev/null; then
  >&2 echo "gpu-epsilon is not installed: python -m pip install -r requirements/ci/gpu_epsilon.txt"
  exit 1
fi
# Validate the arguments first, so a bad flag is reported at once rather than after waiting for the GPU lease, then
# serialize the whole run against other GPU work on the host (no-op unless GPU_LEASE_CMD is set). A full suite is
# host-memory-heavy as well as GPU-heavy, so it takes both host leases and stays serialized with every phase of a CI
# job; an explicit GPU_LEASE_ARGS wins.
python -m gpu_epsilon.special_tests --config "${config}" --check-args "$@"
GPU_LEASE_ARGS="${GPU_LEASE_ARGS:---cpu-heavy}"
gpu_lease_reexec "$0" "$@"
# --resource-debug (or an inherited IT_RESOURCE_DEBUG=1) also snapshots host resources at every collection and test
# boundary, through the runner's event hook
for arg in "$@"; do [[ "${arg}" == "--resource-debug" ]] && export IT_RESOURCE_DEBUG=1; done
if [[ "${IT_RESOURCE_DEBUG:-0}" == "1" ]]; then
  export GPU_EPSILON_EVENT_CMD="${GPU_EPSILON_EVENT_CMD:-bash ${root}/scripts/resource_snapshot.sh}"
fi
exec python -m gpu_epsilon.special_tests --config "${config}" "$@"
