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
# TODO: fill in this placeholder with actual core helper functions
import re
from functools import partial
from typing import List, Optional, Tuple, NamedTuple
from packaging.version import Version
from pkg_resources import get_distribution
from warnings import WarningMessage

import pytest


from tests.helpers.runif import RunIf, EXTENDED_VER_PAT


EXPECTED_WARNS = [
    "The truth value of an empty array is ambiguous",  # for jsonargparse
    "The `use_auth_token` argument is deprecated",  # TODO: need to use `token` instead of `use_auth_token`
]
MIN_VERSION_WARNS = "2.0"
MAX_VERSION_WARNS = "2.2"
# torch version-specific warns go here
EXPECTED_VERSION_WARNS = {MIN_VERSION_WARNS: [],
                          MAX_VERSION_WARNS: [
                              'PairwiseParallel is deprecated and will be removed soon.',  # temp warning for pt 2.2
                              ]}
torch_version = get_distribution("torch").version
extended_torch_ver = EXTENDED_VER_PAT.match(torch_version).group() or torch_version
if Version(extended_torch_ver) < Version(MAX_VERSION_WARNS):
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MIN_VERSION_WARNS])
else:
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MAX_VERSION_WARNS])
ADV_EXPECTED_WARNS = EXPECTED_WARNS + ["Found an `init_pg_lrs` key"]

RUN_FN = "run_experiment.py"



def dummy_step(*args, **kwargs) -> None:
    ...

# RunIf aliases
RUNIF_ALIASES = {
    "lightning": {"lightning": True},
    "bitsandbytes": {"bitsandbytes": True},
}


class TestConfig(NamedTuple):
    test_alias: str
    test_config: Tuple
    marks: Optional[Tuple] = None
    expected_results: Optional[Tuple] = None



def pytest_param_factory(test_configs: List[TestConfig]) -> List:
    return [pytest.param(
            config.test_alias,
            *config.test_config,
            id=config.test_alias,
            marks=RunIf(**RUNIF_ALIASES[config.marks]) if config.marks else tuple(),
        )
        for config in test_configs
    ]


def multiwarn_check(
    rec_warns: List, expected_warns: List, expected_mode: bool = False
) -> List[Optional[WarningMessage]]:
    msg_search = lambda w1, w2: re.compile(w1).search(w2.message.args[0])  # noqa: E731
    if expected_mode:  # we're directed to check that multiple expected warns are obtained
        return [w_msg for w_msg in expected_warns if not any([msg_search(w_msg, w) for w in rec_warns])]
    else:  # by default we're checking that no unexpected warns are obtained
        return [w_msg for w_msg in rec_warns if not any([msg_search(w, w_msg) for w in expected_warns])]


unexpected_warns = partial(multiwarn_check, expected_mode=False)


unmatched_warns = partial(multiwarn_check, expected_mode=True)
