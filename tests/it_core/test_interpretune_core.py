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
# TODO: fill in this placeholder with actual core tests
from typing import Optional, Callable
from dataclasses import dataclass
from functools import partial

import pytest

from interpretune.base.call import it_init, it_session_end
from tests.helpers.core.boring_model import (get_model_input_dtype, cuda_reset, core_train_loop, TestCfg,
                                             bs1_memprof_nowarm_train_memhooks, make_deterministic,
                                             TestITModule, unexpected_warns, pytest_param_factory,
                                             datamodule_factory, bs1_memprofiler, close_results, mem_results,
                                             bs1_mem_sched, nones, bs1_mem_nowarm,
                                             CORE_CONTEXT_WARNS, get_it_cfg, core_test_loop)


# composable test aliases
train_full = ("train", True)
test_full = ("test", True)
default_train = (*train_full, False)  # default train loop: full dataset, no activation checkpointing
default_test = (*test_full, False)  # default train loop: full dataset, no activation checkpointing
bf16_cuda_def_train = ("bf16", "cuda", *default_train)
bf16_cuda_def_test = ("bf16", "cuda", *default_test)
fp32_cpu_def_train = (32, "cpu", *default_train)
fp32_cuda_def_train = (32, "cuda", *default_train)
fp32_cuda_def_test = (32, "cuda", *default_test)
fp32_cpu_def_test = (32, "cpu", *default_test)
default_test_prof = (*bs1_memprofiler, *nones(3))

# expected close results
core_close_results = {
    "cpu_32_train" : (((0, 'loss', 9.54889965057373),), None),
    "cuda_32_train" : (((0, 'loss', 7.308337211608887),), None),
    "cuda_bf16_train" : (((0, 'loss', 5.375034332275391),), None),
}

core_mem_close_results = {
    "cpu_32_test_prof": ("test", "hooks", (391647232,), None),
    "cuda_32_test_prof": ("test", "cuda", (544714240, 666343424, 731906048), None),
    "cuda_bf16_test_prof": ("test", "cuda", (301460992, 345495552, 362807296), None),
    "cpu_32_train_prof": ("train", "hooks", (513175552,), None),
    "cpu_32_train_prof_act": ("train", "hooks", (385560576,), None),
    "cuda_32_train_prof": ("train", "cuda", (1939940352, 2579284992, 2862612480), None),
    "cuda_32_train_prof_act": ("train", "cuda", (1587114496, 2577159168, 2791309312), None),
    "cuda_bf16_train_prof": ("train", "cuda", (1132208128, 1363834880, 1491075072), None),
}

# result factories
core_mem = partial(mem_results, core_mem_close_results)
core_close = partial(close_results, core_close_results)

@dataclass
class CoreMemCfg(TestCfg):
    result_gen: Optional[Callable] = core_mem

# note that while we could access test_alias using the request fixture (request.node.callspec.id), this approach
# allows us to flexibly define test ids, configurations, marks and expected outputs together in a single named tuple
CORE_SINGLE_DEVICE_CONFIGS = (
    TestCfg(alias="cpu_32_train", cfg=(*fp32_cpu_def_train, *nones(5)), result_gen=core_close),
    TestCfg(alias="cuda_32_train", cfg=(*fp32_cuda_def_train, *nones(5)), marks="cuda", result_gen=core_close),
    TestCfg(alias="cuda_bf16_train", cfg=(*bf16_cuda_def_train, *nones(5)), marks="bf16_cuda", result_gen=core_close),
    TestCfg(alias="cpu_32_test", cfg=(*fp32_cpu_def_test, *nones(5))),
    TestCfg(alias="cuda_32_test", cfg=(*fp32_cuda_def_test, *nones(5)), marks="cuda"),
    TestCfg(alias="cuda_bf16_test", cfg=(*bf16_cuda_def_test, *nones(5)), marks="bf16_cuda"),
    TestCfg(alias="cpu_bf16_train", cfg=("bf16", "cpu", *default_train, *nones(5)), marks="skip_win_slow"),
    CoreMemCfg(alias="cpu_32_test_prof", cfg=(*fp32_cpu_def_test, *default_test_prof), marks="prof"),
    CoreMemCfg(alias="cuda_32_test_prof", cfg=(*fp32_cuda_def_test, *default_test_prof), marks="cuda_prof"),
    CoreMemCfg(alias="cuda_bf16_test_prof", cfg=(*bf16_cuda_def_test, *default_test_prof), marks="bf16_cuda_prof"),
    CoreMemCfg(alias="cpu_32_train_prof", cfg=(*fp32_cpu_def_train, *bs1_memprof_nowarm_train_memhooks), marks="prof"),
    CoreMemCfg(alias="cpu_32_train_prof_act", cfg=(32, "cpu", *train_full, True, *bs1_mem_nowarm), marks="prof",),
    CoreMemCfg(alias="cuda_32_train_prof", cfg=(*fp32_cuda_def_train, *bs1_mem_sched), marks="cuda_prof"),
    CoreMemCfg(alias="cuda_32_train_prof_act", cfg=(32, "cuda", *train_full, True, *bs1_mem_sched), marks="cuda_prof"),
    CoreMemCfg(alias="cuda_bf16_train_prof", cfg=(*bf16_cuda_def_train, *bs1_mem_sched), marks="bf16_cuda_prof"),
)

EXPECTED_RESULTS_PARITY_SINGLE_DEVICE = {cfg.alias: cfg.expected for cfg in CORE_SINGLE_DEVICE_CONFIGS}

@pytest.mark.usefixtures("reset_deterministic_algorithm")
@pytest.mark.parametrize(
    ("test_alias", "precision", "device_type", "loop_type", "full_dataset", "act_ckpt", "dm_override_cfg",
     "memprofiling_cfg", "train_steps", "val_steps", "test_steps"),
    pytest_param_factory(CORE_SINGLE_DEVICE_CONFIGS)
)
def test_core_single_device(recwarn, tmp_path, test_alias, precision, device_type, loop_type, full_dataset, act_ckpt,
                            dm_override_cfg, memprofiling_cfg, train_steps, val_steps, test_steps):
    make_deterministic(warn_only=True, fill_uninitialized_memory=True)
    expected_results = EXPECTED_RESULTS_PARITY_SINGLE_DEVICE[test_alias] or {}
    cuda_reset()
    train_steps, val_steps, test_steps = train_steps or 1, val_steps or 1, test_steps or 1
    memprofiling_cfg = memprofiling_cfg or {}
    datamodule = datamodule_factory(dm_override_cfg=dm_override_cfg, full_dataset=full_dataset)
    it_cfg = get_it_cfg(config_type=loop_type, device_type=device_type, precision=precision, act_ckpt=act_ckpt,
                        memprofiling_cfg=memprofiling_cfg, core_log_dir=tmp_path)
    module = TestITModule(it_cfg=it_cfg,
                          # state_log_dir=tmp_path,  # optionally uncomment to enable dump of expected state logs
                          **expected_results,)
    it_init(module=module, datamodule=datamodule)
    if loop_type == "test":
        core_test_loop(module=module, datamodule=datamodule, device_type=device_type, test_steps=test_steps)
    elif loop_type == "train":
        core_train_loop(module=module, datamodule=datamodule, device_type=device_type, train_steps=train_steps,
                        val_steps=val_steps)
    else:
        raise ValueError("Unsupported loop type, loop_type must be 'test' or 'train'")
    it_session_end(module=module, datamodule=datamodule)
    # N.B. ignore forward rss div and cpu mem diff for cuda since it involves substantial cuda *.so files etc.
    assert module.model.score.weight.device.type == device_type
    assert module.model.score.weight.dtype == get_model_input_dtype(precision)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=CORE_CONTEXT_WARNS)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
