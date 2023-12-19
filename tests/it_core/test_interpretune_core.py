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
from typing import Optional, Dict

import pytest

from interpretune.base.call import it_init, it_session_end
from tests.helpers.core.boring_model import (get_model_input_dtype, cuda_reset, core_train_loop,
                                             bs1_memprof_nowarm_trainonly_memhooks,
                                             TestITModule, unexpected_warns, TestCfg, pytest_param_factory,
                                             datamodule_factory, bs1_memprofiler,
                                             bs1_memprof_sched, nones, bs1_memprof_nowarm,
                                             CORE_CONTEXT_WARNS, get_it_cfg, core_test_loop)


## composable aliases
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

def gen_mem_results(loop_type, src, test_values, memstats_tol: Optional[Dict] = None):
    def_end, bytes = '_step.0.0.end', '_bytes.all.'
    fwd_diff, alloc, reserved = 'rss_diff', f'allocated{bytes}', f'reserved{bytes}'
    test_key, train_keys = f'0.test{def_end}', {"cuda": '0.training_step.0.3.end', "hooks": f'0.training{def_end}'}
    mem_keys = (f'{alloc}current', f'{alloc}peak', f'{reserved}peak') if src == "cuda" else (fwd_diff,)
    step_key = f'{src}.{train_keys[src]}' if loop_type == 'train' else f'{src}.{test_key}'
    if not memstats_tol:
        memstats_tol = {'tolerance_map': {k: (0.05, 0) for k in mem_keys}}
    return {**memstats_tol, 'expected_memstats': (step_key, mem_keys, test_values)}

# note that while we could access test_alias using the request fixture (request.node.callspec.id), this approach
# allows us to flexibly define test ids, configurations, marks and expected outputs together in a single named tuple
PARITY_SINGLE_DEVICE_CONFIGS = (
    TestCfg("cpu_32_train", test_cfg=(*fp32_cpu_def_train, *nones(5))),
    TestCfg("cuda_32_train", test_cfg=(*fp32_cuda_def_train, *nones(5)), marks="cuda"),
    TestCfg("cuda_bf16_train", test_cfg=(*bf16_cuda_def_train, *nones(5)), marks="bf16_cuda"),
    TestCfg("cpu_32_test", test_cfg=(*fp32_cpu_def_test, *nones(5))),
    TestCfg("cuda_32_test", test_cfg=(*fp32_cuda_def_test, *nones(5)), marks="cuda"),
    TestCfg("cuda_bf16_test", test_cfg=(*bf16_cuda_def_test, *nones(5)), marks="bf16_cuda"),
    TestCfg("cpu_bf16_train", test_cfg=("bf16", "cpu", *default_train, *nones(5)), marks="skip_win_slow"),
    TestCfg("cpu_32_test_prof", test_cfg=(*fp32_cpu_def_test, *default_test_prof), marks="prof",
            expected_results=gen_mem_results("test", "hooks", (424734720,))),
    TestCfg("cuda_32_test_prof", test_cfg=(*fp32_cuda_def_test, *default_test_prof), marks="cuda_prof",
            expected_results=gen_mem_results("test", "cuda", (544714240, 666343424, 731906048))),
    TestCfg("cuda_bf16_test_prof", test_cfg=(*bf16_cuda_def_test, *default_test_prof), marks="bf16_cuda_prof",
            expected_results=gen_mem_results("test", "cuda", (301460992, 345495552, 362807296))),
    TestCfg("cpu_32_train_prof", test_cfg=(*fp32_cpu_def_train, *bs1_memprof_nowarm_trainonly_memhooks), marks="prof",
            expected_results=gen_mem_results("train", "hooks", (906223616,))),
    TestCfg("cpu_32_train_prof_act_ckpt", test_cfg=(32, "cpu", *train_full, True, *bs1_memprof_nowarm),
            marks="prof", expected_results=gen_mem_results("train", "hooks", (450809856,))),
    TestCfg("cuda_32_train_prof", test_cfg=(*fp32_cuda_def_train, *bs1_memprof_sched), marks="cuda_prof",
            expected_results=gen_mem_results("train", "cuda", (1939940352, 2579284992, 2862612480))),
    TestCfg("cuda_bf16_train_prof", test_cfg=(*bf16_cuda_def_train, *bs1_memprof_sched), marks="bf16_cuda_prof",
            expected_results=gen_mem_results("train", "cuda", (1132208128, 1363834880, 1491075072))),
)

EXPECTED_RESULTS_PARITY_SINGLE_DEVICE = {cfg.test_alias: cfg.expected_results for cfg in PARITY_SINGLE_DEVICE_CONFIGS}

@pytest.mark.usefixtures("reset_deterministic_algorithm")
@pytest.mark.parametrize(
    ("test_alias", "precision", "device_type", "loop_type", "full_dataset", "act_ckpt", "dm_override_cfg",
     "memprofiling_cfg", "train_steps", "val_steps", "test_steps"),
    pytest_param_factory(PARITY_SINGLE_DEVICE_CONFIGS)
)
def test_parity_single_device(recwarn, tmp_path, test_alias, precision, device_type, loop_type, full_dataset, act_ckpt,
                              dm_override_cfg, memprofiling_cfg, train_steps, val_steps, test_steps):
    expected_results = EXPECTED_RESULTS_PARITY_SINGLE_DEVICE[test_alias] or {}
    cuda_reset()
    train_steps, val_steps, test_steps = train_steps or 1, val_steps or 1, test_steps or 1
    memprofiling_cfg = memprofiling_cfg or {}
    datamodule = datamodule_factory(dm_override_cfg=dm_override_cfg, full_dataset=full_dataset)
    it_cfg = get_it_cfg(config_type=loop_type, device_type=device_type, precision=precision, act_ckpt=act_ckpt,
                        memprofiling_cfg=memprofiling_cfg, core_log_dir=tmp_path)
    module = TestITModule(it_cfg=it_cfg, **expected_results)
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
