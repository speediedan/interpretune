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
import os
import dataclasses
from pathlib import Path
from abc import ABC
from collections import ChainMap
from functools import partial
from typing import List, Optional, Tuple, NamedTuple, Callable, Any, Mapping, Sequence, Union, Dict
from packaging.version import Version
from pkg_resources import get_distribution
from warnings import WarningMessage
from collections import OrderedDict, defaultdict
from copy import deepcopy

import pytest
import torch.distributed
import torch.nn.functional
import torch
import datasets
import evaluate
from torch.testing import assert_close
from torch.utils.data import DataLoader
from torch import Tensor
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.config_classes import ITDataModuleConfig, ITConfig, MemProfilerCfg, MemProfilerSchedule
from interpretune.base.it_datamodule import ITDataModule
from interpretune.base.it_module import ITModule
from interpretune.utils.types import STEP_OUTPUT, Optimizable
from it_examples.experiments.rte_boolq.core import RTEBoolqPromptConfig, RTEBoolqModuleMixin
from it_examples.data.rte_bool import RTEBoolqDataModule
from tests.helpers.runif import RunIf, EXTENDED_VER_PAT


_DEVICE = Union[torch.device, str, int]
#_BLOCKING_DEVICE_TYPES = ("cpu",)

EXPECTED_WARNS = [
    "The truth value of an empty array is ambiguous",  # for jsonargparse
    "The `use_auth_token` argument is deprecated",  # TODO: need to use `token` instead of `use_auth_token`
    "please pass in use_reentrant=True or use_reentrant=False explicitly.",  # hf activation checkpoint warning
    #"dtype is not supported. Disabling autocast",  # enable to allow autocast test paths w/ unsupported types
]
CORE_CONTEXT_WARNS = EXPECTED_WARNS + [
    "For Lightning compatibility, this noop",  # expected in a core context with modules that use Lightning log methods
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

# runif components

cuda_mark = {'min_cuda_gpus': 1}
bf16_cuda_mark = {'bf16_cuda': True}
profiling_mark = {'profiling': True}
lightning_mark = {"lightning": True}
bitsandbytes_mark = {"bitsandbytes": True}
skip_win_mark = {'skip_windows': True}
slow_mark = {'slow': True}
# RunIf aliases

RUNIF_ALIASES = {
    "lightning": lightning_mark,
    "bitsandbytes": bitsandbytes_mark,
    "prof": profiling_mark,
    "cuda": cuda_mark,
    "cuda_prof": {**cuda_mark, **profiling_mark},
    "bf16_cuda": bf16_cuda_mark,
    "bf16_cuda_prof": {**bf16_cuda_mark, **profiling_mark},
    "skip_win_slow": {**skip_win_mark, **slow_mark},
}


class TestCfg(NamedTuple):
    test_alias: str
    test_cfg: Tuple
    marks: Optional[Tuple] = None
    expected_results: Optional[Dict] = None


def get_marks(marks: Union[Dict, str]) -> RunIf:
    # support RunIf aliases
    if isinstance(marks, Dict):
        return RunIf(**marks)
    elif isinstance(marks, str):
        return RunIf(**RUNIF_ALIASES[marks])
    else:
        raise ValueError(f"Unexpected marks type (should be Dict or str): {type(marks)}")

def pytest_param_factory(test_configs: List[TestCfg]) -> List:
    return [pytest.param(
            config.test_alias,
            *config.test_cfg,
            id=config.test_alias,
            marks=get_marks(config.marks) if config.marks else tuple(),
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


# useful Lightning_utilities helper functions
def is_namedtuple(obj: object) -> bool:
    """Check if object is type nametuple."""
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def is_dataclass_instance(obj: object) -> bool:
    """Check if object is dataclass."""
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
    include_none: bool = True,
    allow_frozen: bool = False,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        allow_frozen: Whether not to error upon encountering a frozen dataclass instance.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection
    """
    if include_none is False or wrong_dtype is not None or allow_frozen is True:
        # not worth implementing these on the fast path: go with the slower option
        return _apply_to_collection_slow(
            data,
            dtype,
            function,
            *args,
            wrong_dtype=wrong_dtype,
            include_none=include_none,
            allow_frozen=allow_frozen,
            **kwargs,
        )
    # fast path for the most common cases:
    if isinstance(data, dtype):  # single element
        return function(data, *args, **kwargs)
    if isinstance(data, list) and all(isinstance(x, dtype) for x in data):  # 1d homogeneous list
        return [function(x, *args, **kwargs) for x in data]
    if isinstance(data, tuple) and all(isinstance(x, dtype) for x in data):  # 1d homogeneous tuple
        return tuple(function(x, *args, **kwargs) for x in data)
    if isinstance(data, dict) and all(isinstance(x, dtype) for x in data.values()):  # 1d homogeneous dict
        return {k: function(v, *args, **kwargs) for k, v in data.items()}
    # slow path for everything else
    return _apply_to_collection_slow(
        data,
        dtype,
        function,
        *args,
        wrong_dtype=wrong_dtype,
        include_none=include_none,
        allow_frozen=allow_frozen,
        **kwargs,
    )


def _apply_to_collection_slow(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
    include_none: bool = True,
    allow_frozen: bool = False,
    **kwargs: Any,
) -> Any:
    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = _apply_to_collection_slow(
                v,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple_ = is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = _apply_to_collection_slow(
                d,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            v = None
            if field_init:
                v = _apply_to_collection_slow(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    allow_frozen=allow_frozen,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                if allow_frozen:
                    # Quit early if we encounter a frozen data class; return `result` as is.
                    break
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collection` but this is not allowed."
                ) from e
        return result

    # data is neither of dtype, nor a collection
    return data


class _TransferableDataType(ABC):
    """A custom type for data that can be moved to a torch device via ``.to(...)``."""

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is _TransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented


def move_data_to_device(batch: Any, device: _DEVICE) -> Any:
    """Transfers a collection of data to the given device. Any object that defines a method ``to(device)`` will be
    moved and all other objects in the collection will be left untouched.

    Args:
        batch: A tensor or collection of tensors or anything that has a method ``.to(...)``.
            See :func:`apply_to_collection` for a list of supported collection types.
        device: The device to which the data should be moved

    Return:
        the same collection but with all contained tensors residing on the new device.

    See Also:
        - :meth:`torch.Tensor.to`
        - :class:`torch.device`
    """
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:
        kwargs = {}
        # Don't issue non-blocking transfers to CPU
        if isinstance(data, Tensor) and isinstance(device, torch.device) and device.type not in "cpu":
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        # user wrongly implemented the `_TransferableDataType` and forgot to return `self`.
        return data

    return apply_to_collection(batch, dtype=_TransferableDataType, function=batch_to)


def _clear_cuda_memory() -> None:
    # strangely, the attribute function be undefined when torch.compile is used
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        # https://github.com/pytorch/pytorch/issues/95668
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()

def is_state_dict_equal(state0, state1):
    return all(torch.equal(w0.cpu(), w1.cpu()) for w0, w1 in zip(state0.values(), state1.values()))


def is_timing_close(timings_torch, timings_fabric, rtol=1e-2, atol=0.1):
    # Drop measurements of the first iterations, as they may be slower than others
    # The median is more robust to outliers than the mean
    # Given relative and absolute tolerances, we want to satisfy: |torch – fabric| < RTOL * torch + ATOL
    return bool(torch.isclose(torch.median(timings_torch[3:]), torch.median(timings_fabric[3:]), rtol=rtol, atol=atol))


def is_cuda_memory_close(memory_stats_torch, memory_stats_fabric):
    # We require Fabric's peak memory usage to be smaller or equal to that of PyTorch
    return memory_stats_torch["allocated_bytes.all.peak"] >= memory_stats_fabric["allocated_bytes.all.peak"]

def is_cpu_memory_close(memory_stats_torch, memory_stats_fabric):
    return memory_stats_torch["allocated_bytes.all.peak"] >= memory_stats_fabric["allocated_bytes.all.peak"]

def make_deterministic(warn_only=False):
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

def get_model_input_dtype(precision):
    if precision in ("float16", "16-true", "16-mixed", "16", 16):
        return torch.float16
    if precision in ("bfloat16","bf16-true", "bf16-mixed", "bf16"):
        return torch.bfloat16
    if precision in ("64-true", "64", 64):
        return torch.double
    return torch.float32

def cuda_reset():
    if torch.cuda.is_available():
        _clear_cuda_memory()
        torch.cuda.reset_peak_memory_stats()


def to_device(device: _DEVICE, obj: Union[torch.nn.Module, Tensor, Any]) -> Union[torch.nn.Module, Tensor, Any]:
    r"""Move a :class:`torch.nn.Module` or a collection of tensors to the current device, if it is not already on
    that device.

    Args:
        obj: An object to move to the device. Can be an instance of :class:`torch.nn.Module`, a tensor, or a
            (nested) collection of tensors (e.g., a dictionary).

    Returns:
        A reference to the object that was moved to the new device.

    """
    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        return obj
    return move_data_to_device(obj, device=device)


def nones(num_n) -> Tuple:  # to help dedup config
    return (None,) * num_n

# TODO: change these dict-based configs to decomposed yaml strings, files? Might init more dataclasses

test_tokenizer_kwargs = {"tokenizer_kwargs": {"add_bos_token": True, "local_files_only": False, "padding_side": "right",
                         "model_input_names": ['input_ids', 'attention_mask']}}

test_shared_config = {
    "task_name": "pytest_rte",
    "model_name_or_path": "gpt2",
    "tokenizer_id_overrides": {"pad_token_id": 50256},
    "tokenizer_kwargs": test_tokenizer_kwargs,
}

test_signature_columns= ['input_ids', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'labels',
                         'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']

test_datamodule_kwargs = {"prompt_cfg": RTEBoolqPromptConfig, "signature_columns": test_signature_columns,
                          "enable_datasets_cache": True, "prepare_data_map_cfg": {"batched": True},
                          "text_fields": ("premise", "hypothesis"),  "train_batch_size": 2, "eval_batch_size": 2}

test_optimizer_init = {"optimizer_init": {"class_path": "torch.optim.AdamW",
                              "init_args": {"weight_decay": 1.0e-06, "eps": 1.0e-07, "lr": 3.0e-05}}}

test_lr_scheduler_init = {"lr_scheduler_init": {"class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                              "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-06}}}

test_optimizer_scheduler_init = ChainMap(test_optimizer_init, test_lr_scheduler_init)

test_it_module_kwargs = {"use_model_cache": False, "from_pretrained_cfg":
                         {"device_map": "cpu", "torch_dtype": "float32"}, "experiment_tag": "test_itmodule",
                         "auto_model_cfg": {"model_head": "transformers.GPT2ForSequenceClassification"},}

base_memprofiler_kwargs = {"enabled": True, "cuda_allocator_history": True}

base_memprofiler_cfg = {'memprofiler_cfg': MemProfilerCfg(**base_memprofiler_kwargs)}

retain_trainonly_memhooks = {"retain_hooks_for_funcs": ["training_step"]}

bs1_override = {'train_batch_size': 1, 'eval_batch_size': 1}

bs1_memprofiler = (bs1_override, base_memprofiler_cfg)

warm_max_memprofiler = ChainMap(base_memprofiler_kwargs, {"schedule": MemProfilerSchedule(warmup_steps=2, max_step=4)})
nowarm_max_memprofiler = ChainMap(base_memprofiler_kwargs, {"schedule": MemProfilerSchedule(max_step=4)})

bs1_memprof_sched = (bs1_override, {'memprofiler_cfg': MemProfilerCfg(**warm_max_memprofiler)}, 5, 3, None)
bs1_memprof_nowarm = (bs1_override, {'memprofiler_cfg': MemProfilerCfg(**nowarm_max_memprofiler)}, 5, 3, None)
bs1_memprof_nowarm_trainonly_memhooks = (bs1_override, {
    'memprofiler_cfg': MemProfilerCfg(**nowarm_max_memprofiler, **retain_trainonly_memhooks)}, 5, 3, None)

test_it_module_base = ChainMap(test_shared_config, test_it_module_kwargs)
test_it_module_optim = ChainMap(test_it_module_base, test_optimizer_scheduler_init)


PYTEST_RTE = {
    "premise": ["Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to "
                "the Christopher Reeve Foundation.", "Yet, we now are discovering that antibiotics are losing their "
                "effectiveness against illness. Disease-causing bacteria are mutating faster than we can come up with "
                "new antibiotics to fight the new variations."],
    "hypothesis": ["Christopher Reeve had an accident.", "Bacteria is winning the war against antibiotics."],
    "idx": [0, 1],
    "label": [1, 0]
}

class TestITDataModule(ITDataModule):
    def __init__(self, itdm_cfg: ITDataModuleConfig) -> None:
        super().__init__(itdm_cfg=itdm_cfg)
        self.tokenization_func = self._tokenize_for_gpt2

    def _tokenize_for_gpt2(self, example_batch: LazyDict) -> BatchEncoding:
        example_batch['sequences'] = []
        assert example_batch is not None
        assert self.itdm_cfg.text_fields is not None
        assert self.itdm_cfg.prompt_cfg is not None
        # TODO: use promptsource instead of this manual approach after tinkering
        for field1, field2 in zip(example_batch[self.itdm_cfg.text_fields[0]],
                                  example_batch[self.itdm_cfg.text_fields[1]]):
            if self.itdm_cfg.prompt_cfg.cust_task_prompt:
                task_prompt = (self.itdm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" + field1 + "\n\n" +
                               self.itdm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" + field2)
            else:
                task_prompt = (field1 + self.itdm_cfg.prompt_cfg.ctx_question_join + field2 \
                               + self.itdm_cfg.prompt_cfg.question_suffix)
            sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]
        return features

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        dataset_path = Path(self.itdm_cfg.dataset_path)
        # rebuild the test dataset if it does not exist in the test environment
        if not dataset_path.exists():
            dataset = datasets.Dataset.from_dict(PYTEST_RTE)
            dataset = dataset.map(self.tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
            dataset = self._remove_unused_columns(dataset)
            dataset.save_to_disk(self.itdm_cfg.dataset_path)

    def dataloader_factory(self, use_train_batch_size: bool = False) -> DataLoader:
        dataloader_kwargs = {"dataset": self.dataset, "collate_fn":self.data_collator,
                             **self.itdm_cfg.dataloader_kwargs}
        dataloader_kwargs['batch_size'] = self.itdm_cfg.train_batch_size if use_train_batch_size else \
            self.itdm_cfg.eval_batch_size
        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader_factory(use_train_batch_size=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader_factory()

    def test_dataloader(self) -> DataLoader:
        return self.dataloader_factory()

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader_factory()


class TestITDataModuleFullDataset(RTEBoolqDataModule, TestITDataModule):
    def __init__(self, itdm_cfg: ITDataModuleConfig) -> None:
        itdm_cfg.task_name = 'rte'
        itdm_cfg.dataset_path
        TestITDataModule.__init__(self, itdm_cfg=itdm_cfg)

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        self.itdm_cfg.dataset_path = Path(self.itdm_cfg.dataset_path).parent / 'rte'
        RTEBoolqDataModule.prepare_data(self, target_model=target_model)


class TestITModule(RTEBoolqModuleMixin, ITModule):

    def __init__(self, it_cfg: ITConfig, expected_state: Optional[Dict] = None,
                 expected_memstats: Optional[Dict] = None, tolerance_map: Optional[Dict] = None,
                   *args, **kwargs) -> None:
        super().__init__(it_cfg=it_cfg)
        self.expected_state = expected_state
        self.expected_memstats = expected_memstats
        self.tolerance_map = tolerance_map or {}

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._init_entailment_mapping()

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        it_cfg.num_labels = 0 if it_cfg.zero_shot_cfg.enabled else 2
        return it_cfg

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", 'rte', experiment_id=self.init_hparams['experiment_id'])

    def _init_entailment_mapping(self) -> None:
        if self.it_cfg.zero_shot_cfg.enabled:
            tokenizer, zs_cfg = self.datamodule.tokenizer, self.it_cfg.zero_shot_cfg
            zs_cfg.entailment_mapping_indices = torch.tensor(tokenizer.convert_tokens_to_ids(zs_cfg.entailment_mapping))

    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        super().test_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_train_end(self) -> Any | None:
        if self.it_cfg.memprofiler_cfg and self.expected_memstats:
            self._validate_memory_stats()

    def _validate_memory_stats(self) -> None:
        for act, exp in zip(self.expected_memstats[1], self.expected_memstats[2]):
            rtol, atol = self.tolerance_map.get(act, (0, 0))
            assert_close(actual=self.memprofiler.memory_stats[self.expected_memstats[0]][act], expected=exp,
                         rtol=rtol, atol=atol)

def get_itdm_cfg(dm_override_cfg: Optional[Dict] = None, **kwargs) -> ITConfig:
    test_it_datamodule_cfg = deepcopy(test_datamodule_kwargs)
    if dm_override_cfg:
        test_it_datamodule_cfg.update(dm_override_cfg)
    return ITDataModuleConfig(**test_shared_config, **test_it_datamodule_cfg)

def get_it_cfg(device_type: str, precision: Union[int, str], config_type: str = "test", act_ckpt: bool = False,
               memprofiling_cfg: Optional[Dict] = None, core_log_dir: Optional[str| os.PathLike] = None) -> ITConfig:
    if config_type == "test":
        test_it_module_cfg = deepcopy(test_it_module_base)
    elif config_type == "train":
        test_it_module_cfg = deepcopy(test_it_module_optim)
    if act_ckpt:
        test_it_module_cfg.update({"activation_checkpointing": True})
    if memprofiling_cfg:
        test_it_module_cfg.update(memprofiling_cfg)
    if core_log_dir:
        test_it_module_cfg.update({'core_log_dir': core_log_dir})
    test_it_module_cfg = configure_device_precision(test_it_module_cfg, device_type, precision)
    return ITConfig(**test_it_module_cfg)

def configure_device_precision(cfg: Dict, device_type: str, precision: Union[int, str]) -> Dict[str, Any]:
    cfg['from_pretrained_cfg'].update({'torch_dtype': get_model_input_dtype(precision)})
    if device_type == "cuda":
        cfg['from_pretrained_cfg'].update({'device_map': 0})
    return cfg

# TODO: add/test usage of this fixture that inits a basic module/datamodule pair to used for all tests at a function
# (or maybe module) level that use this fixure
#@pytest.fixture(scope="function")
# def test_module_datamodule_base_cuda() -> Tuple[TestITModule,TestITDataModule]:
#     """A fixture that generates a 'best' and 'kth' checkpoint to be used in scheduled fine-tuning resumption
#     testing."""
#     it_cfg = TestITConfigBase
#     torch_dtype = get_model_input_dtype(32)
#     # TODO: update to only override torch_dtype in from_pretrained_cfg only once torch_dtype is a module property
#     torch_dtype_attrs = ("from_pretrained_cfg['torch_dtype']", "torch_dtype")
#     for attr in torch_dtype_attrs:
#         setattr(it_cfg, attr, torch_dtype)
#     it_cfg.from_pretrained_cfg.update({'device_map': 0})
#     cuda_reset()
#     datamodule = TestITDataModule(itdm_cfg=TestITDataModuleConfig)
#     module = TestITModule(it_cfg=it_cfg)
#     it_init(module=module, datamodule=datamodule)
#     return module, datamodule

# while this fixture works to avoid reinitilizing the fulldataset datamodule for each test, most of the resource
# consumption is in the setup/init which is not shared so we use a factory function to get the datamodule for
# each configuration instead of a fixture to bolster flexibility at the expense of some additional initialization cost
# @pytest.fixture(scope="module")
# def fulldataset_datamodule() -> TestITDataModule:
#     return TestITDataModuleFullDataset(itdm_cfg=get_itdm_cfg())

def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)

def datamodule_factory(dm_override_cfg: Optional[Dict] = None, full_dataset: bool = False):
    itdm_cfg = get_itdm_cfg(dm_override_cfg=dm_override_cfg)
    datamodule_class = TestITDataModuleFullDataset if full_dataset else TestITDataModule
    return datamodule_class(itdm_cfg=itdm_cfg)

def fetch_batch(iterator, module) -> BatchEncoding:
    batch = next(iterator)
    to_device(module.model.device, batch)
    return batch

# TODO: add a test using this loop validating memprofile_ctx as a context manager like this (vs decorator invocation)
# def run_profile_ctx_train_step(module, optimizer, iterator, batch_idx, device_type, profiler, epoch_idx = 0):
#     with ProfilerHooksMixin.memprofile_ctx(module.memprofiler, epoch_idx=epoch_idx, step_idx=batch_idx,
#                                             phase="train"):
#         batch = fetch_batch(iterator, module)
#         optimizer.zero_grad()
#         if module.torch_dtype == torch.bfloat16:
#             with torch.autocast(device_type=device_type, dtype=module.torch_dtype):
#                 loss = module.training_step(batch, batch_idx)
#         else:
#             loss = module.training_step(batch, batch_idx)
#         loss.backward()
#         optimizer.step()

def run_step(step_fn, module, iterator, batch_idx, device_type, optimizer: Optional[Optimizable] = None):
    torch.set_printoptions(precision=12)
    batch = fetch_batch(iterator, module)
    step_func = getattr(module, step_fn)
    if step_fn == "training_step":
        optimizer.zero_grad()
    if module.torch_dtype == torch.bfloat16:
        with torch.autocast(device_type=device_type, dtype=module.torch_dtype):
            loss = step_func(batch, batch_idx)
    else:
        loss = step_func(batch, batch_idx)
    if step_fn == "training_step":
        loss.backward()
        optimizer.step()


def core_train_loop(
    module: ITModule,
    datamodule: ITDataModule,
    device_type: str,
    train_steps: int = 1,
    epochs: int = 1,
    val_steps: int = 1,
):
    make_deterministic(warn_only=True)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader() if val_steps > 0 else None
    optim = module.it_optimizers[0]
    train_ctx = {"module": module, "optimizer": optim, "device_type": device_type}
    for epoch_idx in range(epochs):
        module.model.train()
        module._current_epoch = epoch_idx
        iterator = iter(train_dataloader)
        for batch_idx in range(train_steps):
            run_step(step_fn="training_step", iterator=iterator, batch_idx=batch_idx, **train_ctx)
        if val_steps > 0:
            module.model.eval()
            iterator = iter(val_dataloader)
            for batch_idx in range(val_steps):
                with torch.inference_mode():
                    run_step(step_fn="validation_step", iterator=iterator, batch_idx=batch_idx, **train_ctx)
        module.model.train()

def core_test_loop(
    module: ITModule,
    datamodule: ITDataModule,
    device_type: str,
    test_steps: int = 1
):
    make_deterministic(warn_only=True)
    dataloader = datamodule.test_dataloader()
    test_ctx = {"module": module, "device_type": device_type}
    module._current_epoch = 0
    module.model.eval()
    iterator = iter(dataloader)
    for batch_idx in range(test_steps):
        with torch.inference_mode():
            run_step(step_fn="test_step", iterator=iterator, batch_idx=batch_idx, **test_ctx)
