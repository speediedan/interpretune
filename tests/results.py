from typing import List, Optional, Tuple,  Union, Dict, NamedTuple
from enum import auto, Enum
from interpretune.base.config.shared import AutoStrEnum
from collections import defaultdict

from tests.base_defaults import default_test_bs, default_prof_bs
from tests.utils import get_model_input_dtype

################################################################################
# Test Dataset Fingerprint Definitions
################################################################################

########################################################################################################################
# NOTE [Test Dataset Fingerprint]
# A simple fingerprint of the (deterministic) test dataset used to generate the current incarnation of expected results.
# Useful for validating that the test dataset has not changed wrt the test dataset used to generate the reference
# results. A few things to note:
#   - The dataloader kwargs are not currently part of these fingerprint so if the loss of a given test diverges
#      from expectation, one may still need to verify shuffling of the fingerprinted dataset etc. has not been
#      introduced and compare the examples actually passed to the model in a given test/step to the ids below before
#      subsequently assessing other sources of indeterminism that could be the source of the loss change.
#   - One should see `tests.tools.core.modules.TestITDataModule.sample_dataset_state()` for the indices used to generate
#      this fingerprint
#   - The fingerprinted dataset below is not shuffled or sorted with the current dataloader configurations
#   - All current expected loss results were generated with [train|eval]_batch_size = 2
#   - All current memory profile results were generated with [train|eval]_batch_size = 1
NUM_SAMPLE_ROWS = 5
SAMPLE_POSITION = 3
test_datasets = ("rte", "pytest_rte_hf", "pytest_rte_pt", "pytest_rte_tl")
rte_fields = ("premise", "hypothesis")
TEST_TASK_NUM_LABELS = {k: 2 for k in test_datasets}
TEST_TASK_TEXT_FIELD_MAP = {k: rte_fields for k in test_datasets}
# note that we also sample the 'test' split after 'train' and 'validation' though we aren't yet using it
deterministic_token_ids = [5674, 24140, 373, 666, 2233, 303, 783, 783, 2055, 319, 373, 910, 17074, 284, 6108]
EXPECTED_FIRST_FWD_IDS = {"no_sample": ([],),
                          "train": (deterministic_token_ids[:default_test_bs],),
                          "train_prof": (deterministic_token_ids[:default_prof_bs],),
                          "test": (deterministic_token_ids[NUM_SAMPLE_ROWS:(NUM_SAMPLE_ROWS+default_test_bs)],),
                          "test_prof": (deterministic_token_ids[NUM_SAMPLE_ROWS:(NUM_SAMPLE_ROWS+default_prof_bs)],)}


class TestDatasetKey(AutoStrEnum):
    # datamodule task names that are cached separately by HF `datasets`
    pytest_rte_hf = auto()  # dataset inputs tokenized with standard HF 'input_ids'
    pytest_rte_pt = auto()  # dataset inputs tokenized with custom 'tokens'
    pytest_rte_tl = auto()  # dataset inputs tokenized with standard TL 'input'
    #pytest_rte_sl = auto()  # TODO: decide if we should/need to customize sl dataset, probably will in the future
    ANY = auto()  # used for testing that may use multiple dataset cache keys

gpt2_dataset_state = ('GPT2TokenizerFast', deterministic_token_ids)
llama_dataset_state = ('LlamaTokenizerFast', [])
gemma2_dataset_state = ('GemmaTokenizerFast', [])
test_dataset_state_core_gpt2 = (TestDatasetKey.pytest_rte_hf,) + gpt2_dataset_state
test_dataset_state_core_llama = (TestDatasetKey.pytest_rte_hf,) +  llama_dataset_state
test_dataset_state_core_cust = (TestDatasetKey.pytest_rte_pt,) + gpt2_dataset_state
test_dataset_state_tl = (TestDatasetKey.pytest_rte_tl,) + gpt2_dataset_state
test_dataset_state_sl = (TestDatasetKey.pytest_rte_tl,) + gpt2_dataset_state
test_dataset_state_gpt2_dstype_agnostic = (TestDatasetKey.ANY,) + gpt2_dataset_state


# TODO: add current dataloader kwargs to the fingerprint above? May be an excessively rigid check. Consider associating
# a fingerprint of salient config with each specific expected scalar test result in the future. At present, that
# approach seems like overkill given the current codebase.
########################################################################################################################
class MemProfResult(NamedTuple):
    # encapsulate memprofiler result defaults
    # we default to step:
    #   - 3 for cuda in the train phase
    #   - 0 for all other tests (e.g. hook-based (by default, cpu/rss-based) and all test phase assessment)
    # all tests currently default to test epoch 0 to minimize TTS
    epoch = 0
    rank = 0
    default_step = 0
    cuda_train_step = 3
    cuda_mem_keys = ('allocated_bytes.all.current', 'allocated_bytes.all.peak', 'reserved_bytes.all.peak','npp_diff')
    cpu_mem_keys = {"test": ('rss_diff',), "train": ('rss_diff', 'npp_diff'),}
    test_key = f'{rank}.test_step.{epoch}.{default_step}.end'
    train_keys = {"cuda": f'{rank}.training_step.{epoch}.{cuda_train_step}.end',
                  "cpu": f'{rank}.training_step.{epoch}.{default_step}.end'}

################################################################################
# Expected result generation and encapsulation
################################################################################

class TestResult(NamedTuple):
    result_alias: Optional[str] = None  # N.B. diff test aliases may map to the same result alias (e.g. parity tests)
    exact_results: Optional[Dict] = None
    close_results: Optional[Tuple] = None
    mem_results: Optional[Tuple] = None
    tolerance_map: Optional[Dict[str, float]] = None
    dstype_agnostic: bool = False
    callback_results: Optional[Dict] = None

def mem_results(results: Tuple):
    """Result generation function for memory profiling tests."""
    # See NOTE [Memprofiler Key Format]
    # snap keys are rank.phase.epoch_idx.step_idx.step_ctx
    phase, src, test_values = results
    mem_keys = MemProfResult.cuda_mem_keys if src == "cuda" else MemProfResult.cpu_mem_keys[phase]
    step_key = f'{MemProfResult.train_keys[src]}' if phase == 'train' else f'{MemProfResult.test_key}'
    # default tolerance of rtol=0.05, atol=0 for all keys unless overridden with an explicit `tolerance_map`
    tolerance_map = {'tolerance_map': {k: (0.05, 0) for k in mem_keys}}
    return {**tolerance_map, 'expected_memstats': (step_key, mem_keys, test_values)}

def close_results(close_map: Tuple):
    """Result generation function that packages expected close results with a provided tolerance dict or generates
    a default one based upon the test_alias."""
    expected_close = defaultdict(dict)
    close_keys = set()
    for e, k, v in close_map:
        expected_close[e][k] = v
        close_keys.add(k)
    closestats_tol = {'tolerance_map': {k: (0.1, 0) for k in close_keys}}
    return {**closestats_tol, 'expected_close': expected_close}

def exact_results(expected_exact: Tuple):
    """Result generation function that packages."""
    return {'expected_exact': expected_exact}

def callback_results(callback_results: Dict):
    """Result generation function that packages."""
    return {'callback_results': callback_results}

class DatasetState(NamedTuple):
    dataset_key: TestDatasetKey
    tokenizer_name: str
    deterministic_token_ids: List[int]
    expected_first_fwd_ids: List

class DatasetFingerprint(Enum):
    cust: tuple = test_dataset_state_core_cust
    gpt2: tuple = test_dataset_state_core_gpt2
    llama3: tuple = test_dataset_state_core_llama
    tl: tuple = test_dataset_state_tl
    sl: tuple = test_dataset_state_sl
    gpt2_agnostic: tuple = test_dataset_state_gpt2_dstype_agnostic

def def_results(device_type: str, precision: Union[int, str],
                dataset_fingerprint: DatasetFingerprint = DatasetFingerprint.cust, ds_cfg: str = "train_prof"):
    test_dataset_state = DatasetState(*dataset_fingerprint.value, *EXPECTED_FIRST_FWD_IDS[ds_cfg])
    # wrap result dict such that only the first epoch is checked
    return {0: {"device_type": device_type, "precision": get_model_input_dtype(precision),
                "dataset_state": test_dataset_state}}

RESULT_TYPE_MAPPING = {
    "exact_results": exact_results,
    "close_results": close_results,
    "mem_results": mem_results,
    "callback_results": callback_results,
}

def parity_normalize(test_alias) -> str:
    parity_suffixes = ("_l",)
    for ps in parity_suffixes:
        if test_alias.endswith(ps):
            test_alias = test_alias[:-len(ps)]
            break
    return test_alias

def collect_results(result_map: Dict[str, Tuple], test_alias: str, normalize: bool = True):
    if normalize:
        test_alias = parity_normalize(test_alias)
    test_result: TestResult = result_map[test_alias]
    collected_results = defaultdict(dict)
    for rtype, rfunc in RESULT_TYPE_MAPPING.items():
        if rattr := getattr(test_result, rtype):
            collected_results.update(rfunc(rattr))
    if exp_tol := test_result.tolerance_map:
        collected_results['tolerance_map'].update(exp_tol)
    if test_result.dstype_agnostic:
        collected_results['dstype_agnostic'] = True
    return collected_results
