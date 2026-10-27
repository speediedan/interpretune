import hashlib
import os
import shutil
from pathlib import Path
from jaxtyping import Float, Int
from typing import Any, Dict, Callable, List, Tuple
from unittest import mock
from functools import reduce, partial
from dataclasses import dataclass
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem

import torch
import tqdm
import datasets
import evaluate
from torch.testing import assert_close
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import ModelOutput

from interpretune.protocol import ITModuleProtocol, STEP_OUTPUT
from interpretune.config import ITDataModuleConfig, ITConfig
from interpretune.config.datamodule import hf_datasets_cache_root
from interpretune.utils import rank_zero_only, rank_zero_debug
from interpretune import MemProfilerHooks
from tests.rte_component import RTEBoolqDataModule, RTEBoolqModuleMixin, RTEBoolqSteps, RTE_ENTRYPOINT_MODULE
from tests import FinetuningScheduler
from tests.base_defaults import default_test_task
from tests.results import (
    TEST_TASK_NUM_LABELS,
    TEST_TASK_TEXT_FIELD_MAP,
    NUM_SAMPLE_ROWS,
    SAMPLE_POSITION,
    DatasetState,
    save_memory_footprint_results,
    on_rocm,
    ROCM_MEM_KEY,
    parity_normalize,
)
from tests.parity_acceptance.expected import memory_footprints


################################################################################
# Test DataModules
################################################################################


class BaseTestDataModule:
    def __init__(self, itdm_cfg: ITDataModuleConfig, force_prepare_data: bool = False) -> None:
        with mock.patch.multiple(
            RTE_ENTRYPOINT_MODULE,
            TASK_NUM_LABELS=TEST_TASK_NUM_LABELS,
            TASK_TEXT_FIELD_MAP=TEST_TASK_TEXT_FIELD_MAP,
        ):
            super().__init__(itdm_cfg=itdm_cfg)
        self.force_prepare_data = force_prepare_data
        self._key_default_dataset_path_by_tokenizer()

    def _key_default_dataset_path_by_tokenizer(self) -> None:
        """Give the cached test subset one directory per tokenizer, not one per task.

        ``pytest_rte_hf`` is shared by the gpt2, gemma2, gemma3 and llama3 compositions (``pytest_rte_tl`` by gpt2 and
        gemma2), and :meth:`prepare_data` reuses an existing subset rather than rebuilding it. So on a fresh datasets
        cache, whichever tokenizer prepared the subset first won: a gpt2 session then read gemma or llama token ids,
        beyond its own vocabulary, and its first embedding lookup faulted the device. That fault is sticky, so every
        later GPU test in the process failed too (#794: the out-of-range gather in ``test_peft`` on both vendors, which
        a used cache hid). A ``dataset_path`` a config sets explicitly is left alone.
        """
        task_dir = self.itdm_cfg.task_name.replace(":", "_").replace("|", "_")
        path = Path(self.itdm_cfg.dataset_path).resolve()
        cache_root = hf_datasets_cache_root().resolve()
        if path.parent != cache_root or not (path.name == task_dir or path.name.startswith(f"{task_dir}__")):
            return
        tokenizer = self.tokenizer
        identity = f"{type(tokenizer).__name__}|{getattr(tokenizer, 'name_or_path', '')}|{len(tokenizer)}"
        key = hashlib.sha256(identity.encode()).hexdigest()[:12]
        self.itdm_cfg.dataset_path = cache_root / f"{task_dir}__{key}"

    # dataset fingerprinting is not enabled by default
    def sample_dataset_state(self) -> List:
        return []

    def sample_step_input(self, batch: BatchEncoding) -> List:
        return []

    def prepare_data(self, target_model: torch.nn.Module | None = None) -> None:
        """Load the SuperGLUE dataset."""

        tokenization_func = partial(
            self.encode_for_rteboolq,
            tokenizer=self.tokenizer,
            text_fields=self.itdm_cfg.text_fields,
            prompt_cfg=self.itdm_cfg.prompt_cfg,
            template_fn=self.itdm_cfg.prompt_cfg.model_chat_template_fn,
            tokenization_pattern=self.itdm_cfg.cust_tokenization_pattern,
        )
        dataset_path = Path(self.itdm_cfg.dataset_path).resolve()

        rank_zero_debug("[PREPARE_DATA] Environment variables:")
        for env_var in ["HF_DATASETS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"]:
            rank_zero_debug(f"[PREPARE_DATA]   {env_var}={os.environ.get(env_var, 'NOT_SET')}")

        rank_zero_debug(f"[PREPARE_DATA] itdm_cfg.dataset_path: {self.itdm_cfg.dataset_path}")
        rank_zero_debug(f"[PREPARE_DATA] resolved dataset_path: {dataset_path}")
        rank_zero_debug(f"[PREPARE_DATA] dataset_path.exists(): {dataset_path.exists()}")
        rank_zero_debug(f"[PREPARE_DATA] force_prepare_data: {self.force_prepare_data}")
        rank_zero_debug(f"[PREPARE_DATA] enable_datasets_cache: {self.itdm_cfg.enable_datasets_cache}")
        rank_zero_debug(
            f"[PREPARE_DATA] Tokenizer config in prepare_data: "
            f"{self.tokenizer.__class__.__name__}, "
            f"padding_side={getattr(self.tokenizer, 'padding_side', None)}, "
            f"model_input_names={getattr(self.tokenizer, 'model_input_names', None)}"
        )

        # rebuild the test dataset if it does not exist in the test environment
        if not dataset_path.exists() or self.force_prepare_data:
            # regen a cached 'pytest_{default_test_task}_{hf,pt,tl}' subset of the default test task for testing with
            # a given model
            dataset = datasets.load_dataset("aps/super_glue", default_test_task)
            for split in dataset.keys():
                dataset[split] = dataset[split].select(range(10))
                dataset[split] = dataset[split].map(tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
                dataset[split] = self._remove_unused_columns(dataset[split])
            # Overwriting the dataset in place rewrites Arrow files that earlier tests in this
            # process may still hold memory-mapped. POSIX allows that silently; Windows refuses with
            # OSError(EINVAL) from the writer open (18 stderr tracebacks per CI run from a
            # swallowed exception here, loud enough to misdirect unrelated failure diagnosis, while
            # force_prepare silently degraded to keeping the stale copy). Build beside the target and
            # swap; when live maps block the swap, keep the existing copy and say so in ONE debug
            # line. Debug, not warn: on Windows live maps block the swap on effectively every
            # mid-suite rebuild, so a warning here fires routinely and trips the parity tests'
            # unexpected-warnings assertion (5 failures the noise this block used to emit had been
            # burying).
            if dataset_path.exists():
                rebuild_path = dataset_path.with_name(dataset_path.name + ".rebuild")
                shutil.rmtree(rebuild_path, ignore_errors=True)
                dataset.save_to_disk(rebuild_path)
                try:
                    shutil.rmtree(dataset_path)
                    rebuild_path.rename(dataset_path)
                    rank_zero_debug(f"[PREPARE_DATA] rebuilt dataset swapped into {dataset_path}")
                except OSError as e:
                    shutil.rmtree(rebuild_path, ignore_errors=True)
                    rank_zero_debug(
                        f"[PREPARE_DATA] kept existing dataset at {dataset_path}: in-place rebuild "
                        f"blocked by live memory-maps ({type(e).__name__}: {e})"
                    )
            else:
                dataset.save_to_disk(dataset_path)


class TestITDataModule(BaseTestDataModule, RTEBoolqDataModule): ...


class SampleDatasetStateMixin:
    def sample_unpadded_state(self, rows: List) -> List:
        # we strip padding from sampled rows before collecting ids to make our sample padding-side agnostic
        return [list(filter(lambda v: v != self.tokenizer.pad_token_id, t))[SAMPLE_POSITION] for t in rows]

    def sample_dataset_state(self) -> List:
        # NOTE [Dataset State Validation]:
        # note that this only validates the loaded dataset/tokenizer, the dataloaders are not tested in this method
        # so one may still need to inspect downstream variables (e.g. the dataloader kwargs) and the batch actually
        # passed to the model in a given test/step to verify that the tested model inputs align with the expected
        # deterministic dataset state defined in `tests.helpers.cfg_aliases.test_dataset_state`
        sample_state = []
        for split in self.dataset.keys():
            target_input = self.tokenizer.model_input_names[0]
            # as a content heuristic, inspect the id of a given position (sample_pos) for the first sample_rows of each
            # dataset split
            sampled_rows = self.dataset[split][target_input][:NUM_SAMPLE_ROWS]
            sample_state.extend(self.sample_unpadded_state(sampled_rows))
        return sample_state

    def sample_step_input(self, batch: BatchEncoding) -> List:
        # See NOTE [Dataset State Validation]
        sample_state = []
        sampled_rows = batch[self.tokenizer.model_input_names[0]].cpu().tolist()
        # inspect the id of a given position for each batch example
        sample_state.extend(self.sample_unpadded_state(sampled_rows))
        return sample_state


class FingerprintTestITDataModule(SampleDatasetStateMixin, BaseTestDataModule, RTEBoolqDataModule): ...


################################################################################
# Toy Configurable Transformer (non-TransformerLens)
# A toy configurable non-TransformerLens based transformer originally based on
# https://bit.ly/toy_transformer. The intention is to ensure we test a more
# heterogenous set of toy configurable transformer implementations
################################################################################


@dataclass
class TestModelArgs:
    n_layers: int = 1
    vocab_size: int = 50257
    max_seq_len: int = 10
    dim: int = 10
    n_heads: int = 2
    dropout_p: float = 0.1
    use_attn_mask: bool = True
    weight_tying: bool = True
    tokenizer: Callable | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    # handle below can be used at runtime to allow this model's `generate` to adapt to various configuration contexts
    ctx_handle: ITModuleProtocol | None = None

    def __post_init__(self):
        if self.ctx_handle:
            # snag potentially useful context references and then delete the handle
            self.tokenizer = self.tokenizer or self.ctx_handle.it_cfg.tokenizer
            self.device = self.device or self.ctx_handle.device
            self.dtype = self.dtype or self.ctx_handle.dtype
            del self.ctx_handle


class Attention(torch.nn.Module):
    def __init__(self, args: TestModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = torch.nn.Dropout(args.dropout_p)
        self.use_attn_mask = args.use_attn_mask

        self.wq = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.wk = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.wv = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.wo = torch.nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        output = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
            self.use_attn_mask,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim)
        self.gelu = torch.nn.GELU()
        self.w2 = torch.nn.Linear(hidden_dim, dim)
        self.resid_dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))


class TransformerBlock(torch.nn.Module):
    def __init__(self, args: TestModelArgs):
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = torch.nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(torch.nn.Module):
    def __init__(self, args: TestModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.device = args.device
        self.dtype = args.dtype
        self.tokenizer = args.tokenizer
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = torch.nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = torch.nn.Dropout(args.dropout_p)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = torch.nn.LayerNorm(args.dim)
        self.output = torch.nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @torch.inference_mode()
    def generate(
        self,
        tokens: str | Float[torch.Tensor, "batch pos"] = "",
        max_new_tokens: int = 5,
        eos_token_id: int | None = None,
        output_logits: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> ModelOutput | Int[torch.Tensor, "batch pos_plus_new_tokens"]:
        """Toy generate function to support non-HF/TransformerLens tests with the same interface.

        Args:
            tokens (str | Int[torch.Tensor, "batch pos"])): A batch of tokens ([batch, pos]).
            max_new_tokens (int): Maximum number of tokens to generate.
            eos_token_id (int | Sequence | None): The token ID to use for end of sentence.
            output_logits (`bool`, *optional*, defaults to `False`): Whether or not to return the prediction scores.
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens.
        """
        # To enable a broader range of testing contexts, use the configuration context of the parent_handle
        # TODO: update this method to use parent_handle if available for broader range of testing
        out_logits = [] if output_logits else None

        assert isinstance(tokens, torch.Tensor)
        batch_size, ctx_length = tokens.shape
        gen_device = self.device or tokens.device
        tokens = tokens.to(gen_device)

        stop_tokens = []
        eos_token_for_padding = 0

        tokenizer_has_eos_token = self.tokenizer is not None and self.tokenizer.eos_token_id is not None
        if eos_token_id is None:
            assert tokenizer_has_eos_token, "Must pass an `eos_token_id` if tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id

        stop_tokens = [eos_token_id]
        eos_token_for_padding = eos_token_id

        # An array to track which sequences in the batch have finished.
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=gen_device)

        for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            # While generating, we keep generating logits, throw away all but the final logits,
            # and then use those logits to sample from the distribution We keep adding the
            # sampled tokens to the end of tokens.
            # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
            # the cache.
            logits = self.forward(tokens)
            final_logits = logits[:, -1, :]
            if output_logits:
                out_logits.append(final_logits)

            sampled_tokens = final_logits.argmax(-1).to(gen_device)

            # For all unfinished sequences, add on the next token. If a sequence was
            # finished, throw away the generated token and add eos_token_for_padding
            # instead.
            sampled_tokens[finished_sequences] = eos_token_for_padding
            finished_sequences.logical_or_(torch.isin(sampled_tokens, torch.tensor(stop_tokens).to(gen_device)))

            tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

            if finished_sequences.all():
                break

        if output_logits:
            assert out_logits is not None
            # For HF semantics, when `output_logits` is requested, return a ModelOutput with `logits`
            return ModelOutput(sequences=tokens, logits=tuple(out_logits))
        else:
            return tokens


def get_filesystem(path: str | Path, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


################################################################################
# Test Modules
# All test modules inherit from BaseTestModule and can be used in the same way,
# adjusting the desired methods and configuration to load a pretrained model
# (e.g. HF `from_pretrained`) or custom model and using the desired adapter
# compositions (e.g. `transformer_lens`, `Lightning`, native PyTorch)
################################################################################


class StateLogInspectMixin:
    def __init__(
        self,
        *args,
        expected_exact: Dict | None = None,
        expected_close: Dict | None = None,
        expected_memstats: Tuple | None = None,
        tolerance_map: Dict | None = None,
        test_alias: str | None = None,
        state_log_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.expected_memstats = expected_memstats
        self.expected_exact = expected_exact
        self.expected_close = expected_close
        self.state_log_dir = state_log_dir
        self.test_alias = test_alias
        self.tolerance_map = tolerance_map or {}
        self.dev_expected_exact = {}
        self.dev_expected_close = {}
        super().__init__(*args, **kwargs)

    def inspect_or_assert(self, current_exact, current_close, state_key) -> None:
        if not self.state_log_dir:
            if self.expected_exact and self.expected_exact.get(state_key, None):
                for exp_k, exp_v in self.expected_exact[state_key].items():
                    assert current_exact[exp_k] == exp_v
            if self.expected_close and self.expected_close.get(state_key, None):
                for exp_k, exp_v in self.expected_close[state_key].items():
                    rtol, atol = self.tolerance_map.get(exp_k, (0, 0))
                    assert_close(actual=current_close[exp_k], expected=exp_v, rtol=rtol, atol=atol)
        else:
            self.dev_expected_exact[state_key] = current_exact
            self.dev_expected_close[state_key] = current_close

    @rank_zero_only
    def log_dev_state(self) -> None:
        if self.expected_memstats:
            save_memory_footprint_results(memory_footprints)
        dump_path = Path(self.state_log_dir)
        state_log = dump_path / "dev_state_log.yaml"
        fs = get_filesystem(state_log)
        with fs.open(state_log, "a", newline="") as fp:
            fp.write(f"State log for test `{self.test_alias}`:{os.linesep}")
            for dev_d in [self.dev_expected_exact, self.dev_expected_close]:
                fp.write(os.linesep)
                for k, v in dev_d.items():  # control formatting precisely to allow copy/paste expected output
                    fp.write(f"{' ' * 8}{k}: {v},{os.linesep}")

    def _validate_memory_stats(self) -> None:
        """Compare every expected stat, then fail once naming all that differ, so one run measures them all.

        Logging dev state on ROCm records into the footprint's ROCm overrides, leaving the CUDA expectation as it is.
        """
        mismatches = []
        for act, exp in self.expected_memstats[1].items():
            actual_val = self.memprofiler.memory_stats[self.expected_memstats[0]][act]
            if not self.state_log_dir:
                rtol, atol = self.tolerance_map.get(act, (0, 0))
                try:
                    assert_close(actual=actual_val, expected=exp, rtol=rtol, atol=atol)
                except AssertionError:
                    mismatches.append(f"{act}: expected {exp} but got {actual_val} (rtol={rtol}, atol={atol})")
            else:
                self.dev_expected_close[act] = actual_val
                footprint = memory_footprints[parity_normalize(self.test_alias)]
                key = ROCM_MEM_KEY if on_rocm() else "expected_mem"
                footprint.setdefault(key, {})[act] = actual_val
        assert not mismatches, f"{len(mismatches)} memory stat(s) differ: " + "; ".join(mismatches)


class DivergeOnEpochMixin:
    """Mixin that adds diverging loss behavior starting at a specified epoch.

    This mixin overrides training_step and validation_step to blend real logits with fake (zeros) logits based on
    current_epoch/max_epochs ratio, causing gradual loss divergence.
    """

    def __init__(self, *args, diverge_on_epoch: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        it_cfg = reduce(lambda o, a: getattr(o, a, None), ("it_cfg", "model_cfg"), self)
        self.diverge_on_epoch = it_cfg.get("diverge_on_epoch", diverge_on_epoch) if it_cfg else diverge_on_epoch

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        it_cfg = super()._before_it_cfg_init(it_cfg) if hasattr(super(), "_before_it_cfg_init") else it_cfg
        if it_cfg.model_cfg and "diverge_on_epoch" in it_cfg.model_cfg:
            self.diverge_on_epoch = it_cfg.model_cfg.get("diverge_on_epoch", None)
        if self.diverge_on_epoch is not None:
            rank_zero_debug(f"Divergence enabled: will diverge starting at epoch {self.diverge_on_epoch}")
        return it_cfg

    def _compute_diverging_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss with gradual divergence using additive penalty.

        Adds a baseline-relative penalty that increases from 0.5x baseline at diverge_on_epoch to 2.0x baseline at
        max_epochs. This ensures monotonic loss increase regardless of learning improvements from parameter thawing,
        enabling reliable cross-architecture testing.
        """
        base_loss = self.loss_fn(logits, labels)

        if (
            self.diverge_on_epoch is not None
            and hasattr(self, "current_epoch")
            and self.current_epoch >= self.diverge_on_epoch
        ):
            # Store baseline loss at divergence start (first epoch where divergence activates)
            if not hasattr(self, "_divergence_baseline"):
                self._divergence_baseline = base_loss.detach().item()

            # Calculate additive penalty based on progress from diverge_on_epoch to max_epochs
            max_epochs = self.trainer.max_epochs if hasattr(self, "trainer") else 5
            epochs_diverging = self.current_epoch - self.diverge_on_epoch + 1
            total_diverge_epochs = max(max_epochs - self.diverge_on_epoch, 1)
            progress = min(epochs_diverging / total_diverge_epochs, 1.0)

            # Add increasing penalty: starts at 0.5x baseline, grows to 2.0x baseline
            penalty = self._divergence_baseline * (0.5 + 1.5 * progress)
            diverged_loss = base_loss + penalty

            rank_zero_debug(
                f"Epoch {self.current_epoch} >= {self.diverge_on_epoch}: DIVERGING "
                f"(penalty={penalty:.3f}, baseline={self._divergence_baseline:.3f}, "
                f"base={base_loss.item():.3f}, total={diverged_loss.item():.3f}, progress={progress:.3f})"
            )
            return diverged_loss
        else:
            # Normal behavior: use original loss function
            return base_loss

    @MemProfilerHooks.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        answer_logits, labels, _ = self.logits_and_labels(batch, batch_idx)
        loss = self._compute_diverging_loss(answer_logits, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @MemProfilerHooks.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT | None:
        answer_logits, labels, orig_labels = self.logits_and_labels(batch, batch_idx)
        val_loss = self._compute_diverging_loss(answer_logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.collect_answers(answer_logits, orig_labels)


class BaseTestModule(StateLogInspectMixin):
    def __init__(self, *args, req_grad_mask: Dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.req_grad_mask = req_grad_mask or {}
        self.epoch_losses = {}
        self.sampled_fwd_inputs = None

    # TODO: use mock TASK_NUM_LABELS here (not just in __init__) to rely on rteboolq example method here for testing
    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        it_cfg.num_labels = 0 if it_cfg.generative_step_cfg.enabled else 2
        return it_cfg

    def load_metric(self) -> None:
        self.metric = evaluate.load(
            "super_glue", default_test_task, experiment_id=self._it_state._init_hparams["experiment_id"]
        )

    def model_init(self) -> None:
        """If we're not using a from-configuration model, we initialize the model here."""
        self.device = self.it_cfg.model_cfg.get("device", None) or self.device
        self.dtype = self.it_cfg.model_cfg.get("dtype", None) or self.dtype
        cust_config = TestModelArgs(**self.it_cfg.model_cfg.get("model_args", {}), ctx_handle=self)
        self.model = Transformer(args=cust_config)
        # TODO: add note to documentation that for now user is responsible for setting device/dtype w model_init
        self.model.to(device=self.device, dtype=self.dtype)

    def _get_current_exact(self) -> Dict:
        # gather device and precision and dataset state info
        device_type = self.device.type if isinstance(self.device, torch.device) else self.output_device.type
        model_dtype = self.model.dtype if hasattr(self.model, "dtype") else self.tl_cfg.dtype
        return {"device_type": device_type, "precision": model_dtype, **self._get_dataset_state()}

    def _get_dataset_state(self) -> Dict:
        dataset_state = (
            self.datamodule.tokenizer.__class__.__name__,
            self.datamodule.sample_dataset_state(),
            self.sampled_fwd_inputs,
        )
        return {"dataset_state": DatasetState(*dataset_state)}

    def _epoch_end_validation(self, *args, **kwargs) -> None:
        state_key = self.current_epoch
        current_close = {}
        if self.epoch_losses and self.epoch_losses.get(state_key, None):
            current_close.update({"loss": self.epoch_losses[state_key]})
        self.inspect_or_assert(self._get_current_exact(), current_close, state_key)

    def configure_optimizers(self):
        # maybe apply a given `req_grad_mask` (for non-FTS context testing)
        # usually should be used for a small subset of parameters so we don't iterate through ``named_parameters`` but
        # use `get_parameter`` call instead
        # A name the model does not hold may name a parameter of an SAE spliced at init (`attached_saes`, #772). A
        # name neither resolves is a broken test configuration, so it fails here rather than leaving the mask a
        # silent no-op (which once let a parity case "freeze" SAE parameters that were never in the optimizer).
        for n, val in self.req_grad_mask.items():
            try:
                p = self.model.get_parameter(n)
            except AttributeError:
                if not hasattr(self, "attached_sae_parameter"):
                    raise
                p = self.attached_sae_parameter(n)
            p.requires_grad_(val)
        return super().configure_optimizers()

    def _on_test_or_train_batch_start(self, batch: Any, batch_idx: int, *args, **kwargs):
        if self.global_step == 0 and batch_idx == 0:
            self.sampled_fwd_inputs = self.datamodule.sample_step_input(batch)

    def on_test_batch_start(self, batch: Any, batch_idx: int, *args, **kwargs):
        self._on_test_or_train_batch_start(batch, batch_idx, *args, **kwargs)

    def on_train_batch_start(self, batch: Any, batch_idx: int, *args, **kwargs):
        self._on_test_or_train_batch_start(batch, batch_idx, *args, **kwargs)

    def on_train_batch_end(self, batch: Any, batch_idx: int, outputs: STEP_OUTPUT, *args, **kwargs):
        if self.epoch_losses.get(self.current_epoch, None) is None:
            try:
                if isinstance(outputs, torch.Tensor):
                    self.epoch_losses[self.current_epoch] = outputs.item()
                # TODO: add this helper attribute to CoreHelperAttributes?
                elif callback_metrics := reduce(getattr, "trainer.callback_metrics".split("."), self):
                    self.epoch_losses[self.current_epoch] = callback_metrics["train_loss"].item()
            except AttributeError as ae:
                raise AttributeError(
                    f"Could not find and log loss output for epoch {self.current_epoch}, "
                    f"batch {batch_idx}, received: {ae}"
                )
            assert self.epoch_losses.get(self.current_epoch, None) is not None, "No loss recorded for current epoch!"

    def on_test_epoch_end(self, *args, **kwargs):
        self._epoch_end_validation(*args, **kwargs)

    def on_train_epoch_start(self, *args, **kwargs):
        pass  # TODO: planning to add some on epoch start validation

    def on_train_epoch_end(self, *args, **kwargs):
        self._epoch_end_validation(*args, **kwargs)

    def on_session_end(self) -> Any | None:
        super().on_session_end()
        if self.it_cfg.memprofiler_cfg and self.expected_memstats:
            self._validate_memory_stats()
        if self.state_log_dir:
            self.log_dev_state()


class TestITModule(BaseTestModule, RTEBoolqSteps, RTEBoolqModuleMixin): ...


class DivergeTestITModule(DivergeOnEpochMixin, BaseTestModule, RTEBoolqSteps, RTEBoolqModuleMixin):
    """Test module with diverging loss behavior for FTS multi-level schedule testing."""

    pass


class CanonicalViewDivergeTestITModule(DivergeTestITModule):
    """``DivergeTestITModule`` declaring the identity view, so its schedules name the module's own parameters.

    A composition chooses its view by overriding ``build_tunable_view()``; this one does so over a TransformerLens
    composition, whose own view resolves TransformerLens names.
    """

    def build_tunable_view(self):
        from interpretune.tuning.views import identity_view

        return identity_view(self, roles=self.tunable_roles())


class TestFTS(StateLogInspectMixin, FinetuningScheduler):
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        state_key = trainer.current_epoch
        fts_current_state = {
            "curr_depth": self.curr_depth,
            "depth_remaining": self.depth_remaining,
            "ft_epoch": self._fts_state._ft_epoch,
            "current_ckpt_depth": self._fts_state._fts_ckpt_metadata["current_ckpt_depth"],
            "best_ckpt_depth": self._fts_state._fts_ckpt_metadata["best_ckpt_depth"],
            "best_ckpt_pgs_len": len(self._fts_state._fts_ckpt_metadata["best_ckpt_pgs"]),
            "curr_thawed_params": len(self._fts_state._curr_thawed_params),
            "optim_pg_len": len(self._internal_optimizer_metadata[0]),
            "ckpt_cback_current_ckpt_depth": trainer.checkpoint_callback.current_ckpt_depth,
            "ckpt_cback_best_ckpt_depth": trainer.checkpoint_callback.best_ckpt_depth,
        }
        current_close = {}
        self.inspect_or_assert(fts_current_state, current_close, state_key)

    def on_train_end(self, trainer, pl_module) -> None:
        if self.state_log_dir:
            self.log_dev_state()
