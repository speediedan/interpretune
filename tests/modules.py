import os
from pathlib import Path
from jaxtyping import Float, Int
from typing import Optional, Any, Dict, Tuple, NamedTuple, Union, Callable, List
from unittest import mock
from functools import reduce
from dataclasses import dataclass
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem

import torch
import tqdm
import datasets
import evaluate
from torch.testing import assert_close
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.contract.protocol import ITModuleProtocol
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.utils.types import STEP_OUTPUT
from interpretune.utils.logging import rank_zero_only
from it_examples.experiments.rte_boolq.modules import RTEBoolqModuleMixin, RTEBoolqSteps
from it_examples.experiments.rte_boolq.datamodules import GPT2RTEBoolqDataModule, Llama2RTEBoolqDataModule
from tests.parity_acceptance.base.cfg_aliases import (TEST_TASK_NUM_LABELS, TEST_TASK_TEXT_FIELD_MAP, NUM_SAMPLE_ROWS,
                                                      SAMPLE_POSITION)


################################################################################
# Test DataModules
################################################################################

class BaseTestDataModule:

    def __init__(self, itdm_cfg: ITDataModuleConfig, force_prepare_data: bool = False) -> None:
        with mock.patch.multiple('it_examples.experiments.rte_boolq.datamodules', TASK_NUM_LABELS=TEST_TASK_NUM_LABELS,
                                 TASK_TEXT_FIELD_MAP=TEST_TASK_TEXT_FIELD_MAP):
            super().__init__(itdm_cfg=itdm_cfg)
        self.force_prepare_data = force_prepare_data

    def sample_unpadded_state(self, rows: List) -> List:
        # we strip padding from sampled rows before collecting ids to make our sample padding-side agnostic
        return [list(filter(lambda v: v != self.tokenizer.pad_token_id, t))[SAMPLE_POSITION] for t in rows]

    def sample_dataset_state(self) -> Tuple:
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
        return (self.itdm_cfg.task_name, self.tokenizer.__class__.__name__, sample_state)

    def sample_step_input(self, batch: BatchEncoding) -> Tuple:
        # See NOTE [Dataset State Validation]
        sample_state = []
        sampled_rows = batch[self.tokenizer.model_input_names[0]].cpu().tolist()
        # inspect the id of a given position for each batch example
        sample_state.extend(self.sample_unpadded_state(sampled_rows))
        return sample_state

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        dataset_path = Path(self.itdm_cfg.dataset_path)
        # rebuild the test dataset if it does not exist in the test environment
        if not dataset_path.exists() or self.force_prepare_data:
            # regen a cached 'pytest_rte_{hf,pt,tl}' subset of rte for testing with a given model
            dataset = datasets.load_dataset("super_glue", 'rte', trust_remote_code=True)
            for split in dataset.keys():
                dataset[split] = dataset[split].select(range(10))
                dataset[split] = dataset[split].map(self.tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
                dataset[split] = self._remove_unused_columns(dataset[split])
            # TODO: remove below temporary workaround (converting Path to str) once upstream issue resolved
            # looks like https://github.com/huggingface/datasets/pull/6704 broke `save_to_disk` method for
            # Path objects https://bit.ly/datasets_save_to_disk_blame
            dataset.save_to_disk(str(dataset_path))
            #dataset.save_to_disk(dataset_path)

class TestITDataModule(BaseTestDataModule, GPT2RTEBoolqDataModule):
    ...

class Llama2TestITDataModule(BaseTestDataModule, Llama2RTEBoolqDataModule):
    def sample_dataset_state(self) -> Tuple:
        # dataset validation not currently implemented for Llama2 tests
        sample_state = []
        return (self.itdm_cfg.task_name, self.tokenizer.__class__.__name__, sample_state)

    def sample_step_input(self, batch: BatchEncoding) -> Tuple:
        # dataset validation not currently implemented for Llama2 tests
        sample_state = []
        return sample_state

class SampledOutput(NamedTuple):
    """Sampled Output Named Tuple.

    Named tuple object for if we want to output both logits and tokens.
    """

    tokens: Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]
    logits: Float[torch.Tensor, "batch pos d_vocab"]


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
    tokenizer: Optional[Callable] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    # handle below can be used at runtime to allow this model's `generate` to adapt to various configuration contexts
    ctx_handle: Optional[ITModuleProtocol] = None

    def __post_init__(self):
        if self.ctx_handle:
            # snag potentially useful context references and then delete the handle
            self.tokenizer = self.tokenizer or self.ctx_handle.it_cfg.tokenizer
            self.device = self.device or self.ctx_handle.device
            self.dtype = self.dtype or self.ctx_handle.torch_dtype
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
            queries, keys, values, None,
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
        tokens: Union[str, Float[torch.Tensor, "batch pos"]] = "",
        max_new_tokens: int = 5,
        eos_token_id: Optional[int] = None,
        output_logits: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Union[SampledOutput, Int[torch.Tensor, "batch pos_plus_new_tokens"]]:
        """Toy generate function to support non-HF/TransformerLens tests with the same interface.

        Args:
            tokens (Union[str, Int[torch.Tensor, "batch pos"])]): A batch of tokens ([batch, pos]).
            max_new_tokens (int): Maximum number of tokens to generate.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end of sentence.
            output_logits (`bool`, *optional*, defaults to `False`): Whether or not to return the prediction scores.
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens.
        """
        # To enable a broader range of testing contexts, use the configuration context of the parent_handle
        # TODO: update this method to use parent_handle if available for broader range of testing
        out_logits = () if output_logits else None

        assert isinstance(tokens, torch.Tensor)
        batch_size, ctx_length = tokens.shape
        gen_device = self.device or tokens.device
        tokens = tokens.to(gen_device)

        stop_tokens = []
        eos_token_for_padding = 0

        tokenizer_has_eos_token = (self.tokenizer is not None and self.tokenizer.eos_token_id is not None)
        if eos_token_id is None:
            assert (tokenizer_has_eos_token), \
            "Must pass an `eos_token_id` if tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id

        stop_tokens = [eos_token_id]
        eos_token_for_padding = eos_token_id

        # An array to track which sequences in the batch have finished.
        finished_sequences = torch.zeros(
            batch_size, dtype=torch.bool, device=gen_device
        )

        for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            # While generating, we keep generating logits, throw away all but the final logits,
            # and then use those logits to sample from the distribution We keep adding the
            # sampled tokens to the end of tokens.
            # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
            # the cache.
            logits = self.forward(tokens)
            final_logits = logits[:, -1, :]
            if output_logits:
                out_logits += (final_logits,)

            sampled_tokens = final_logits.argmax(-1).to(gen_device)

            # For all unfinished sequences, add on the next token. If a sequence was
            # finished, throw away the generated token and add eos_token_for_padding
            # instead.
            sampled_tokens[finished_sequences] = eos_token_for_padding
            finished_sequences.logical_or_(
                torch.isin(sampled_tokens, torch.tensor(stop_tokens).to(gen_device))
            )

            tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

            if finished_sequences.all():
                break

        if output_logits:
            return SampledOutput(tokens, torch.stack(out_logits, dim=1))
        else:
            return tokens

def get_filesystem(path: str | Path, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs

################################################################################
# Test Modules
# All test modules inherit from BaseTestModule and can be used in the same way,
# adjusting the desired methods and configuration to load a pretrained model
# (e.g. HF `from_pretrained`) or custom model and using the desired plugins
# (e.g. `transformer_lens`) while running on a supported famework
# (initially Lightning) or native PyTorch.
################################################################################

class BaseTestModule:
    def __init__(self, it_cfg: ITConfig, expected_exact: Optional[Dict] = None, expected_close: Optional[Dict] = None,
                expected_memstats: Optional[Dict] = None, tolerance_map: Optional[Dict] = None,
                test_alias: Optional[str] = None, state_log_dir: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(it_cfg=it_cfg)
        self.expected_memstats = expected_memstats
        self.expected_exact = expected_exact
        self.expected_close = expected_close
        self.state_log_dir = state_log_dir
        self.test_alias = test_alias
        self.tolerance_map = tolerance_map or {}
        self.epoch_losses = {}
        self.sampled_fwd_inputs = None
        self.dev_expected_exact = {}
        self.dev_expected_close = {}

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)

    # TODO: use mock TASK_NUM_LABELS here (not just in __init__) to rely on rteboolq example method here for testing
    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        it_cfg.num_labels = 0 if it_cfg.zero_shot_cfg.enabled else 2
        return it_cfg

    def load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", 'rte', experiment_id=self._it_state._init_hparams['experiment_id'])

    def model_init(self) -> None:
        """If we're not using a from-configuration model, we initialize the model here."""
        self.device = self.it_cfg.model_cfg.get("device", None) or self.device
        self.torch_dtype = self.it_cfg.model_cfg.get("dtype", None) or self.torch_dtype
        cust_config = TestModelArgs(**self.it_cfg.model_cfg.get('model_args', {}), ctx_handle=self)
        self.model = Transformer(args=cust_config)
        # TODO: add note to documentation that for now user is responsible for setting device/dtype w model_init
        self.model.to(device=self.device, dtype=self.torch_dtype)

    def _get_current_exact(self) -> Dict:
        # gather device and precision info for both core and transformer lens contexts
        device_type = self.device.type if isinstance(self.device, torch.device) else self.output_device.type
        model_dtype = self.model.dtype if hasattr(self.model, "dtype") else self.tl_cfg.dtype
        return {'device_type': device_type, 'precision': model_dtype, **self._get_dataset_state()}

    def _get_dataset_state(self) -> Dict:
        return {'dataset_state': self.datamodule.sample_dataset_state() + (self.sampled_fwd_inputs,)}

    def _epoch_end_validation(self, *args, **kwargs) -> None:
        state_key = self.current_epoch
        current_close = {}
        if self.epoch_losses and self.epoch_losses.get(state_key, None):
            current_close.update({'loss': self.epoch_losses[state_key]})
        self.inspect_or_assert(self._get_current_exact(), current_close, state_key)

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
                # TODO: add this helper attribute to CoreHerlperAttributes?
                elif callback_metrics := reduce(getattr, "trainer.callback_metrics".split("."), self):
                    self.epoch_losses[self.current_epoch] = callback_metrics['train_loss'].item()
            except AttributeError as ae:
                raise AttributeError(f"Could not find and log loss output for epoch {self.current_epoch}, "
                                    f"batch {batch_idx}, received: {ae}")
            assert self.epoch_losses.get(self.current_epoch, None) is not None, 'No loss recorded for current epoch!'

    def on_test_epoch_end(self, *args, **kwargs):
        self._epoch_end_validation(*args, **kwargs)

    def on_train_epoch_start(self, *args, **kwargs):
        pass  # TODO: planning to add some on epoch start validation

    def on_train_epoch_end(self, *args, **kwargs):
        self._epoch_end_validation(*args, **kwargs)

    def on_session_end(self) -> Optional[Any]:
        super().on_session_end()
        if self.it_cfg.memprofiler_cfg and self.expected_memstats:
            self._validate_memory_stats()
        if self.state_log_dir:
            self.log_dev_state()

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
        dump_path = Path(self.state_log_dir)
        state_log = dump_path / "dev_state_log.yaml"
        fs = get_filesystem(state_log)
        with fs.open(state_log, "w", newline="") as fp:
            fp.write(f"State log for test `{self.test_alias}`:{os.linesep}")
            for dev_d in [self.dev_expected_exact, self.dev_expected_close]:
                fp.write(os.linesep)
                for k, v in dev_d.items():  # control formatting precisely to allow copy/paste expected output
                    fp.write(f"{' ' * 8}{k}: {v},{os.linesep}")

    def _validate_memory_stats(self) -> None:
        for act, exp in zip(self.expected_memstats[1], self.expected_memstats[2]):
            if not self.state_log_dir:
                rtol, atol = self.tolerance_map.get(act, (0, 0))
                assert_close(actual=self.memprofiler.memory_stats[self.expected_memstats[0]][act], expected=exp,
                             rtol=rtol, atol=atol)
            else:
                self.dev_expected_close[act] = self.memprofiler.memory_stats[self.expected_memstats[0]][act]


class TestITModule(BaseTestModule, RTEBoolqSteps, RTEBoolqModuleMixin):
    ...
