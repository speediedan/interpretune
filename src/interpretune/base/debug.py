import os
import pickle
from dataclasses import fields
from typing import Optional, List, Any, Dict, Tuple, Union
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, Dataset
from psutil import Process

from interpretune.utils.logging import rank_zero_warn, _get_rank, rank_zero_only
from interpretune.base.config_classes import MemProfilerCfg
from interpretune.utils.import_utils import resolve_funcs


import torch
from tqdm import tqdm

# accessed in global scope to track non-parameter packed bytes (npp) as a simple proxy (ceiling) for activation memory
_npp_bytes = 0

def _hook_npp_pre_forward(module, *args, **kwargs):
    mem = module.mem_info_handle()
    global _npp_bytes
    module.npp_pre_forward = _npp_bytes
    module.rss_pre_forward = mem.rss
    return None

def _hook_npp_post_forward(module, *args, **kwargs):
    global _npp_bytes
    module.npp_post_forward = _npp_bytes
    module.npp_diff = module.npp_post_forward - module.npp_pre_forward
    mem = module.mem_info_handle()
    module.rss_post_forward = mem.rss
    rss_diff = module.rss_post_forward - module.rss_pre_forward
    module.rss_diff = rss_diff + (module.rss_diff if hasattr(module, "rss_diff") else 0)
    return None

def _reset_memory_hooks_state(model, reset_attrs: List[str]):
    global _npp_bytes
    _npp_bytes = 0
    for module in model.modules():
        for attr in reset_attrs:
            setattr(module, attr, 0)

def _npp_hook(x):
    global _npp_bytes
    if not isinstance(x, torch.nn.Parameter):
        _npp_bytes += x.nbytes
    return x

class MemProfiler:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.memory_stats = defaultdict(dict)
        self._enabled = {}
        self._module = None
        self._cuda_snapshot_dir = None
        self._curr_pid = None
        self._rank = _get_rank() or 0  # for future use, currently only single rank supported
        self._snap_indices = {}
        self._configured_hooks = {}
        self._saved_tensors_funcs = []
        self._hook_handles = defaultdict(list)
        self._done_prof_funcs = []

    def connect(self, obj_ref: Any) -> None:
        self._module = obj_ref
        self._curr_pid = Process(os.getpid())
        if self.memprofiler_cfg.enable_saved_tensors_hooks:
            self._saved_tensors_funcs = resolve_funcs(cfg_obj=self.memprofiler_cfg, func_type='saved_tensors_funcs')

    @property
    def memprofiler_cfg(self) -> MemProfilerCfg:
        return self._module.it_cfg.memprofiler_cfg

    @property
    def schedule(self) -> int:
        return self._module.it_cfg.memprofiler_cfg.schedule

    def remove_memprofiler_hooks(self) -> None:
        for handle_list in self._hook_handles.values():
            for handle in handle_list:
                handle.remove()

    def exec_reset_state_hooks(self) -> None:
        for hook in self._configured_hooks["reset_state_hooks"]:
            hook(self._module.model, self.memprofiler_cfg.save_hook_attrs)

    def add_memprofiler_hooks(self) -> None:
        # TODO: extend supported hook points (e.g. backward, etc.) and if/once supporting additional hook points,
        # use a hook_type to registration function mapping
        memory_hooks_cfg = self.memprofiler_cfg.memory_hooks
        has_hooks = any(getattr(memory_hooks_cfg, ht.name) for ht in fields(memory_hooks_cfg))
        if not has_hooks:
            rank_zero_warn("MemProfilerCfg is configured to enable memory hooks but MemProfilerHooks does not have"
                           " any specified.")
        for supported_hooks in fields(memory_hooks_cfg):
            if getattr(memory_hooks_cfg, supported_hooks.name):
                self._configured_hooks[supported_hooks.name] = resolve_funcs(cfg_obj=memory_hooks_cfg,
                                                                             func_type=supported_hooks.name)
        for module in self._module.model.modules():
            module.mem_info_handle = self._curr_pid.memory_info
            for hook_func in self._configured_hooks["pre_forward_hooks"]:
                self._hook_handles[hook_func].append(module.register_forward_pre_hook(hook_func))
            for hook_func in self._configured_hooks["post_forward_hooks"]:
                self._hook_handles[hook_func].append(module.register_forward_hook(hook_func))
        self.exec_reset_state_hooks()

    def init_cuda_snapshots_dir(self) -> None:
        self._cuda_snapshot_dir = self.memprofiler_cfg.save_dir or Path(self._module.core_log_dir) / "memprofiler"
        self._cuda_snapshot_dir = Path(self._cuda_snapshot_dir)  # ensure the dir is a Path
        self._cuda_snapshot_dir.mkdir(exist_ok=True, parents=True)

    def cuda_allocator_history_snap(self, snap_key: Tuple) -> Dict:
        cuda_snapshot_file = (self._cuda_snapshot_dir / ".".join(("cuda_alloc", snap_key))).with_suffix('.pickle')
        torch.cuda.memory._dump_snapshot(cuda_snapshot_file)

    def done(self, step_idx: int) -> bool:
        return self.schedule.max_step and step_idx >= self.schedule.max_step

    def _process_hooks(self, snap_key) -> None:
        if self.memprofiler_cfg.enable_memory_hooks:
            if len(self._hook_handles) == 0:
                self.add_memprofiler_hooks()
            else:
                *_,  step_idx, step_ctx = snap_key
                if step_idx == self.schedule.warmup_steps and step_ctx == 'start':
                    # conservatively reset hooks on first step after warmup since existing hooks may have accumulated
                    # state during untracked steps from a previous phase and pre-warmup steps of the current phase
                    self.exec_reset_state_hooks()
            collected = {attr: getattr(self._module.model, attr, None) for attr in self.memprofiler_cfg.save_hook_attrs}
            self.memory_stats[snap_key].update(collected)

    def _collect_snap(self, snap_key, reset_mem_hooks: bool = False) -> None:
        _, phase, *_ = snap_key
        snap_key = ".".join(map(str, snap_key))
        mem_cfg = self.memprofiler_cfg
        self._process_hooks(snap_key)
        if phase in mem_cfg.enabled_funcs.cpu:
            mem = self._curr_pid.memory_info()
            self.memory_stats[snap_key].update({"rss": mem.rss, "vms": mem.vms})
        if phase in mem_cfg.enabled_funcs.cuda:
            self.memory_stats[snap_key].update(torch.cuda.memory_stats())
        if phase in mem_cfg.enabled_funcs.cuda_allocator_history and mem_cfg.cuda_allocator_history:
            self.cuda_allocator_history_snap(snap_key)
        if mem_cfg.enable_memory_hooks and reset_mem_hooks:
            self.exec_reset_state_hooks()

    @property
    def _should_remove_hooks(self) -> bool:
        return all(func in self._done_prof_funcs for func in self.memprofiler_cfg.retain_hooks_for_funcs)

    def teardown_prof(self, phase: str, step_ctx: str) -> None:
        self._enabled[(phase, step_ctx)] = False
        if not any(self._enabled[(phase, step_ctx)] for step_ctx in ["start", "end"]):
            self._done_prof_funcs.append(phase)
        if self.memprofiler_cfg.retain_hooks_for_funcs and self._should_remove_hooks:
            self.remove_memprofiler_hooks()
            self.memprofiler_cfg.enable_memory_hooks = False

    def gen_snap_keys(self, phase: str, step_ctx: str, epoch_idx: Optional[int] = None,
                      step_idx: Optional[int] = None) -> Tuple[int, int, Tuple]:
        # NOTE [Memprofiler Key Format]
        # snap key format is rank.phase.epoch_idx.step_idx.step_ctx
        # e.g. 0.training_step.0.0.end keys hook output for the end of training step 0, epoch 0 for rank 0
        # 0.training_step.1.2.start keys mem stats for the start of training step 2, epoch 1 for rank 0
        epoch_idx = next(e_idx for e_idx in (epoch_idx, self._module.current_epoch) if e_idx is not None)
        if step_idx is None:
            step_idx = self._snap_indices[(phase, step_ctx)]
        return epoch_idx, step_idx, (self._rank, phase, epoch_idx, step_idx, step_ctx)

    def maybe_init_phase(self, phase: str, step_ctx: str) -> None:
        if not self._snap_indices.get((phase, step_ctx), None):
            self._snap_indices[(phase, step_ctx)] = 0
            self._enabled[(phase, step_ctx)] = True

    def snap(self, phase: str, step_ctx: str, epoch_idx: Optional[int] = None, step_idx: Optional[int] = None,
             reset_mem_hooks: bool = False) -> None:
        self.maybe_init_phase(phase, step_ctx)
        if not self._enabled[(phase, step_ctx)]:
            return
        epoch_idx, step_idx, snap_key = self.gen_snap_keys(phase, step_ctx, epoch_idx, step_idx)
        if step_idx >= self.schedule.warmup_steps:
            if not self.done(step_idx):
                self._collect_snap(snap_key, reset_mem_hooks)
            else:
                self.teardown_prof(phase, step_ctx)
        if self._enabled[(phase, step_ctx)]:
            self._snap_indices[(phase, step_ctx)] += 1

    @rank_zero_only
    def dump_memory_stats(self) -> None:
        # TODO: all gather memory stats in the future if/when multiple ranks are supported
        filename = self._cuda_snapshot_dir / "memory_stats.pickle"
        with open(filename, "wb") as f:
            pickle.dump(self.memory_stats, f)


class DebugGeneration:
    """Give user-provided callbacks with the ability to connect to another user-provided callback.

    This resolution logic is provided in order to avoid callback-dependent trainer attributes (e.g.
    trainer.finetuningscheduler_callback)
    """
    # TODO:
    # - after basic functionality set, get device from config instead of hardcoding
    # - may make sense to add some additional debugging methods that parse and analyze all of the generated outputs
    #   including `output_attentions` and `output_hidden_states` etc.

    def __init__(
        self,
    ) -> None:
        """Arguments."""
        super().__init__()
        self.phandle = None

    def connect(self, obj_ref: Any) -> None:
        self.phandle = obj_ref

    def debug_sequences(self, sequences: Optional[Union[List, str]] = None) -> List:
        """_summary_

        Args:
            sequences (Optional[List], optional): _description_. Defaults to None.

        Returns:
            List: _description_

        Usage:
        ```python
        # one can use this method to probe non-chat fine-tuned LLAMA2 models (just the raw sequences, no SYS
        # or INST metadata)
        self.lm_debug.debug_generate_batch(self.no_sys_inst_debug_sequences(), max_new_tokens=25)
        ```
        """
        sequences = sequences or self.phandle.it_cfg.debug_lm_cfg.raw_debug_sequences
        if not isinstance(sequences, list):
            sequences = [sequences]
        return [f"{ex.strip()}" for ex in sequences]

    def _debug_generate(self, inputs: List|torch.Tensor, max_new_tokens_override: Optional[int] = None,
                        gen_config_override: Optional[Dict] = None) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_
            max_new_tokens_override (Optional[int], optional): _description_. Defaults to None.
            gen_config_override (Optional[Dict], optional): _description_. Defaults to None.

        Usage:

        ```python
        self.lm_debug.debug_generate_batch(['my sequence potentially with chat specific tags', 'another sequence'])
        # to narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.lm_debug.debug_generate_serial(['my sequence potentially with chat specific tags', 'another sequence'])
        # to override the defaults (both questions and current `max_new_tokens` config)
        self.lm_debug.debug_generate_batch(self.lm_debug.sys_inst_debug_sequences([
            'What is the color of a cloudless sky?', 'How many days are in a year?']), max_new_tokens=25)
        ```
        """
        # note we're not using a context manager here, keeping our new override for subsequent debugging convenience
        if max_new_tokens_override:
            self.phandle.it_cfg.zero_shot_cfg.lm_generation_cfg.max_new_tokens = max_new_tokens_override
        gen_config_dict = self.phandle.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__
        if gen_config_override:
            gen_config_dict.update(gen_config_override)
        return self.phandle.model.generate(input_ids=inputs,
                                           pad_token_id=self.phandle.trainer.datamodule.tokenizer.pad_token_id,
                                           **gen_config_dict)

    def perplexity_on_sample(self, corpus: Optional[Dataset] = None) -> float:
        if not corpus:
            corpus = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # max_length_sample_index = np.argmax(np.array([len(i) for i in corpus["text"]]))
        encoded_corpus = self.phandle.trainer.datamodule.tokenizer("\n\n".join(corpus["text"]), return_tensors="pt")
        return self.naive_perplexity(encoded_corpus)

    def top1_token_accuracy_on_sample(self, sample: str) -> Tuple[float, List[str]]:
        sample_input_ids = self.phandle.trainer.datamodule.tokenizer.encode(sample)
        sample_input_ids = torch.tensor(sample_input_ids).to("cuda:0")
        sample_input_ids = sample_input_ids.unsqueeze(0)
        with torch.no_grad():
            output = self.phandle.trainer.model(input_ids=sample_input_ids)
        logits = output['logits']
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = sample_input_ids.squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()
        correct_tokens = self.phandle.trainer.datamodule.tokenizer.batch_decode(prediction[prediction == true_tokens])
        return num_correct/len(true_tokens), correct_tokens

    def naive_perplexity(self, encoded_corpus, stride: int = 512) -> float:
        max_length = self.phandle.trainer.model.model.config.n_positions
        stride = stride
        seq_len = encoded_corpus.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encoded_corpus.input_ids[:, begin_loc:end_loc].to("cuda:0")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.phandle.trainer.model(input_ids=input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl

    def debug_generate_batch(self, sequences: List, max_new_tokens: Optional[int] = None,
                             gen_config_override: Optional[Dict] = None) -> Tuple[List, List]:
        fake_input_ids = self.phandle.trainer.datamodule.tokenizer.batch_encode_plus(sequences)
        fake_input_ids = self.phandle.trainer.datamodule.data_collator(fake_input_ids)
        fake_input_ids = fake_input_ids.to("cuda:0")
        outputs = self._debug_generate(fake_input_ids['input_ids'], max_new_tokens, gen_config_override)
        answers = self.phandle.trainer.datamodule.tokenizer.batch_decode(outputs['sequences'],
                                                                         skip_special_tokens=False,
                                                                         clean_up_tokenization_spaces=True)
        return answers, outputs

    def debug_generate_serial(self, sequences: List, max_new_tokens: Optional[int] = None,
                              gen_config_override: Optional[Dict] = None) -> Tuple[List, List]:
        answers = []
        full_outputs = []
        for seq in sequences:
            fake_input_ids = self.phandle.trainer.datamodule.tokenizer.encode(seq)
            fake_input_ids = torch.tensor(fake_input_ids).to("cuda:0")
            fake_input_ids = fake_input_ids.unsqueeze(0)
            output = self._debug_generate(fake_input_ids, max_new_tokens, gen_config_override)
            sequences = output['sequences'].unbind()
            for seq in sequences:  # in case num_return_sequences > 1
                answers.append(self.phandle.trainer.datamodule.tokenizer.decode(seq,
                                                                                skip_special_tokens=False,
                                                                                clean_up_tokenization_spaces=True))
            full_outputs.append(output)
        return answers, full_outputs
