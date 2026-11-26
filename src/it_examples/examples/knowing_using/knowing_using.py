r"""Knowing but not using: the Knowing-Using Gap's layer-wise representation self-distillation (LRSD) recipe.

Entrypoint of the ``knowing_using`` component. It reproduces part of Table 9 of Dai et al., "Towards Mechanistically
Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning"
(https://www.alphaxiv.org/abs/2607.08393): LoRA fine-tuning on 1,000 single-hop biomedical facts, with plain
cross-entropy ($\lambda = 0$) or cross-entropy plus an LRSD term ($\lambda = 1$), scored after every epoch by greedy
exact match on the two-hop (chaining) questions those facts compose into.

The LRSD term aligns a mid-layer state of the fact's head entity with a late-layer one from the same forward pass. With
$h^{\ell}$ the output of decoder block $\ell$ and $E$ the head entity's token positions in the prompt, it is
$\mathrm{relMSE}(h^{s}[E], \mathrm{sg}(h^{t}[E]))$, the mean over $E$ of $\lVert a - b \rVert^2 / \lVert b \rVert^2$
where $s$ is the student layer, $t$ the teacher layer and $\mathrm{sg}$ stops the gradient, so only the student path
learns from it. The loss is the completion cross-entropy plus $\lambda$ times the term's mean over a batch's items that
have a span.

The fact sampler, the head-entity span finder and the prompt construction reimplement the authors' MIT-licensed code
(the chaining sample: seed, shuffle, then a randomized subset-sum to exactly ``n`` memorization tasks); nothing here is
copied from it. The source data is the paper's ``chaining_tasks.json`` (derived from the STaRK benchmark over PrimeKG),
published unchanged as the Hub dataset ``speediedan/knowing-using-gap-chaining`` (MIT). The datamodule downloads it at a
pinned revision by default (:data:`KUG_DATA_REPO`, :data:`KUG_DATA_FILENAME`, :data:`KUG_DATA_REVISION`), or reads a
local copy named by ``KUGDataModuleConfig.data_path``: the sample is deterministic given that file and the fact seed.

Classes reachable by ``class_path`` (``knowing_using.<name>``): :class:`KUGPromptConfig`, :class:`KUGDataModuleConfig`,
:class:`KUGDataModule`, :class:`KUGLRSDConfig`, :class:`KUGModule` and the core-loop callback
:class:`ExactMatchReport`.
"""

import json
import random
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding

from interpretune import (
    STEP_OUTPUT,
    ChatTemplatePromptConfig,
    ITConfig,
    ITDataModule,
    ITDataModuleConfig,
    MisconfigurationException,
    rank_zero_info,
)

# the paper's chaining tasks on the Hub, unchanged, at the revision this recipe pins (the file's sha256 is
# 7ea065e4e3ddaa0f6ee100695961ddf8f9a9581938e5b6127168858718598d5d)
KUG_DATA_REPO = "speediedan/knowing-using-gap-chaining"
KUG_DATA_FILENAME = "chaining_tasks.json"
KUG_DATA_REVISION = "a1bb9cee0bb47bc956df037a8e013a4350d127db"
SYSTEM_PROMPT = "You are a biomedical assistant. Answer the question with the most appropriate entity name."
# the paper's per-model fact seeds: Qwen2.5 samples with 848 (409 unique two-hop questions), Llama 3.2 with 1340 (427)
QWEN_FACT_SEED = 848
LLAMA_FACT_SEED = 1340
# the longest entity span, in tokens, the span finder considers
MAX_SPAN_TOKENS = 40
IGNORE_INDEX = -100


# -- the sample, the spans and the prompts -------------------------------------------------------------------------


def sample_chaining_items(items: Sequence[Mapping[str, Any]], n: int = 1000, seed: int = QWEN_FACT_SEED) -> list:
    """The paper's chaining sample: chaining items whose memorization tasks number exactly ``n``.

    The items are shuffled with a generator seeded by ``seed``, then a randomized subset-sum runs over their
    memorization-task counts: every reachable total keeps one subset, replaced by a newer one with probability one half,
    until ``n`` is reachable. A local generator is used, which draws the stream ``random.seed(seed)`` would without
    reseeding the process's global one.

    Raises:
        ValueError: If no subset of the items sums to ``n`` memorization tasks.
    """
    rng = random.Random(seed)
    indexed = list(enumerate(items))
    rng.shuffle(indexed)
    combos: dict[int, list[int]] = {0: []}
    for idx, item in indexed:
        m = len(item.get("memorization_tasks", []))
        if m == 0:
            continue
        for total, picked in list(combos.items()):
            t = total + m
            if t > n:
                continue
            if t not in combos or rng.random() < 0.5:
                combos[t] = picked + [idx]
        if n in combos:
            break
    if n not in combos:
        raise ValueError(f"no subset of the {len(items)} chaining items has exactly {n} memorization tasks")
    return [items[i] for i in combos[n]]


def answer_list(answer: str | Sequence[str]) -> list[str]:
    """An answer field (a string, or a list of accepted strings) as a list."""
    return [answer] if isinstance(answer, str) else list(answer)


def find_entity_span(tokenizer: Any, ids: Sequence[int], entity: str, max_len: int = MAX_SPAN_TOKENS) -> list | None:
    """The token positions of ``entity`` in ``ids``: the shortest window whose decoded text, stripped and
    lowercased, equals the entity's, or ``None`` when there is none.

    A window stops growing once its text is more than four characters longer than the entity's.
    """
    target, best = entity.strip().lower(), None
    for i in range(len(ids)):
        for j in range(i + 1, min(i + max_len, len(ids)) + 1):
            text = tokenizer.decode(ids[i:j]).strip().lower()
            if text == target:
                if best is None or (j - i) < (best[1] - best[0]):
                    best = (i, j)
            elif len(text) > len(target) + 4:
                break
    return list(range(*best)) if best else None


def is_exact_match(prediction: str, golds: Sequence[str]) -> bool:
    """Whether ``prediction`` equals any gold answer as a whole string, after stripping and lowercasing both."""
    pred = prediction.strip().lower()
    return any(gold.strip().lower() == pred for gold in golds)


@dataclass(kw_only=True)
class KUGPromptConfig(ChatTemplatePromptConfig):
    """The paper's prompts: a system message and the question as the user turn, through the tokenizer's chat
    template with the generation prompt appended; the training target is the assistant turn.

    ``chat_template_kwargs`` are passed to every ``apply_chat_template`` call (a Llama 3 template's ``date_string``, for
    one, which otherwise stamps the current date into every prompt).
    """

    system_prompt: str = SYSTEM_PROMPT
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def build_messages(self, task_prompt: str) -> list[dict[str, str]]:
        """The system message, then the question as the user turn (unstripped, as the paper's code sends it)."""
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": task_prompt}]

    def _bound_tokenizer(self) -> Any:
        tokenizer = self._tokenizer
        if tokenizer is None or not getattr(tokenizer, "chat_template", None):
            raise MisconfigurationException(
                "KUGPromptConfig builds every prompt with the tokenizer's chat template, but no tokenizer with one is"
                " bound (`bind_tokenizer`): the recipe's prompts and its assistant-turn targets need the template."
            )
        return tokenizer

    def prompt_text(self, question: str) -> str:
        """The prompt for ``question``, ending with the generation prompt."""
        return self._bound_tokenizer().apply_chat_template(
            self.build_messages(question), tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
        )

    def qa_ids(self, question: str, answer: str) -> tuple[list[int], list[int]]:
        """The prompt's token ids and the completion's: the answer plus the template's end-of-turn tokens.

        Raises:
            ValueError: If the templated conversation does not extend the prompt, as text and as tokens: the
                completion would then not be the assistant turn alone.
        """
        tokenizer = self._bound_tokenizer()
        prompt = self.prompt_text(question)
        messages = self.build_messages(question) + [{"role": "assistant", "content": answer}]
        full = tokenizer.apply_chat_template(messages, tokenize=False, **self.chat_template_kwargs)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full, add_special_tokens=False).input_ids
        if not full.startswith(prompt) or full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"the chat template's assistant turn does not extend the prompt for {question!r}")
        return prompt_ids, full_ids[len(prompt_ids) :]


def build_kug_splits(
    sampled: Sequence[Mapping[str, Any]], tokenizer: Any, prompt_cfg: KUGPromptConfig
) -> tuple[list[dict], list[dict]]:
    """The training rows and the evaluation rows for a chaining sample.

    One training row per memorization task (its fact gives the head entity): ``input_ids`` (prompt then completion),
    ``prompt_len`` and ``span`` (the head entity's positions in the prompt, empty when it cannot be found). One
    evaluation row per unique question: the two-hop questions (``kind`` ``twohop``) then the single-hop ones (``mem``),
    each with its prompt's ``input_ids`` and every gold answer the sample gives it.
    """
    train: list[dict] = []
    mem_gold: dict[str, set[str]] = {}
    twohop_gold: dict[str, set[str]] = {}
    for item in sampled:
        for task, fact in zip(item["memorization_tasks"], item["facts"]):
            question, answers = task["prompt"], answer_list(task["answer"])
            prompt_ids, completion_ids = prompt_cfg.qa_ids(question, answers[0])
            span = find_entity_span(tokenizer, prompt_ids, fact["head"])
            train.append({"input_ids": prompt_ids + completion_ids, "prompt_len": len(prompt_ids), "span": span or []})
            mem_gold.setdefault(question, set()).update(answers)
        task = item["generalization_tasks"][0]
        twohop_gold.setdefault(task["prompt"], set()).update(answer_list(task["answer"]))
    evaluation = [
        {
            "input_ids": tokenizer(prompt_cfg.prompt_text(question), add_special_tokens=False).input_ids,
            "gold": sorted(golds),
            "kind": kind,
        }
        for kind, gold_map in (("twohop", twohop_gold), ("mem", mem_gold))
        for question, golds in gold_map.items()
    ]
    return train, evaluation


_TRAIN_FEATURES = datasets.Features(
    {
        "input_ids": datasets.Sequence(datasets.Value("int64")),
        "prompt_len": datasets.Value("int64"),
        "span": datasets.Sequence(datasets.Value("int64")),
    }
)
_EVAL_FEATURES = datasets.Features(
    {
        "input_ids": datasets.Sequence(datasets.Value("int64")),
        "gold": datasets.Sequence(datasets.Value("string")),
        "kind": datasets.Value("string"),
    }
)


# -- the datamodule ------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class KUGDataModuleConfig(ITDataModuleConfig):
    """The datamodule configuration, with the sample's source and size.

    Args:
        data_repo_id: The Hub dataset holding the paper's chaining tasks.
        data_filename: The chaining-tasks file in that dataset (each item has ``facts``, ``memorization_tasks`` and
            ``generalization_tasks``).
        data_revision: The dataset revision the file is downloaded at; the default pins the published commit.
        data_path: A local copy of the chaining-tasks file. When set it is read instead, and nothing is downloaded.
        fact_seed: The sample's seed (:data:`QWEN_FACT_SEED` for Qwen2.5, :data:`LLAMA_FACT_SEED` for Llama 3.2).
        n_memorization: How many memorization tasks (single-hop facts) the sample holds.
    """

    data_repo_id: str = KUG_DATA_REPO
    data_filename: str = KUG_DATA_FILENAME
    data_revision: str = KUG_DATA_REVISION
    data_path: str | None = None
    fact_seed: int = QWEN_FACT_SEED
    n_memorization: int = 1000

    def chaining_tasks_path(self) -> Path:
        """The chaining-tasks file: ``data_path`` when set, otherwise the Hub dataset's file, downloaded (or read
        from the Hugging Face cache) at ``data_revision``."""
        if self.data_path is not None:
            return Path(self.data_path)
        return Path(
            hf_hub_download(self.data_repo_id, self.data_filename, repo_type="dataset", revision=self.data_revision)
        )


class KUGDataModule(ITDataModule):
    """The chaining sample as a ``train`` split (one row per single-hop fact) and a ``validation`` split (the
    unique two-hop and single-hop questions, scored by exact match).

    Training batches are right-padded, so each item's positions (and its head-entity span) start at zero; evaluation
    batches are left-padded for generation.
    """

    itdm_cfg: KUGDataModuleConfig

    def __init__(self, itdm_cfg: KUGDataModuleConfig) -> None:
        super().__init__(itdm_cfg=itdm_cfg)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self, target_model: torch.nn.Module | None = None) -> None:
        """Sample the facts, build both splits and save them to ``dataset_path``."""
        cfg = self.itdm_cfg
        if not isinstance(cfg, KUGDataModuleConfig):
            raise MisconfigurationException(
                "KUGDataModule needs a KUGDataModuleConfig (the chaining tasks' source and the sample's size), not a"
                f" {type(cfg).__name__}."
            )
        prompt_cfg = cfg.prompt_cfg
        if not isinstance(prompt_cfg, KUGPromptConfig):
            raise MisconfigurationException(
                f"KUGDataModule builds its prompts with a KUGPromptConfig, not a {type(prompt_cfg).__name__}."
            )
        prompt_cfg.bind_tokenizer(self.tokenizer)
        items = json.loads(cfg.chaining_tasks_path().read_text(encoding="utf-8"))
        sampled = sample_chaining_items(items, cfg.n_memorization, cfg.fact_seed)
        train, evaluation = build_kug_splits(sampled, self.tokenizer, prompt_cfg)
        counts = {kind: sum(row["kind"] == kind for row in evaluation) for kind in ("twohop", "mem")}
        rank_zero_info(
            f"Knowing-Using sample (fact seed {cfg.fact_seed}): {len(sampled)} chaining items, {len(train)} training"
            f" facts ({sum(not row['span'] for row in train)} without a head-entity span), {counts['twohop']} unique"
            f" two-hop and {counts['mem']} unique single-hop questions."
        )
        dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_list(train, features=_TRAIN_FEATURES),
                "validation": datasets.Dataset.from_list(evaluation, features=_EVAL_FEATURES),
            }
        )
        self.save_prepared_dataset(dataset)

    def _pad_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        assert isinstance(pad_id, int), "the tokenizer needs a pad token (set from its eos token when it has none)"
        return pad_id

    def collate_train(self, rows: Sequence[Mapping[str, Any]]) -> BatchEncoding:
        """Right-padded ``input_ids`` and ``attention_mask``, ``labels`` that keep only the completion tokens, and
        each item's head-entity ``spans`` (a list per item, empty when it has none)."""
        pad_id = self._pad_id()
        width = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for r, row in enumerate(rows):
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
            input_ids[r, : len(ids)] = ids
            attention_mask[r, : len(ids)] = 1
            labels[r, row["prompt_len"] : len(ids)] = ids[row["prompt_len"] :]
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "spans": [list(row["span"]) for row in rows],
            }
        )

    def collate_eval(self, rows: Sequence[Mapping[str, Any]]) -> BatchEncoding:
        """Left-padded prompts, each question's gold answers and its kind (``twohop`` or ``mem``)."""
        pad_id = self._pad_id()
        width = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for r, row in enumerate(rows):
            n = len(row["input_ids"])
            input_ids[r, width - n :] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask[r, width - n :] = 1
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "gold": [list(row["gold"]) for row in rows],
                "kind": [row["kind"] for row in rows],
            }
        )

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        train = split == "train"
        kwargs = {**self.itdm_cfg.dataloader_kwargs, "shuffle": shuffle, "drop_last": False}
        dataset: Any = self.dataset[split]  # a `datasets.Dataset`, which DataLoader indexes as a map-style dataset
        return DataLoader(
            dataset,
            batch_size=self.itdm_cfg.train_batch_size if train else self.itdm_cfg.eval_batch_size,
            collate_fn=self.collate_train if train else self.collate_eval,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("validation", shuffle=False)


# -- the objective and the evaluation ------------------------------------------------------------------------------


@dataclass(kw_only=True)
class KUGLRSDSettings:
    """The LRSD settings of the module configuration (:class:`KUGLRSDConfig`).

    Args:
        lrsd_lambda: The LRSD term's weight; ``0`` trains on cross-entropy alone.
        lrsd_student_layer: The decoder block whose head-entity state is trained toward the teacher's.
        lrsd_teacher_layer: The deeper decoder block whose head-entity state is the (stop-gradient) target.
        grad_clip_norm: The total gradient norm the trained parameters are clipped to before each optimizer step on
            the core loop, or ``None`` for no clipping.
    """

    lrsd_lambda: float = 1.0
    lrsd_student_layer: int = 14
    lrsd_teacher_layer: int = 21
    grad_clip_norm: float | None = 1.0


@dataclass(kw_only=True)
class KUGLRSDConfig(KUGLRSDSettings, ITConfig):
    """The module configuration: :class:`~interpretune.config.module.ITConfig` with the LRSD settings."""


def relative_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    r"""The mean over positions of $\lVert a - b \rVert^2 / \lVert b \rVert^2$, in fp32, the denominator clamped at
    ``1e-6``. Stopping the teacher's gradient is the caller's choice."""
    a, b = student.float(), teacher.float()
    return (((a - b) ** 2).sum(-1) / (b**2).sum(-1).clamp_min(1e-6)).mean()


def lrsd_term(
    hidden_states: Sequence[torch.Tensor], spans: Sequence[Sequence[int]], student_layer: int, teacher_layer: int
) -> torch.Tensor | None:
    """The LRSD term: the mean over the items with a span of the relative MSE between the student block's output and
    the teacher block's, detached, at the item's head-entity positions; ``None`` when no item has a span.

    ``hidden_states`` is a Hugging Face model's ``output_hidden_states`` tuple, whose entry ``l + 1`` is the output of
    decoder block ``l``.
    """
    student, teacher = hidden_states[student_layer + 1], hidden_states[teacher_layer + 1]
    terms = [
        relative_mse(student[r, list(span)], teacher[r, list(span)].detach()) for r, span in enumerate(spans) if span
    ]
    return torch.stack(terms).mean() if terms else None


def completion_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy in fp32, averaged over the tokens ``labels`` keeps (the completions)."""
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX
    )


def clip_optimizer_grads(optimizer: torch.optim.Optimizer, max_norm: float) -> None:
    """Clip the total gradient norm of ``optimizer``'s parameters to ``max_norm``."""
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def _autocast_state(device_type: str) -> str:
    """The autocast in force for ``device_type``: its dtype's name, or ``none``."""
    if not torch.is_autocast_enabled(device_type):
        return "none"
    return str(torch.get_autocast_dtype(device_type)).removeprefix("torch.")


class ExactMatchTally:
    """Correct and total counts of exact-match answers per question kind, summed over batches.

    The rate is the ratio of the sums, not a mean of per-batch rates, which would weight a short final batch as much as
    a full one.
    """

    def __init__(self) -> None:
        self.counts: dict[str, list[int]] = {}
        self.autocast: str | None = None

    def update(self, predictions: Sequence[str], golds: Sequence[Sequence[str]], kinds: Sequence[str]) -> None:
        """Count one batch: each prediction against its question's gold answers, under its question's kind."""
        for prediction, gold, kind in zip(predictions, golds, kinds, strict=True):
            counts = self.counts.setdefault(kind, [0, 0])
            counts[0] += is_exact_match(prediction, gold)
            counts[1] += 1

    def metrics(self) -> dict[str, float]:
        """``<kind>_em``, ``<kind>_correct`` and ``<kind>_total`` for every kind counted."""
        out: dict[str, float] = {}
        for kind, (correct, total) in sorted(self.counts.items()):
            out.update({f"{kind}_em": correct / total, f"{kind}_correct": correct, f"{kind}_total": total})
        return out

    def reset(self) -> None:
        """Forget every count (and the autocast state recorded with them)."""
        self.counts.clear()
        self.autocast = None


class KUGSteps:
    r"""The recipe's steps: completion cross-entropy plus $\lambda$ times the LRSD term in training, greedy
    generation tallied by exact match in validation.

    Training logs ``train_ce``, ``train_lrsd`` (when $\lambda \gt 0$ and a batch has a span) and ``train_loss``.
    Validation logs nothing: it adds each batch to ``em_tally``, whose totals :class:`ExactMatchReport` reports once per
    evaluation, because ``log()`` averages per batch.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.em_tally = ExactMatchTally()
        self._clipped_optimizers: weakref.WeakSet = weakref.WeakSet()

    def _before_it_cfg_init(self, it_cfg: Any) -> Any:
        it_cfg = super()._before_it_cfg_init(it_cfg)  # pyright: ignore[reportAttributeAccessIssue]  # cooperative MRO
        it_cfg.num_labels = 0  # a generative LM head, not a classification head
        return it_cfg

    def setup(self, *args: Any, **kwargs: Any) -> None:
        super().setup(*args, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]  # cooperative MRO call
        self._check_lrsd_layers()

    def _check_lrsd_layers(self) -> None:
        """Refuse a layer pair that is not a shallower student and a deeper teacher whose outputs the model's
        ``hidden_states`` hold unnormalized (the last entry is the final norm's output, not the last block's)."""
        cfg = self.it_cfg  # pyright: ignore[reportAttributeAccessIssue]  # the composed module provides it_cfg
        n_layers = self.model.config.num_hidden_layers  # pyright: ignore[reportAttributeAccessIssue]
        student, teacher = cfg.lrsd_student_layer, cfg.lrsd_teacher_layer
        if not 0 <= student < teacher <= n_layers - 2:
            raise MisconfigurationException(
                f"The LRSD layer pair (student {student}, teacher {teacher}) must satisfy 0 <= student < teacher <="
                f" {n_layers - 2} for this {n_layers}-block model: the student is the shallower block, and the last"
                " block's output is reported only after the final norm."
            )

    def _install_grad_clip(self) -> None:
        """Clip before every step of each optimizer the module holds (a callback may replace it), once per
        optimizer."""
        max_norm = self.it_cfg.grad_clip_norm  # pyright: ignore[reportAttributeAccessIssue]
        if max_norm is None:
            return
        for optimizer in self.optimizers or ():  # pyright: ignore[reportAttributeAccessIssue]
            if optimizer not in self._clipped_optimizers:
                optimizer.register_step_pre_hook(lambda opt, args, kwargs: clip_optimizer_grads(opt, max_norm))
                self._clipped_optimizers.add(optimizer)

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        self._install_grad_clip()
        cfg = self.it_cfg  # pyright: ignore[reportAttributeAccessIssue]
        spans, labels = batch.pop("spans"), batch.pop("labels")
        lam = float(cfg.lrsd_lambda)
        out = self(**batch, output_hidden_states=lam > 0)  # pyright: ignore[reportCallIssue]
        ce = completion_cross_entropy(out.logits, labels)
        loss = ce
        self.log("train_ce", ce.detach())  # pyright: ignore[reportAttributeAccessIssue]
        if lam > 0:
            lrsd = lrsd_term(out.hidden_states, spans, cfg.lrsd_student_layer, cfg.lrsd_teacher_layer)
            if lrsd is not None:
                loss = loss + lam * lrsd
                self.log("train_lrsd", lrsd.detach())  # pyright: ignore[reportAttributeAccessIssue]
        self.log("train_loss", loss.detach())  # pyright: ignore[reportAttributeAccessIssue]
        return loss

    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT | None:
        golds, kinds = batch.pop("gold"), batch.pop("kind")
        tokenizer = self.datamodule.tokenizer  # pyright: ignore[reportAttributeAccessIssue]
        input_ids = batch["input_ids"]
        if self.em_tally.autocast is None:
            self.em_tally.autocast = _autocast_state(input_ids.device.type)
        generate = self.it_generate  # pyright: ignore[reportAttributeAccessIssue]  # the composed module provides it
        generated = generate(batch, pad_token_id=tokenizer.pad_token_id)
        sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
        predictions = tokenizer.batch_decode(
            sequences[:, input_ids.shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        self.em_tally.update(predictions, golds, kinds)
        return None


class KUGModule(KUGSteps, torch.nn.Module): ...


# -- phases as experimental stages ---------------------------------------------------------------------------------


class MetricThresholdTransition:
    """A finetuning-scheduler transition policy that opens the next phase once an epoch metric reaches a threshold.

    With :class:`ExactMatchReport` before ``FTSCoreCallback`` in the run's callbacks, ``monitor: mem_em`` opens the
    next phase in the epoch memorization first reaches ``threshold``. It never ends the final phase, and an epoch
    without the metric continues the phase.

    Args:
        monitor: The epoch metric watched.
        threshold: The value that opens the next phase.
        mode: ``"max"`` advances once the metric is at least ``threshold``, ``"min"`` once it is at most.
    """

    granularity = frozenset({"epoch"})

    def __init__(self, monitor: str = "mem_em", threshold: float = 0.9, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise MisconfigurationException(f"`mode` must be 'max' or 'min', got {mode!r}.")
        self.monitor, self.threshold, self.mode = monitor, float(threshold), mode

    def evaluate(self, ctx: Any) -> Any:
        from finetuning_scheduler.core.transitions import Decision

        if ctx.is_final_phase or (value := ctx.metrics.get(self.monitor)) is None:
            return Decision.CONTINUE
        value = float(value)
        if value >= self.threshold if self.mode == "max" else value <= self.threshold:
            return Decision.ADVANCE, f"{self.monitor} {value:.4f} reached {self.threshold}"
        return Decision.CONTINUE

    def reset(self, depth: int) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {
            "class_path": f"{type(self).__module__}.{type(self).__qualname__}",
            "init_args": {"monitor": self.monitor, "threshold": self.threshold, "mode": self.mode},
        }


class PhaseLRSD:
    """A finetuning-scheduler phase-start hook that sets the module's LRSD settings from the phase's
    ``phase_config``.

    A phase's ``phase_config`` may name any :class:`KUGLRSDSettings` field (``lrsd_lambda: 1.0`` turns the term on);
    a phase without one keeps the settings it inherits. Configured as ``phase_hooks: {start: [{class_path:
    knowing_using.PhaseLRSD}]}`` in ``FTSCoreCallback``'s ``tuning_scheduler_kwargs``.
    """

    def __call__(self, event: Any) -> None:
        settings = dict(event.phase_config or {})
        if unknown := sorted(set(settings) - set(KUGLRSDSettings.__dataclass_fields__)):
            raise MisconfigurationException(
                f"phase {event.depth}'s phase_config names unknown LRSD settings {unknown}."
            )
        it_cfg = event.tuning_scheduler.model.it_cfg
        for name, value in settings.items():
            setattr(it_cfg, name, value)


# -- reporting exact match on the core loop ------------------------------------------------------------------------


class ExactMatchReport:
    """A core-loop callback that reports the module's exact-match totals after every epoch's validation, and runs
    one evaluation before training.

    The epoch's metrics gain ``twohop_em``, ``twohop_correct``, ``twohop_total`` and the same for ``mem`` (so a later
    callback, a checkpoint policy say, can read them), which the callback prints. Every evaluation is kept in
    ``history`` as ``{"epoch": <training epochs completed>, **metrics}`` (``epoch`` 0 before training), saved with the
    loop's checkpoints and, with ``log_path``, rewritten there as JSON after each evaluation.

    Args:
        precision: The autocast precision the evaluation before training runs under. Set it to the run's
            ``run_cfg.precision``, which the loop's own validation runs under: the callback refuses the run, after its
            first validation, if the two differ.
        evaluate_before_training: Whether to evaluate the untrained model.
        log_path: A JSON file the history is written to.
    """

    def __init__(
        self, precision: str | None = None, evaluate_before_training: bool = True, log_path: str | None = None
    ) -> None:
        self.precision = precision
        self.evaluate_before_training = evaluate_before_training
        self.log_path = log_path
        self.history: list[dict[str, Any]] = []
        self._first_autocast: str | None = None

    def on_train_start(self, ctx: Any) -> None:
        if ctx.dist_run is not None:
            raise MisconfigurationException(
                "ExactMatchReport counts one process's answers: under a data-parallel placement each rank would report"
                " its own share. Run the evaluation in a single process."
            )
        if self.evaluate_before_training:
            self._evaluate(ctx.module)
            self._report(ctx.module, 0)

    def on_validation_end(self, ctx: Any, metrics: Mapping[str, float]) -> None:
        ctx.epoch_metrics.update(self._report(ctx.module, ctx.epoch + 1))

    def _evaluate(self, module: Any) -> None:
        from interpretune.runners.core import run_step

        module.model.eval()
        try:
            for batch_idx, batch in enumerate(module.datamodule.val_dataloader()):
                with torch.inference_mode():
                    run_step(
                        step_fn="validation_step",
                        module=module,
                        batch=batch,
                        batch_idx=batch_idx,
                        precision=self.precision,  # pyright: ignore[reportArgumentType]  # checked by run_step
                    )
        finally:
            module.model.train()

    def _report(self, module: Any, epoch: int) -> dict[str, float]:
        tally: ExactMatchTally = module.em_tally
        metrics = tally.metrics()
        if self._first_autocast is None:
            self._first_autocast = tally.autocast
        elif tally.autocast is not None and tally.autocast != self._first_autocast:
            raise MisconfigurationException(
                f"The evaluation before training ran under autocast '{self._first_autocast}' and the loop's validation"
                f" under '{tally.autocast}', so their exact-match scores are not comparable: set ExactMatchReport's"
                " `precision` to the run's `run_cfg.precision`."
            )
        tally.reset()
        if not metrics:
            return {}
        self.history.append({"epoch": epoch, **metrics})
        summary = ", ".join(f"{k}: {v:.4f}" if k.endswith("_em") else f"{k}: {int(v)}" for k, v in metrics.items())
        print(f"\nExact match after {epoch} epoch(s): {{{summary}}}")
        if self.log_path is not None:
            Path(self.log_path).write_text(json.dumps(self.history, indent=1), encoding="utf-8")
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {"history": list(self.history), "first_autocast": self._first_autocast}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.history = list(state.get("history", []))
        self._first_autocast = state.get("first_autocast")


__all__ = [
    "KUG_DATA_FILENAME",
    "KUG_DATA_REPO",
    "KUG_DATA_REVISION",
    "ExactMatchReport",
    "ExactMatchTally",
    "KUGDataModule",
    "KUGDataModuleConfig",
    "KUGLRSDConfig",
    "KUGLRSDSettings",
    "KUGModule",
    "KUGPromptConfig",
    "KUGSteps",
    "MetricThresholdTransition",
    "PhaseLRSD",
    "build_kug_splits",
    "clip_optimizer_grads",
    "completion_cross_entropy",
    "find_entity_span",
    "is_exact_match",
    "lrsd_term",
    "relative_mse",
    "sample_chaining_items",
]
