# Tuning: the Knowing-Using Gap (LRSD)

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view. This page
describes a research example built on it: the `speediedan/knowing-using` component, which reproduces the layer-wise
representation self-distillation (LRSD) result of Dai et al.,
[Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning](https://www.alphaxiv.org/abs/2607.08393),
on Qwen2.5-1.5B-Instruct, as a one-phase LoRA run on the core loop.

```bash
interpretune --run_command train --config hub://speediedan/knowing-using/runs/kug_lrsd.qwen2_5_1_5b.core.yaml
```

## The Knowing-Using Gap

Fine-tune a model on single-hop facts, such as "Which gene or protein is expressed in optic choroid?" (GLUL) and "Which
drug targets the gene or protein GLUL?" (L-Glutamine), and it learns to answer them. Ask the two-hop question they
compose into, "Which drug targets the genes or proteins that are expressed in optic choroid?", and it mostly fails. The
model knows the facts but does not use them: the paper calls this the Knowing-Using Gap. Its remedy, LRSD, adds a
training term that pulls a middle layer's representation of each fact's head entity toward a later layer's
representation of the same entity in the same forward pass.

## What this example reproduces

The Qwen2.5-1.5B row of the paper's Table 9: plain fine-tuning against LRSD with $\lambda = 1$ on the layer pair 21 to
14 (block 14's state is trained toward block 21's), three training seeds each. Over epochs 30 to 50 the paper reports
two-hop exact match of 0.085 for plain fine-tuning and 0.166 with $\lambda = 1$, at memorization exact match of 0.985
and 0.987.

The paper's main result applies the LRSD term from the start of training, and so does this example. The authors' code
also offers switching the term on at a fixed later epoch (its `--start_epoch` option, 0 by default), an ablation this
example does not reproduce.

## The recipe

| setting | value |
| --- | --- |
| model | Qwen2.5-1.5B-Instruct (28 decoder blocks), at a pinned revision |
| adapters | LoRA with rank 8, alpha 32 and dropout 0.05 on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj` of every block; nothing else trains |
| optimizer | AdamW, learning rate $10^{-4}$, weight decay 0.01 |
| gradient clipping | total norm 1 |
| batch | 10 facts |
| epochs | 50 |
| precision | BF16 autocast over FP32 weights |
| evaluation | greedy decoding of up to 64 tokens, exact match after stripping and lowercasing, before training and after every epoch |

Each fact is a chat: the paper's system prompt, the question as the user turn and the answer as the assistant turn. With
$h_i^{(\ell)}$ the output of decoder block $\ell$ (counting from 0) at prompt position $i$, $E$ the positions of the
fact's head entity in the prompt and $\mathrm{sg}$ a stop-gradient, the loss is

$$
\mathcal{L} = \mathrm{CE} + \lambda \frac{1}{|E|} \sum_{i \in E} \frac{\lVert h_i^{(14)} - \mathrm{sg}(h_i^{(21)}) \rVert^2}{\lVert \mathrm{sg}(h_i^{(21)}) \rVert^2}
$$

where $\mathrm{CE}$ is the next-token cross-entropy over the assistant turn alone. The second term, the relative squared
error between the two residual states over the head-entity span, is averaged over a batch's facts whose head entity was
found in the prompt, and $\lambda = 0$ is plain fine-tuning. Because block 21's state is a stopped target, only blocks 0
to 14 learn from the term.

## How interpretune expresses it

- **The module.** `KUGModule` composes the recipe's steps (`KUGSteps`) over an interpretune module, and its
  configuration, `KUGLRSDConfig`, mixes the LRSD settings (`lrsd_lambda`, `lrsd_student_layer`, `lrsd_teacher_layer`,
  `grad_clip_norm`) into `ITConfig`. The training step reads the Hugging Face model's hidden states and adds the term;
  the module refuses a layer pair that is not a shallower student and a deeper teacher below the last block, whose
  output the hidden states report only after the final norm. The LoRA adapters come from
  `hf_from_pretrained_cfg.lora_cfg`.
- **The schedule.** `FTSCoreCallback` drives a one-phase Fine-Tuning Scheduler schedule
  (`ft_schedules/kug_lrsd.qwen2_5_1_5b.yaml`) that thaws the LoRA tensors and leaves every other parameter frozen, for
  the run's 50 epochs.
- **The evaluation.** `ExactMatchReport`, a core-loop callback, scores the untrained model and then the model after
  every epoch. It adds `twohop_em` and `mem_em` (with their correct and total counts) to the epoch's metrics, and writes
  the history to a JSON file (`log_path`). It counts over the whole evaluation set rather than averaging per batch, and
  refuses a run whose evaluation before training would use a different precision from the loop's own validation. No
  checkpoint is selected: the questions scored are the paper's test questions.

## The data

The paper's chaining tasks are published unchanged as the Hub dataset
[`speediedan/knowing-using-gap-chaining`](https://huggingface.co/datasets/speediedan/knowing-using-gap-chaining) (MIT
License, derived from the STaRK benchmark's STaRK-Prime knowledge base, which builds on PrimeKG). Its
`chaining_tasks.json` holds 11,997 items, each two single-hop facts with a question for each and the two-hop question
that composes them. The datamodule downloads the file at the pinned revision `a1bb9cee0bb47bc956df037a8e013a4350d127db`
(sha256 `7ea065e4e3ddaa0f6ee100695961ddf8f9a9581938e5b6127168858718598d5d`), or reads a local copy named by the
datamodule's `data_path`.

From it the datamodule draws the paper's 1,000-fact sample: fact seed 848 (the paper's seed for Qwen models), a shuffle,
then a randomized subset sum to exactly 1,000 single-hop facts. That gives 500 items, 409 unique two-hop questions and
771 unique single-hop questions; each evaluation scores all 1,180.

## Running it

The run configuration trains the $\lambda = 1$ arm with training seed 848. The paper's seeds are 848, 1 and 2, and
`lrsd_lambda` set to 0 gives the plain arm:

```bash
CONFIG=hub://speediedan/knowing-using/runs/kug_lrsd.qwen2_5_1_5b.core.yaml

interpretune --run_command train --config "${CONFIG}" --seed_everything=1

interpretune --run_command train --config "${CONFIG}" --seed_everything=1 \
  --registered_cfg '{"module_cfg": {"init_args": {"lrsd_lambda": 0.0}}}'

# a local copy of the data
interpretune --run_command train --config "${CONFIG}" \
  --registered_cfg '{"datamodule_cfg": {"init_args": {"data_path": "/path/to/chaining_tasks.json"}}}'
```

Each run writes its exact-match history to `kug_lrsd.qwen2_5_1_5b.core.exact_match.json` in the working directory, so
run each arm and seed in a directory of its own.

On one RTX 4090, an epoch of training takes about 26 seconds and an evaluation about 20, so a 50-epoch run with its 51
evaluations takes about 38 minutes, at a peak of 15.7 GiB of GPU memory. Both arms over three seeds take about four
hours.

`tests/core/test_knowing_using.py` runs the shipped run configuration end to end on a CPU for two epochs, with a tiny
random Qwen2 model and synthetic data.

## Results

Both arms, three training seeds each (848, 1 and 2), 50 epochs on one RTX 4090 through this example's run
configuration. Exact match is averaged over epochs 30 to 50, then over seeds:

| arm | two-hop, this example | two-hop, paper | memorization, this example | memorization, paper |
| --- | ---: | ---: | ---: | ---: |
| plain | 0.078 | 0.085 | 0.990 | 0.985 |
| $\lambda = 1$ | 0.159 | 0.166 | 0.975 | 0.987 |

Per seed, two-hop exact match was 0.089, 0.065 and 0.080 for plain fine-tuning and 0.153, 0.165 and 0.160 with
$\lambda = 1$, so LRSD gained 0.064, 0.100 and 0.080: a mean of 0.081, the paper's own difference. Both arms memorize the
facts almost completely, and LRSD roughly doubles how often the model composes them. The scores differ from the
paper's by less than a seed's spread. Memorization here is scored once per unique single-hop question (771), where the
paper scores every evaluation row, which may account for its small differences.
