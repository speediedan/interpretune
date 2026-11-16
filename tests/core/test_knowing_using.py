"""The ``knowing_using`` component (the Knowing-Using Gap's LRSD recipe), on CPU with no network.

A tiny randomly initialised Qwen2 model and a word-level tokenizer with a ChatML-style template stand in for
Qwen2.5-1.5B-Instruct, and a synthetic chaining-tasks file for the paper's. The unit tests cover the fact sampler, the
data's source (the pinned Hub download, with the download function replaced, and the local override), the head-entity
span finder and the splits built from them, the LRSD term's gradient path (the teacher's state is detached, so only the
blocks feeding the student learn from it), exact-match tallying over batches that do not divide the questions, and
gradient clipping. The end-to-end test runs the shipped run configuration through the CLI for two epochs on the tiny
model.
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

pytest.importorskip("peft")
pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)

KUG_DIR = Path(__file__).resolve().parents[2] / "src" / "it_examples" / "examples" / "knowing_using"
KUG_REPO = "speediedan/knowing-using"
RUN_KEY = "kug_lrsd.qwen2_5_1_5b.core"
CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[UNK]"]
N_LAYERS = 4


@pytest.fixture(scope="module")
def kug_revision():
    """The in-tree component, published locally (with its in-tree entrypoint) into the components cache."""
    from interpretune.hub.components import local_publish

    return local_publish(KUG_DIR, KUG_REPO, entrypoint_src=KUG_DIR / "knowing_using.py")


@pytest.fixture(scope="module")
def kug(kug_revision):
    """The published entrypoint, imported as the loader imports it."""
    from interpretune.hub.entrypoints import import_snapshot_entrypoint

    return import_snapshot_entrypoint(KUG_REPO, "knowing_using.py", revision=kug_revision, what="the test entrypoint")


# -- synthetic data ------------------------------------------------------------------------------------------------

ANATOMY = ["optic choroid", "duodenum", "renal cortex", "liver lobule", "spinal cord", "left atrium", "skin", "colon"]
GENES = ["GLUL", "NT5C", "ACE", "TP53", "EGFR", "BRCA1", "MYC", "KRAS"]
DRUGS = ["glutamine", "aspirin", "lisinopril", "imatinib", "gefitinib", "olaparib", "metformin", "sotorasib"]


def chaining_items() -> list[dict]:
    """Eight two-fact chaining items in the paper's shape.

    Items 0 and 1 share a two-hop question (so its gold answers are the union of theirs), and item 7's first question
    does not mention its head entity (so it has no span).
    """
    items = []
    for i, (organ, gene, drug) in enumerate(zip(ANATOMY, GENES, DRUGS)):
        q1 = (
            f"Which gene or protein is expressed in {organ}?" if i != 7 else "Which gene or protein is expressed there?"
        )
        twohop_organ = ANATOMY[0] if i == 1 else organ
        items.append(
            {
                "facts": [
                    {"head": organ, "relation": "expression present", "tail": gene},
                    {"head": gene, "relation": "target", "tail": drug},
                ],
                "memorization_tasks": [
                    {"prompt": q1, "answer": gene},
                    {"prompt": f"Which drug targets the gene or protein {gene}?", "answer": drug},
                ],
                "generalization_tasks": [
                    {"prompt": f"Which drug targets the genes or proteins expressed in {twohop_organ}?", "answer": drug}
                ],
            }
        )
    return items


def tiny_tokenizer(items: list[dict]) -> Any:
    """A word-level tokenizer over the items' text and the system prompt, with a ChatML-style chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = ["system user assistant", "You are a biomedical assistant. Answer the question with the most appropriate"]
    corpus.append("entity name.")
    for item in items:
        for task in item["memorization_tasks"] + item["generalization_tasks"]:
            corpus += [task["prompt"], task["answer"]]
        corpus += [fact["head"] for fact in item["facts"]]
    backend = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=SPECIALS))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        chat_template=CHATML,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
    )


def tiny_model(vocab_size: int, tokenizer: Any = None, n_layers: int = N_LAYERS) -> Any:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    ids = {} if tokenizer is None else {"eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        **ids,
    )
    return Qwen2ForCausalLM(cfg)


# -- the sample ----------------------------------------------------------------------------------------------------


def _paper_sample(items, n, seed):
    """The sampling procedure as the paper's code states it, on the global generator: the reference the
    entrypoint's local-generator version must reproduce."""
    random.seed(seed)
    indexed = list(enumerate(items))
    random.shuffle(indexed)
    combos = {0: []}
    for idx, item in indexed:
        m = len(item.get("memorization_tasks", []))
        if m == 0:
            continue
        for s, picked in list(combos.items()):
            t = s + m
            if t > n:
                continue
            cand = picked + [idx]
            if t not in combos or random.random() < 0.5:
                combos[t] = cand
        if n in combos:
            break
    return [items[i] for i in combos[n]]


def _varied_items(count: int = 40) -> list[dict]:
    rng = random.Random(7)
    return [{"id": i, "memorization_tasks": [{}] * rng.choice([0, 1, 2, 3])} for i in range(count)]


@pytest.mark.parametrize("n", [5, 17, 30])
def test_sampler_hits_exactly_n_deterministically(kug, n):
    items = _varied_items()
    samples = {}
    for seed in (848, 1, 2):
        state = random.getstate()
        sample = kug.sample_chaining_items(items, n, seed)
        assert random.getstate() == state  # the process's global generator is left alone
        assert sum(len(item["memorization_tasks"]) for item in sample) == n
        assert sample == kug.sample_chaining_items(items, n, seed)
        assert sample == _paper_sample(items, n, seed)
        samples[seed] = [item["id"] for item in sample]
    assert len({tuple(ids) for ids in samples.values()}) > 1  # the seed matters


def test_sampler_refuses_an_unreachable_count(kug):
    items = [{"memorization_tasks": [{}, {}]} for _ in range(5)]
    with pytest.raises(ValueError, match="exactly 7 memorization tasks"):
        kug.sample_chaining_items(items, 7, 0)


@pytest.mark.skipif(not os.environ.get("KUG_CHAINING_TASKS"), reason="set KUG_CHAINING_TASKS to the paper's data file")
def test_sampler_reproduces_the_papers_qwen_and_llama_samples(kug):
    import hashlib

    items = json.loads(Path(os.environ["KUG_CHAINING_TASKS"]).read_text())
    for seed, digest, n_questions in ((848, "a93b77beb067", 409), (1340, "e660ebd28355", 427)):
        sample = kug.sample_chaining_items(items, 1000, seed)
        assert hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()[:12] == digest
        assert len({item["generalization_tasks"][0]["prompt"] for item in sample}) == n_questions


# -- the data's source ---------------------------------------------------------------------------------------------

DATA_REPO = "speediedan/knowing-using-gap-chaining"
DATA_REVISION = "a1bb9cee0bb47bc956df037a8e013a4350d127db"


def test_the_data_downloads_from_the_pinned_hub_revision_by_default(kug, tmp_path, monkeypatch):
    calls = []
    downloaded = tmp_path / "hub" / "chaining_tasks.json"

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(downloaded)

    monkeypatch.setattr(kug, "hf_hub_download", fake_download)
    cfg = kug.KUGDataModuleConfig(task_name="knowing_using", dataset_path=str(tmp_path / "dataset"))
    assert cfg.chaining_tasks_path() == downloaded
    assert calls == [((DATA_REPO, "chaining_tasks.json"), {"repo_type": "dataset", "revision": DATA_REVISION})]
    # the shipped configurations name the same pin
    for rel in ("configs/datamodule.knowing_using.yaml", f"configs/{RUN_KEY}.yaml"):
        shipped = yaml.safe_load((KUG_DIR / rel).read_text())
        init_args = (shipped.get("registered_cfg", shipped))["datamodule_cfg"]["init_args"]
        assert (init_args["data_repo_id"], init_args["data_filename"], init_args["data_revision"]) == (
            DATA_REPO,
            "chaining_tasks.json",
            DATA_REVISION,
        )
        assert init_args["data_path"] is None
    (dataset,) = yaml.safe_load((KUG_DIR / "it_component.yaml").read_text())["module"]["task"]["datasets"]
    assert (dataset["path"], dataset["data_files"], dataset["revision"]) == (
        DATA_REPO,
        "chaining_tasks.json",
        DATA_REVISION,
    )


def test_a_local_data_path_overrides_the_download(kug, tmp_path, monkeypatch):
    def refuse_download(*args, **kwargs):
        raise AssertionError("a local data_path must not download")

    monkeypatch.setattr(kug, "hf_hub_download", refuse_download)
    local = tmp_path / "chaining_tasks.json"
    cfg = kug.KUGDataModuleConfig(
        task_name="knowing_using", dataset_path=str(tmp_path / "dataset"), data_path=str(local)
    )
    assert cfg.chaining_tasks_path() == local


# -- spans and splits ----------------------------------------------------------------------------------------------


def test_entity_spans_and_splits(kug):
    items = chaining_items()
    tokenizer = tiny_tokenizer(items)
    prompt_cfg = kug.KUGPromptConfig()
    prompt_cfg.bind_tokenizer(tokenizer)

    prompt_ids, completion_ids = prompt_cfg.qa_ids(items[0]["memorization_tasks"][0]["prompt"], "GLUL")
    span = kug.find_entity_span(tokenizer, prompt_ids, "Optic Choroid ")
    assert span is not None and len(span) == 2  # the shortest window, matched stripped and lowercased
    assert tokenizer.decode([prompt_ids[i] for i in span]) == "optic choroid"
    assert kug.find_entity_span(tokenizer, prompt_ids, "renal cortex") is None
    assert tokenizer.decode(completion_ids) == "GLUL <|im_end|>"  # the answer and the end of the assistant turn

    train, evaluation = kug.build_kug_splits(items, tokenizer, prompt_cfg)
    assert len(train) == 16
    assert [i for i, row in enumerate(train) if not row["span"]] == [14]  # item 7's head is not in its question
    for row, (item, k) in zip(train, [(item, k) for item in items for k in range(2)]):
        assert max(row["span"], default=-1) < row["prompt_len"]
        if row["span"]:
            decoded = tokenizer.decode([row["input_ids"][i] for i in row["span"]])
            assert decoded.lower() == item["facts"][k]["head"].lower()
    twohop = [row for row in evaluation if row["kind"] == "twohop"]
    assert len(twohop) == 7 and twohop[0]["gold"] == ["aspirin", "glutamine"]  # items 0 and 1 share a question
    assert sum(row["kind"] == "mem" for row in evaluation) == 16


# -- the LRSD gradient path ----------------------------------------------------------------------------------------


def _lora_model(n_layers: int = 6) -> Any:
    from peft import LoraConfig, get_peft_model

    model = tiny_model(50, n_layers=n_layers)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=targets))
    with torch.no_grad():  # LoRA's B starts at zero, which would leave every A without a gradient
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)
    return model


def _block_grads(model: Any, n_layers: int) -> list[bool]:
    """Per decoder block, whether any of its LoRA parameters received a nonzero gradient."""
    out = []
    for layer in range(n_layers):
        grads = [p.grad for n, p in model.named_parameters() if f"layers.{layer}." in n and "lora_" in n]
        out.append(any(g is not None and bool(g.abs().sum() > 0) for g in grads))
    return out


def test_lrsd_gradient_reaches_only_the_student_path(kug):
    n_layers, student, teacher = 6, 2, 4
    ids = torch.randint(0, 50, (3, 9), generator=torch.Generator().manual_seed(0))
    spans = [[2, 3], [], [5]]

    model = _lora_model(n_layers)
    hidden = model(input_ids=ids, output_hidden_states=True).hidden_states
    kug.lrsd_term(hidden, spans, student, teacher).backward()
    # the term depends on blocks 0..student through the student's state; the teacher's state is detached
    assert _block_grads(model, n_layers) == [True, True, True, False, False, False]

    # the check can fail: without the stop-gradient, the blocks between the student and the teacher learn too
    model = _lora_model(n_layers)
    hidden = model(input_ids=ids, output_hidden_states=True).hidden_states
    terms = [kug.relative_mse(hidden[student + 1][r, s], hidden[teacher + 1][r, s]) for r, s in enumerate(spans) if s]
    torch.stack(terms).mean().backward()
    assert _block_grads(model, n_layers) == [True, True, True, True, True, False]

    assert kug.lrsd_term(hidden, [[], []], student, teacher) is None


def test_relative_mse_matches_its_definition(kug):
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    expected = torch.stack([(a[i] - b[i]).pow(2).sum() / b[i].pow(2).sum() for i in range(4)]).mean()
    assert torch.allclose(kug.relative_mse(a, b), expected)


# -- exact match and clipping --------------------------------------------------------------------------------------


def test_exact_match_tallies_counts_over_uneven_batches(kug):
    assert kug.is_exact_match("  Aspirin ", ["aspirin"])
    assert not kug.is_exact_match("aspirin tablets", ["aspirin"])  # the whole string, not a substring
    preds = ["a", "x", "c", "d", "x", "f", "g"]
    golds = [["a"], ["b"], ["c"], ["d"], ["e"], ["F", "z"], ["y"]]
    kinds = ["twohop"] * 5 + ["mem"] * 2
    tally = kug.ExactMatchTally()
    batch_rates = []
    for i in range(0, 7, 3):  # batches of 3, 3 and 1
        tally.update(preds[i : i + 3], golds[i : i + 3], kinds[i : i + 3])
        batch_rates.append(sum(kug.is_exact_match(p, g) for p, g in zip(preds[i : i + 3], golds[i : i + 3])) / 3)
    metrics = tally.metrics()
    assert metrics == {
        "mem_correct": 1,
        "mem_em": 0.5,
        "mem_total": 2,
        "twohop_correct": 3,
        "twohop_em": 0.6,
        "twohop_total": 5,
    }
    overall = (metrics["twohop_correct"] + metrics["mem_correct"]) / 7
    assert overall != pytest.approx(sum(batch_rates) / len(batch_rates))  # a per-batch average would differ
    tally.reset()
    assert tally.metrics() == {}


def test_clip_optimizer_grads(kug):
    params = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(3))]
    params[0].grad, params[1].grad = torch.full((4,), 3.0), torch.full((3,), 4.0)
    optimizer = torch.optim.SGD(params, lr=0.1)
    kug.clip_optimizer_grads(optimizer, 1.0)
    total = torch.sqrt(sum(p.grad.pow(2).sum() for p in params))
    assert total == pytest.approx(1.0, rel=1e-4)


# -- the shipped run, end to end -----------------------------------------------------------------------------------


class KUGRunRecorder:
    """Records each epoch's metrics and, at the end, the trainable parameters, whether LoRA's B matrices moved from
    zero, and the optimizer's step pre-hooks (the module's gradient clipping)."""

    def __init__(self, out: str) -> None:
        self.out = out
        self.epochs: list[dict] = []

    def on_train_epoch_end(self, ctx: Any, metrics: dict) -> None:
        self.epochs.append({"epoch": ctx.epoch, **{k: float(v) for k, v in metrics.items()}})

    def on_train_end(self, ctx: Any) -> None:
        params = dict(ctx.module.named_parameters())
        thawed = sorted(n for n, p in params.items() if p.requires_grad)
        b_moved = any(bool(p.abs().sum() > 0) for n, p in params.items() if "lora_B" in n)
        # the module registers its gradient clipping on the optimizer the tuning scheduler built
        clip_hooks = len(ctx.optimizer._optimizer_step_pre_hooks)
        record = {"epochs": self.epochs, "thawed": thawed, "b_moved": b_moved, "clip_hooks": clip_hooks}
        Path(self.out).write_text(json.dumps(record))


def _run_config(revision: str, work: Path) -> Path:
    from interpretune.hub.components import resolve_component_file

    shipped = resolve_component_file(f"hub://{KUG_REPO}@{revision}/runs/{RUN_KEY}.yaml")
    text = shipped.read_text(encoding="utf-8").replace(f"hub://{KUG_REPO}/", f"hub://{KUG_REPO}@{revision}/")
    cfg = yaml.safe_load(text)
    cfg["session_ref"] = f"{cfg['session_ref']}@{revision}"
    callbacks = cfg["run_cfg"]["callbacks"]
    callbacks[1]["init_args"]["log_path"] = str(work / "exact_match.json")
    callbacks.append({"class_path": f"{__name__}.KUGRunRecorder", "init_args": {"out": str(work / "record.json")}})
    path = work / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_the_shipped_run_trains_lora_and_scores_every_epoch(kug, kug_revision, clean_cli_env, tmp_path, monkeypatch):
    from interpretune.base.components.cli import core_cli_main

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.chdir(tmp_path)
    items = chaining_items()
    data_path = tmp_path / "chaining_tasks.json"
    data_path.write_text(json.dumps(items))
    tokenizer = tiny_tokenizer(items)
    # the tokenizer in a directory of its own: beside a Qwen2 `config.json`, `AutoTokenizer` would load it as a Qwen2
    # BPE tokenizer
    model_dir, tokenizer_dir = tmp_path / "tiny_qwen2", tmp_path / "tiny_tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    tiny_model(len(tokenizer), tokenizer).save_pretrained(model_dir)

    n_facts, fact_seed = 12, 0  # 12 of the 16 facts; the batch sizes divide neither split
    sample = kug.sample_chaining_items(items, n_facts, fact_seed)
    n_twohop = len({item["generalization_tasks"][0]["prompt"] for item in sample})
    n_mem = len({task["prompt"] for item in sample for task in item["memorization_tasks"]})

    registered = {
        "datamodule_cfg": {
            "init_args": {
                "data_path": str(data_path),
                "dataset_path": str(tmp_path / "dataset"),
                "fact_seed": fact_seed,
                "n_memorization": n_facts,
                "train_batch_size": 5,
                "eval_batch_size": 4,
            }
        },
        "module_cfg": {
            "init_args": {
                "lrsd_student_layer": 1,
                "lrsd_teacher_layer": 2,
                "hf_from_pretrained_cfg": {"init_args": {"pretrained_kwargs": {"device_map": "cpu"}}},
            }
        },
    }
    args = ["--config", str(_run_config(kug_revision, tmp_path))]
    args += [
        "--shared_config",
        json.dumps({"model_name_or_path": str(model_dir), "tokenizer_name": str(tokenizer_dir)}),
    ]
    args += ["--registered_cfg", json.dumps(registered), "--run_cfg.max_epochs=2"]
    core_cli_main(run_mode="train", args=args)

    history = json.loads((tmp_path / "exact_match.json").read_text())
    assert [entry["epoch"] for entry in history] == [0, 1, 2]  # before training, then after each epoch
    for entry in history:
        assert (entry["twohop_total"], entry["mem_total"]) == (n_twohop, n_mem)
        assert entry["twohop_em"] == entry["twohop_correct"] / n_twohop

    record = json.loads((tmp_path / "record.json").read_text())
    assert record["thawed"] and all(".lora_A." in n or ".lora_B." in n for n in record["thawed"])
    assert len(record["thawed"]) == N_LAYERS * 7 * 2
    assert record["b_moved"] and record["clip_hooks"] == 1
    for epoch in record["epochs"]:
        assert {"train_ce", "train_lrsd", "train_loss", "twohop_em", "mem_em"} <= set(epoch)
