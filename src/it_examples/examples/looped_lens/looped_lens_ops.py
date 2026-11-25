"""The ``looped_lens_read`` op: a looped lens read from Inkling 2 Mini through interpretune's op dispatch.

Locally the op hooks the module's own model (an HF ``Inkling2LoopForConditionalGeneration``, or an nnsight wrapper
around one) and runs one forward pass per prompt, the regime the lens was selected in. With ``looped_lens_remote`` it
runs one NDIF trace per prompt through the module's nnsight model and decodes the returned states with the
checkpoint's head (``looped_lens_head``); coda readouts come back as patched batch rows (``looped_lens_runtime``).
"""

from __future__ import annotations

import pathlib
import sys
import weakref
from typing import Any

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# resolved from the component directory, which the line above puts on the path
import looped_lens_runtime as R  # noqa: E402  # pyright: ignore[reportMissingImports]

_RUNTIMES: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def _arg(analysis_batch: Any, kwargs: dict, name: str, default: Any = None) -> Any:
    value = kwargs.get(name)
    if value is None and analysis_batch is not None:
        value = analysis_batch.get(name) if hasattr(analysis_batch, "get") else None
    return default if value is None else value


def _hf_model(model: Any) -> Any:
    """The torch model under an nnsight wrapper, or the model itself.

    nnsight 0.8 keeps the wrapped module at ``_module``. It is read from the instance dict, because an envoy forwards
    any attribute it lacks to that module, so ``getattr`` would answer for the wrong object.
    """
    inner = vars(model).get("_module") if hasattr(model, "__dict__") else None
    return inner if isinstance(inner, torch.nn.Module) else model


def _local_runtime(module: Any, table_path: str, map_path: str | None, translators_path: str | None):
    cache = _RUNTIMES.setdefault(module, {})
    key = (table_path, map_path, translators_path)
    if key not in cache:
        table = R.load_table(table_path, map_path)
        translators = torch.load(translators_path, weights_only=False, map_location="cpu") if translators_path else None
        from interpretune.analysis.optools import resolve_tokenizer

        cache[key] = R.LensRuntime(_hf_model(module.model), resolve_tokenizer(module), table, translators)
    return cache[key]


def looped_lens_read_impl(module, analysis_batch, batch, batch_idx: int, **kwargs):
    """Read the lens at one read position of every prompt in the batch (a book may place a site at its question's
    end).

    Returns the site labels, per prompt the filtered token lists (the N x K labelled tokens the challenge pools) and
    where each site was read, and the unfiltered top 100 ids per site for scoring or a later op.
    """
    table_path = _arg(analysis_batch, kwargs, "looped_lens_table")
    if table_path is None:
        raise ValueError(
            "looped_lens_read needs `looped_lens_table`: a resolved site table (.json) or a lens book "
            "(.yaml, with `looped_lens_map`)"
        )
    map_path = _arg(analysis_batch, kwargs, "looped_lens_map")
    remote = bool(_arg(analysis_batch, kwargs, "looped_lens_remote", False))
    translators_path = _arg(analysis_batch, kwargs, "looped_lens_translators")
    input_ids, mask = batch["input_ids"], batch.get("attention_mask")
    offsets = _arg(analysis_batch, kwargs, "looped_lens_positions")
    prompts = []
    for b in range(input_ids.shape[0]):
        ids = input_ids[b] if mask is None else input_ids[b][mask[b].bool()]
        prompts.append([int(t) for t in ids])
    offs = (
        [int(o) for o in (offsets.tolist() if hasattr(offsets, "tolist") else offsets)]
        if offsets is not None
        else [-1] * len(prompts)
    )
    if len(offs) != len(prompts):
        raise ValueError(f"looped_lens_positions has {len(offs)} entries for {len(prompts)} prompts")
    if remote:
        from safetensors.torch import load_file

        from interpretune.analysis.optools import resolve_tokenizer

        head_path = _arg(analysis_batch, kwargs, "looped_lens_head")
        if head_path is None:
            raise ValueError("a remote read decodes NDIF states locally: pass `looped_lens_head` (see ARTIFACTS.md)")
        table = R.load_table(table_path, map_path)
        R.check_remote(table)
        head = load_file(head_path)
        w_u = head["model.llm.unembed.weight"][: R.UNPADDED_VOCAB].float()
        gain = head["model.llm.norm.weight"].float()
        translators = torch.load(translators_path, weights_only=False, map_location="cpu") if translators_path else None
        tok = resolve_tokenizer(module)
        llm = module.model.model.language_model
        vocab = [tok.decode([i]) for i in range(R.UNPADDED_VOCAB)]

        def top_ids(ids: list[int], where: dict[str, int]) -> dict[str, torch.Tensor]:
            return R.remote_top_ids(module.model, llm, table, ids, where, w_u, gain, translators)
    else:
        rt = _local_runtime(module, str(table_path), map_path, translators_path)
        table, tok, vocab = rt.table, rt.tok, rt.vocab

        def top_ids(ids: list[int], where: dict[str, int]) -> dict[str, torch.Tensor]:
            return rt.top_ids(ids, where)

    spec = table["spec"]
    labels = [s["label"] for s in table["sites"]]
    k, filters = int(spec.get("K", 3)), set(spec.get("filters", []))
    all_top, tokens, positions, site_positions = [], [], [], []
    for ids, off in zip(prompts, offs):
        pos = len(ids) + off
        where = R.site_positions(table, ids, pos, tok)
        top = top_ids(ids, where)
        stacked = torch.stack([top[lab][0] for lab in labels])  # (N, 100)
        all_top.append(stacked)
        tokens.append(R.build_lists(stacked, vocab, tok.decode(ids), k, filters))
        positions.append(pos)
        site_positions.append([where[lab] for lab in labels])
    analysis_batch.update(
        looped_lens_labels=labels,
        looped_lens_tokens=tokens,
        looped_lens_top_ids=torch.stack(all_top).long(),
        looped_lens_positions=torch.tensor(positions),
        looped_lens_site_positions=torch.tensor(site_positions),
        looped_lens_book=str(spec.get("book", table_path)),
    )
    return analysis_batch
