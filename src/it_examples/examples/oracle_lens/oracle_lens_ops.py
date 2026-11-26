"""The ``oracle_lens_read`` op: WorkspaceBench's oracle lens read from the session's own Qwen3.6-27B.

For each prompt and each requested layer the op captures the base model's block output at one position (every
adapter off), then samples readouts with the lens adapter mounted on the same model and switched on only for the read.
With ``oracle_lens_ar`` each readout's bullets are also mapped to directions by the reconstructor and fit to the
activation in the whitened space, giving the fraction explained and one non-negative coefficient per bullet.
"""

from __future__ import annotations

import json
import pathlib
import sys
import weakref
from typing import Any

import torch

try:
    # a copy loaded from the Hub is a package, which holds the runtime of its own revision
    from . import oracle_lens_runtime as R
except ImportError:
    if __package__:
        raise
    # loaded from a directory on IT_ANALYSIS_OP_PATHS, the module stands alone: its directory is the search path
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import oracle_lens_runtime as R  # pyright: ignore[reportMissingImports]

_ARS: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def _arg(analysis_batch: Any, kwargs: dict, name: str, default: Any = None) -> Any:
    value = kwargs.get(name)
    if value is None and analysis_batch is not None:
        value = analysis_batch.get(name) if hasattr(analysis_batch, "get") else None
    return default if value is None else value


def _ints(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(v) for v in (value if isinstance(value, (list, tuple)) else [value])]


def _scratch(revision: str) -> pathlib.Path:
    from interpretune.hub.cache import IT_CACHE

    return IT_CACHE / "oracle_lens" / revision


def _ar(module: Any, root: pathlib.Path) -> Any:
    cache = _ARS.setdefault(module, {})
    if root not in cache:
        cache[root] = R.AR(module.model, root)
    return cache[root]


def oracle_lens_read_impl(module, analysis_batch, batch, batch_idx: int, **kwargs):
    """Read the oracle lens at one position of every prompt, at each requested layer.

    Returns, per prompt, layer and sample, the readout text and its bullets; with ``oracle_lens_ar``, also the
    fraction of the whitened activation the bullets explain and their coefficients.
    """
    from interpretune.analysis.optools import resolve_tokenizer

    name = str(_arg(analysis_batch, kwargs, "oracle_lens_adapter", "space"))
    R.check_adapter(name)
    layers = [R.check_layer(layer) for layer in _ints(_arg(analysis_batch, kwargs, "oracle_lens_layers", [44]))]
    use_ar = bool(_arg(analysis_batch, kwargs, "oracle_lens_ar", False))
    revision = str(_arg(analysis_batch, kwargs, "oracle_lens_revision", R.LENS_REV))
    root = R.lens_root(revision, _arg(analysis_batch, kwargs, "oracle_lens_root"))
    samples = int(_arg(analysis_batch, kwargs, "oracle_lens_samples", 1))
    seed = int(_arg(analysis_batch, kwargs, "oracle_lens_seed", 0))
    max_new_tokens = int(_arg(analysis_batch, kwargs, "oracle_lens_max_new_tokens", 128))
    temperature = float(_arg(analysis_batch, kwargs, "oracle_lens_temperature", 0.8))
    top_p = float(_arg(analysis_batch, kwargs, "oracle_lens_top_p", 0.95))
    sampling = dict(samples=samples, seed=seed, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    scratch = _scratch(revision)
    R.mount_lens(module.model, name, root, scratch)
    ar = None
    if use_ar:
        R.mount_lens(module.model, R.AR_NAME, root, scratch)
        ar = _ar(module, root)
    tok = resolve_tokenizer(module)

    input_ids, mask = batch["input_ids"], batch.get("attention_mask")
    offsets = _arg(analysis_batch, kwargs, "oracle_lens_positions")
    n = input_ids.shape[0]
    offs = _ints(offsets) if offsets is not None else [-1] * n
    if len(offs) != n:
        raise ValueError(f"oracle_lens_positions has {len(offs)} entries for {n} prompts")

    readouts, bullets, fves, coefs, positions = [], [], [], [], []
    for b in range(n):
        ids = input_ids[b] if mask is None else input_ids[b][mask[b].bool()]
        pos = len(ids) + offs[b] if offs[b] < 0 else offs[b]
        if not 0 <= pos < len(ids):
            raise ValueError(f"prompt {b}: position {offs[b]} is outside its {len(ids)} tokens")
        acts = R.capture(module.model, ids.cpu(), tuple(layers))
        per_layer_text, per_layer_bullets, per_layer_fve, per_layer_coef = [], [], [], []
        for layer in layers:
            h = acts[layer][pos][None]
            texts = R.read(
                module.model,
                tok,
                name,
                layer,
                h,
                samples=samples,
                seed=seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )[0]
            bl = [R.bullets(t) for t in texts]
            per_layer_text.append(texts)
            per_layer_bullets.append(bl)
            if ar is not None:
                y = ar.white(h[0], layer)
                f_row, c_row = [], []
                for phrases in bl:
                    if not phrases:  # nothing to fit: recorded as absent, never as zero explained
                        f_row.append(None)
                        c_row.append([])
                        continue
                    spans = [tok(" " + p, add_special_tokens=False).input_ids for p in phrases]
                    D = ar.white(ar.predict(spans, layer), layer)
                    fve, c = R.joint_fve(D, y)
                    f_row.append(fve)
                    c_row.append(c.tolist())
                per_layer_fve.append(f_row)
                per_layer_coef.append(c_row)
        readouts.append(per_layer_text)
        bullets.append(per_layer_bullets)
        fves.append(per_layer_fve)
        coefs.append(per_layer_coef)
        positions.append(pos)
    analysis_batch.update(
        oracle_lens_readouts=readouts,
        oracle_lens_bullets=bullets,
        oracle_lens_layers=torch.tensor(layers),
        oracle_lens_positions=torch.tensor(positions),
        oracle_lens_sampling=json.dumps({"adapter": name, "revision": revision, **sampling}, sort_keys=True),
    )
    if ar is not None:
        analysis_batch.update(oracle_lens_fve=fves, oracle_lens_coefs=coefs)
    return analysis_batch
