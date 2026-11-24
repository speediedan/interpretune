"""Mount a pretrained PEFT adapter on a session's Hugging Face model, for an op that reads through it.

Some lenses are adapters on the model they read: they take the base model's activations, and decode them with the
adapter switched on. Such an op should not load a second copy of the base, and it should not leave the session's
model changed outside its read. :func:`mount_adapter` attaches an adapter to the session's own HF model, switched off,
and :func:`adapter_enabled` switches it on for the length of a ``with`` block only.

Training a new adapter is a different contract and stays with the module's ``lora_cfg``
(``HFFromPretrainedMixin._configure_peft``). ``peft`` is imported only when an adapter is mounted.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Any, Iterator

import torch

MOUNTS_ATTR = "_it_mounted_adapters"
_SNAPSHOT = re.compile(r"models--([^/\\]+)--([^/\\]+)[/\\]snapshots[/\\]")


@dataclass(frozen=True)
class AdapterMount:
    """What was mounted under ``name``: the adapter's source and revision, and the base it was trained on."""

    name: str
    source: str
    revision: str | None
    base: str


def hf_model_of(model: Any) -> Any:
    """The Hugging Face model an op reads through: the model itself, or the one an nnsight wrapper holds.

    nnsight keeps the wrapped module at ``_module``. It is read from the instance dict, because an envoy forwards any
    attribute it lacks to that module, so ``getattr`` would answer for the wrong object. A TransformerLens bridge is
    refused: its read path has no test that an adapter mounted on its HF model is off outside the block.
    """
    inner = vars(model).get("_module") if hasattr(model, "__dict__") else None
    hf = inner if isinstance(inner, torch.nn.Module) else model
    if type(hf).__name__ == "TransformerBridge":
        raise TypeError(
            "mounting a pretrained adapter is supported on an HF model or an nnsight wrapper of one, not on a "
            "TransformerLens TransformerBridge"
        )
    if not hasattr(hf, "load_adapter"):
        raise TypeError(
            f"{type(hf).__name__} has no PEFT adapter support (transformers' `load_adapter`); mount the adapter on a "
            "transformers PreTrainedModel"
        )
    return hf


def base_id(name_or_path: str) -> str:
    """A Hub id for a model reference: the id itself, or the id a Hub cache snapshot path was downloaded from."""
    m = _SNAPSHOT.search(name_or_path)
    return f"{m[1]}/{m[2]}" if m else name_or_path.rstrip("/\\")


def _require_peft() -> None:
    try:
        import peft  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "mounting a pretrained adapter needs `peft`, which interpretune imports only when an op mounts one: "
            "install it with `uv pip install peft`"
        ) from err


def _declared_base(source: str, revision: str | None) -> str:
    from peft import PeftConfig

    return str(PeftConfig.from_pretrained(source, revision=revision).base_model_name_or_path or "")


def mounted_adapters(model: Any) -> dict[str, AdapterMount]:
    """The adapters mounted on ``model`` (or the HF model under its nnsight wrapper), by name."""
    return dict(vars(hf_model_of(model)).get(MOUNTS_ATTR, {}))


def mount_adapter(
    model: Any, source: str, *, name: str, base: str, revision: str | None = None, **load_kwargs: Any
) -> AdapterMount:
    """Attach the adapter at ``source`` (a Hub id or a local directory) to the session's HF model, switched off.

    ``base`` is the model the adapter was trained on, as a Hub id. The session's model and the adapter's own config
    must both name it (a Hub cache snapshot path counts as the id it came from); anything else is refused, since an
    adapter on the wrong base loads without error and reads nonsense. Mounting again under the same name returns the
    existing mount when the source and revision match, and is refused when they differ.
    """
    hf = hf_model_of(model)
    _require_peft()
    found = base_id(str(getattr(hf.config, "_name_or_path", "") or ""))
    if found != base:
        raise ValueError(f"the adapter {source!r} is for {base!r}, but the session's model is {found or 'unnamed'!r}")
    want = AdapterMount(name, str(source), revision, base)
    mounts: dict[str, AdapterMount] = hf.__dict__.setdefault(MOUNTS_ATTR, {})
    if name in mounts:
        if mounts[name] != want:
            raise ValueError(f"an adapter is already mounted as {name!r}: {mounts[name]}; asked for {want}")
        return want
    declared = base_id(_declared_base(str(source), revision))
    if declared and declared != base:
        raise ValueError(f"the adapter {source!r} declares the base {declared!r}, not {base!r}")
    # the revision reaches the adapter's config and weights through `adapter_kwargs` (download options), not as a
    # top-level keyword, which `load_adapter` passes to its state-dict loader
    adapter_kwargs = {**load_kwargs.pop("adapter_kwargs", {}), **({"revision": revision} if revision else {})}
    info = hf.load_adapter(str(source), adapter_name=name, adapter_kwargs=adapter_kwargs or None, **load_kwargs)
    _check_fit(hf, name, str(source), info)
    hf.disable_adapters()  # loading switches the adapter on; a mount leaves the session's model as it was
    mounts[name] = want
    return want


def _check_fit(hf: Any, name: str, source: str, info: Any) -> None:
    """Refuse an adapter whose tensors did not all land on the model: PEFT gives every configured target fresh LoRA
    weights, so an adapter whose keys match nothing loads without error and reads as a random adapter would."""
    if info is None:
        hf.delete_adapter(name)
        raise RuntimeError(f"loading {source!r} returned no load report, so the mount cannot be checked")
    unexpected = sorted(getattr(info, "unexpected_keys", None) or ())
    missing = sorted(k for k in (getattr(info, "missing_keys", None) or ()) if f".{name}." in k)
    mismatched = sorted(str(k) for k in (getattr(info, "mismatched_keys", None) or ()))
    if unexpected or missing or mismatched:
        hf.delete_adapter(name)
        parts = [
            f"{len(keys)} {what} (e.g. {keys[0]})"
            for keys, what in (
                (unexpected, "of its tensors match no parameter of the model"),
                (missing, "adapter parameters were left uninitialized"),
                (mismatched, "tensors have the wrong shape"),
            )
            if keys
        ]
        raise ValueError(f"the adapter {source!r} does not fit the session's model: " + "; ".join(parts))


@contextlib.contextmanager
def adapter_enabled(model: Any, name: str) -> Iterator[Any]:
    """Switch the mounted adapter ``name`` on for the block, and off again when it ends, including on an error."""
    hf = hf_model_of(model)
    mounts = vars(hf).get(MOUNTS_ATTR, {})
    if name not in mounts:
        raise KeyError(f"no adapter is mounted as {name!r}; mounted: {sorted(mounts) or 'none'}")
    hf.set_adapter(name)
    hf.enable_adapters()
    try:
        yield hf
    finally:
        hf.disable_adapters()
