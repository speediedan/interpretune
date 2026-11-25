"""The oracle lens runtime for Qwen3.6-27B: an activation in, sampled verbal readouts (and, with the AR,
coefficients) out.

The lens is WorkspaceBench's metamodel lens, released as LoRA adapters on ``Qwen/Qwen3.6-27B``
(``agu18dec/olens_and_ar``). The authors' reader package is not public, so this follows the checkpoint cards:

* the verbalizer prompt carries an injection marker, the first character of U+3200..U+33FF that is exactly one token of
  the fully rendered prompt (:func:`render_prompt`), and the marker's embedding row is overwritten with
  ``transform(h)`` (:func:`inject`): ``unit`` is ``alpha * h / |h|``, ``scaled`` is ``alpha * h``;
* the activation is the base model's (adapters off), read after a decoder block; the readout is sampled with the
  named adapter on (:func:`read`);
* the reconstructor (``AR``) maps a phrase to a direction at a layer, so a readout's bullets can be fit to the
  activation they came from in the whitened space the AR was trained in (:func:`joint_fve`).

Every contract term the checkpoints fix (the base, the trained layer band, the prompt and transform per adapter) is
refused when a caller asks for something else. Adapters are mounted on the caller's model with
``interpretune.utils.peft_mount``, so no second copy of the base is loaded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from interpretune.utils.peft_mount import adapter_enabled, hf_model_of, mount_adapter

BASE_ID, BASE_REV = "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LENS_REPO, LENS_REV = "agu18dec/olens_and_ar", "74e9dcdceafa3f14c38e223b5217ca9db44b3cba"
LAYERS = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)
MAX_SPAN = 32  # the AR was trained on spans of at most 32 tokens
PROMPTS = {
    "concepts_raw": (
        "An activation vector from layer {layer} of a language model is enclosed in activation tags:\n"
        "<activation>{char}</activation>. Produce distinct concepts that encode this activation, each as\n"
        "a '- ' bullet on its own line."
    ),
    "continuation_raw": (
        "An activation vector from layer {layer} of a language model is enclosed in activation tags:\n"
        "<activation>{char}</activation>. Produce the text that immediately follows this activation."
    ),
}
# per adapter, as the cards publish it: the subfolder, the prompt it was trained on and the injection transform
CONTRACT = {
    "space": dict(dir="olens_s3d_space_rl600", prompt="concepts_raw", transform="unit", alpha=16000.0),
    "rl600": dict(dir="olens_s3d_rl600", prompt="concepts_raw", transform="unit", alpha=16000.0),
    "student": dict(dir="olens_s3d_student_step251", prompt="concepts_raw", transform="unit", alpha=16000.0),
    "inverter": dict(dir="olens_s3d_inverter", prompt="continuation_raw", transform="scaled", alpha=122.32250086762771),
}
AR_NAME = "ar"


@dataclass
class Prompt:
    """A rendered verbalizer prompt: token ids, the marker's slot and the marker's token id."""

    ids: torch.Tensor
    slot: int
    char_id: int


def adapter_name(name: str) -> str:
    """The name an oracle-lens adapter is mounted under on the session's model."""
    return f"oracle_lens_{name}"


def check_layer(layer: int) -> int:
    if layer not in LAYERS:
        raise ValueError(f"layer {layer} is outside the oracle lens's trained band {LAYERS}")
    return layer


def check_adapter(name: str) -> dict:
    if name not in CONTRACT:
        raise ValueError(f"unknown oracle lens adapter {name!r}; the released ones are {sorted(CONTRACT)}")
    return CONTRACT[name]


def prompt_text(kind: str, layer: int, char: str, wrap: bool = False) -> str:
    """The cards print the prompts hard-wrapped; ``wrap=False`` joins the lines with spaces, as the lens was
    read."""
    if kind not in PROMPTS:
        raise ValueError(f"unknown oracle lens prompt {kind!r}; the published ones are {sorted(PROMPTS)}")
    text = PROMPTS[kind].format(layer=layer, char=char)
    return text if wrap else " ".join(text.split("\n"))


def _chat_ids(tok: Any, text: str) -> list[int]:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    return tok(rendered, add_special_tokens=False).input_ids


def render_prompt(tok: Any, kind: str, layer: int, wrap: bool = False) -> Prompt:
    """The prompt for ``kind`` at ``layer`` with its marker: never hardcoded, found on the fully rendered
    prompt."""
    for cp in range(0x3200, 0x3400):
        char = chr(cp)
        ids = _chat_ids(tok, prompt_text(kind, layer, char, wrap))
        alone = tok(char, add_special_tokens=False).input_ids
        if len(alone) == 1 and ids.count(alone[0]) == 1:
            return Prompt(torch.tensor(ids), ids.index(alone[0]), alone[0])
    raise ValueError("no character in U+3200..U+33FF is a single token appearing once in the rendered prompt")


def transform(h: torch.Tensor, kind: str, alpha: float) -> torch.Tensor:
    h = h.float()
    if kind == "unit":
        return alpha * h / h.norm(dim=-1, keepdim=True)
    if kind == "scaled":
        return alpha * h
    raise ValueError(f"unknown oracle lens transform {kind!r} (unit, scaled)")


def inject(model: Any, prompt: Prompt, vecs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Input embeddings [b, seq, d] with the marker's row replaced by ``vecs`` [b, d], and the attention mask."""
    emb = hf_model_of(model).get_input_embeddings()
    ids = prompt.ids.to(emb.weight.device)
    out = emb(ids)[None].expand(vecs.shape[0], -1, -1).clone()
    out[:, prompt.slot] = vecs.to(out.device, out.dtype)
    return out, torch.ones(out.shape[:2], dtype=torch.long, device=out.device)


def decoder_layers(model: Any) -> Any:
    """The decoder blocks of the session's model, whether a causal LM or a multimodal model around one."""
    hf = hf_model_of(model)
    dec = hf.get_decoder() if hasattr(hf, "get_decoder") else hf.model
    if not hasattr(dec, "layers"):
        raise TypeError(f"{type(hf).__name__} has no decoder blocks at `get_decoder().layers`")
    return dec


def lens_root(revision: str = LENS_REV, local_dir: str | Path | None = None) -> Path:
    """The released checkpoint directory: ``local_dir`` when given, else the Hub snapshot at ``revision``."""
    if local_dir is not None:
        return Path(local_dir)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(LENS_REPO, revision=revision))


def _remap_ar(src: Path, scratch: Path) -> Path:
    """The AR's LoRA keys name a compiled backbone (``base_model.model.layers.N._orig_mod.``); rewrite them for a
    causal LM (``base_model.model.model.layers.N.``).

    Two keys mapping to one is refused. The rewritten copy is kept under ``scratch`` by the identity of the released
    file (its resolved path, size and modification time), so another checkpoint never reads an earlier one's copy, and
    it is moved into place whole, so an interrupted write is not taken for a finished one.
    """
    from safetensors.torch import load_file, save_file

    weights = (src / "adapter_model.safetensors").resolve()
    stat = weights.stat()
    key = hashlib.sha256(f"{weights}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    dst = scratch / f"{AR_NAME}-{key}"
    if dst.exists():
        return dst
    out: dict[str, torch.Tensor] = {}
    for k, v in load_file(str(weights)).items():
        nk = k.replace("base_model.model.layers.", "base_model.model.model.layers.").replace("._orig_mod.", ".")
        if nk in out:
            raise ValueError(f"two AR keys map to {nk}")
        out[nk] = v
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        part = Path(tmp) / "adapter"
        part.mkdir()
        save_file(out, str(part / "adapter_model.safetensors"))
        shutil.copyfile(src / "adapter_config.json", part / "adapter_config.json")
        try:
            part.rename(dst)
        except OSError:
            if not dst.exists():  # anything but another process finishing the same copy first
                raise
    return dst


def mount_lens(model: Any, name: str, root: Path, scratch: Path) -> str:
    """Mount the named adapter (or the AR) on the session's model, switched off; returns the name it is mounted
    under.

    ``mount_adapter`` refuses a base other than Qwen3.6-27B and an adapter whose tensors do not all land on the model.
    """
    if name == AR_NAME:
        src = _remap_ar(root / "ar_ptag_pooled" / "lora", scratch)
    else:
        src = root / check_adapter(name)["dir"]
    mounted = adapter_name(name)
    mount_adapter(model, str(src), name=mounted, base=BASE_ID)  # a remount of the same source is a no-op
    return mounted


@torch.no_grad()
def capture(model: Any, input_ids: torch.Tensor, layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Each layer's block output [seq, d] for one prompt, with every adapter off (the base model's activation)."""
    dec, out, hooks = decoder_layers(model), {}, []
    for layer in layers:

        def hook(_m, _a, o, layer=layer):
            out[layer] = (o[0] if isinstance(o, tuple) else o)[0].float().cpu()

        hooks.append(dec.layers[layer].register_forward_hook(hook))
    hf = hf_model_of(model)
    try:
        dec(input_ids=input_ids[None].to(hf.get_input_embeddings().weight.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return out


@torch.no_grad()
def read(
    model: Any,
    tok: Any,
    name: str,
    layer: int,
    h: torch.Tensor,
    *,
    samples: int = 1,
    seed: int = 0,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    wrap: bool = False,
) -> list[list[str]]:
    """Readouts [n][samples] for activations ``h`` [n, d] at ``layer`` under the named adapter's contract."""
    c = check_adapter(name)
    p = render_prompt(tok, c["prompt"], check_layer(layer), wrap)
    v = transform(h, c["transform"], c["alpha"]).repeat_interleave(samples, 0)
    emb, mask = inject(model, p, v)
    torch.manual_seed(seed)
    with adapter_enabled(model, adapter_name(name)) as hf:
        out = hf.generate(
            inputs_embeds=emb,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    text = [tok.decode(o, skip_special_tokens=True) for o in out]
    return [text[i * samples : (i + 1) * samples] for i in range(h.shape[0])]


def bullets(text: str) -> list[str]:
    """The ``- `` bullets of a readout, in order; the concept prompts ask for one per line."""
    return [
        line.strip()[2:].strip()
        for line in text.splitlines()
        if line.strip().startswith("- ") and line.strip()[2:].strip()
    ]


class AR:
    """The s3d reconstructor: a phrase at a layer to a direction, from the AR adapter's last real hidden state.

    ``heads.pt`` holds the per-layer tag tokens and a Linear head. The head's LayerNorm has no learned parameters, and
    the hidden state is taken after the model's final norm; step 1 of interpretune#1202 checked both by reproducing
    the card's per-layer FVE.
    """

    def __init__(self, model: Any, root: Path):
        sd = torch.load(root / "ar_ptag_pooled" / "heads.pt", map_location="cpu", weights_only=True)
        self.model, self.root = model, root
        self.layers, self.tag_ids = list(sd["layers"]), sd["tag_ids"]
        self.w, self.b = sd["head"]["linear.weight"].float(), sd["head"]["linear.bias"].float()
        self._whiteners: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def predict(self, spans: list[list[int]], layer: int, pad_id: int = 0) -> torch.Tensor:
        """[b, d] directions for token spans (each capped at :data:`MAX_SPAN`)."""
        if layer not in self.layers:
            raise ValueError(f"the AR has no head for layer {layer}; it covers {self.layers}")
        tag = self.tag_ids[self.layers.index(layer)].tolist()
        seqs = [tag + s[:MAX_SPAN] for s in spans]
        n = max(map(len, seqs))
        dev = hf_model_of(self.model).get_input_embeddings().weight.device
        ids = torch.full((len(seqs), n), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, device=dev)
            mask[i, : len(s)] = 1
        with adapter_enabled(self.model, adapter_name(AR_NAME)):
            hs = decoder_layers(self.model)(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        h = hs[torch.arange(len(seqs), device=hs.device), (mask.sum(1) - 1).to(hs.device)].float()
        h = torch.nn.functional.layer_norm(h, h.shape[-1:])
        return h @ self.w.to(h.device).T + self.b.to(h.device)

    def white(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """``(x - mu) @ W`` with the released per-layer whitener; the AR's loss and the FVE are in this space."""
        if layer not in self._whiteners:
            from safetensors.torch import load_file

            f = load_file(str(self.root / "whitener" / "W" / f"whitening_iolens_pooled_W_L{layer}.safetensors"))
            self._whiteners[layer] = (f["mu"].float(), f["w"].float())
        mu, w = self._whiteners[layer]
        return (x.float().cpu() - mu) @ w


def nnls(D: torch.Tensor, y: torch.Tensor, iters: int = 2000) -> torch.Tensor:
    """``argmin_{c >= 0} |y - D^T c|^2`` for D [k, d], y [d]: scipy's exact solver when installed (as step 1 of
    interpretune#1202 measured), else projected gradient."""
    try:
        from scipy.optimize import nnls as exact
    except ImportError:
        exact = None
    if exact is not None:
        return torch.tensor(exact(D.T.double().numpy(), y.double().numpy())[0], dtype=torch.float32)
    G, b = (D @ D.T).double(), (D @ y).double()
    top = float(torch.linalg.eigvalsh(G)[-1])
    if top <= 0:
        return torch.zeros(D.shape[0])
    c = torch.zeros(D.shape[0], dtype=torch.float64)
    for _ in range(iters):
        c = (c - (G @ c - b) / top).clamp_min(0)
    return c.float()


def joint_fve(D: torch.Tensor, y: torch.Tensor) -> tuple[float, torch.Tensor]:
    """The fraction of ``|y|^2`` the rows of D explain with non-negative coefficients, and the coefficients.

    A zero target is refused: its fraction explained is undefined.
    """
    y, D = y.float().cpu(), D.float().cpu()
    # one reduction for both sums, in float64: an all-zero fit must explain exactly 0, not a rounding error either side
    yy = float(y.double().pow(2).sum())
    if yy == 0:
        raise ValueError("the whitened activation has zero norm, so its fraction explained is undefined")
    c = nnls(D, y)
    return 1 - float((y.double() - D.T.double() @ c.double()).pow(2).sum()) / yy, c


def adapter_config(root: Path, name: str) -> dict:
    """The released adapter config, for the card and for checks that read it directly."""
    sub = root / ("ar_ptag_pooled/lora" if name == AR_NAME else check_adapter(name)["dir"])
    return json.loads((sub / "adapter_config.json").read_text(encoding="utf-8"))
