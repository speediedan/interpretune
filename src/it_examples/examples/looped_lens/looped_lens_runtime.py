"""The looped lens runtime for Inkling 2 Mini: a prompt in, the top tokens at each of a lens's read sites out.

A lens is a venngram address book (``books/*.yaml``): its ``sets`` address the read sites on venngram's Inkling 2 Mini
map (``point:state.active & layers:15 & passes:3``), and its ``lens`` section lists what is read at each site, how many
tokens to keep ($K$) and the list filters. :func:`resolve_book` turns a book into a site table (module, call index,
input or output, value path, visit), so the runtime itself needs only torch and transformers (locally) or the nnsight
client (on NDIF), not venngram.

Readouts at a visit (a core layer runs three passes, so a visit is a weight layer and a pass):

- ``active``: the active block (``state.active``), decoded as the model decodes its final state (RMS norm with the
  checkpoint's gain, then the unembedding; the softcap is monotone, so it is skipped for ranking);
- ``moe_in``: the MoE sublayer's input (``mlp.mix``; on core visits of passes 2 and 3 the gated message input);
- ``final_input_mix``: ``final_input`` applied to the blocks completed at the visit and the active block;
- ``coda``: the visit's state finished through the coda layers (18 to 23), ``final_input`` and ``norm``; on NDIF, for a
  core visit at layers 12 to 17, as an extra batch row of the same prompt with layer 17's pass-3 output (and, for
  passes 1 and 2, layer 11's) patched to the visit's states;
- ``tl_final`` / ``tl_near``: a trained per-visit affine translator on the active block, then the decode.

Anything outside these, or a readout with no path on the chosen backend, is refused by name.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
from typing import Any

import torch

UNPADDED_VOCAB = 200058
LOOP = (6, 18)  # core layers [6, 18) run three passes
CODA_LAST, CODA_THIRD = 17, 11
READOUT_POINTS = {
    "active": ["state.active"],
    "final_input_mix": ["state.enter", "state.completed", "state.active"],
    "coda": ["state.enter", "state.completed", "state.active"],
    "tl_final": ["state.active"],
    "tl_near": ["state.active"],
}
BOOK_READOUTS = {"plain", "final_input_mix", "coda", "translator:final", "translator:near"}
REMOTE_READOUTS = {"active", "final_input_mix", "tl_final", "tl_near", "moe_in", "coda"}
FILTERS = {"content", "echo"}
# Where a site is read in a prompt: at the read position, or at the end of the question (the last token containing "?"
# in the user's turn; the read position when there is none). The Looped Lens Challenge reads chat questions at
# <|message_model|>, where the model is about to format its answer; at the question's end it still holds the answer.
POSITIONS = {"read", "question_end"}
WORD = re.compile(r"\w", re.UNICODE)
TOPK = 100


# ---------------------------------------------------------------------------------------------------------------------
# lens books -> site tables
# ---------------------------------------------------------------------------------------------------------------------
def resolve_book(book_path: str | pathlib.Path, map_path: str | pathlib.Path) -> dict:
    """Resolve a lens book against venngram's Inkling 2 Mini map into the site table the runtime reads."""
    import yaml

    # imported here because only resolving a book needs venngram (a resolved site table does not)
    from venngram.book import AddressBook
    from venngram.maps import load_component_map

    book = AddressBook.from_file(str(book_path))
    cmap = load_component_map(str(map_path))
    # A loop's visits are addressed through the map's execution section, which venngram before its Inkling map (main
    # at c49e5f58) does not read; without it no visit can be told from another.
    execution = getattr(cmap, "execution", None)
    if execution is None:
        raise ValueError(
            f"{map_path}: venngram read no execution map, so a lens book's visits (layer and pass) cannot be resolved;"
            " resolving a book needs a venngram with execution maps (main at c49e5f58 or later), or pass a resolved"
            " site table (.json) instead"
        )
    book.check_applies(architecture=yaml.safe_load(pathlib.Path(map_path).read_text())["architecture"])
    lens = yaml.safe_load(pathlib.Path(book_path).read_text()).get("lens")
    if not lens:
        raise ValueError(f"{book_path}: no `lens` section (K, filters, readouts)")
    sets = book.sets()

    def one(address: str, sets_=None):
        ss = execution.resolve(address, sets=sets_)
        if len(ss) != 1:
            raise ValueError(f"{address!r} resolves to {len(ss)} sites; a lens readout reads exactly one visit")
        return ss[0]

    def row(s, point: str) -> dict:
        path = [q if isinstance(q, int) else {"axis": q.axis, "index": q.index} for q in s.path]
        return {
            "point": point,
            "module": s.module,
            "call_index": s.call_index,
            "hook": s.hook,
            "path": path,
            "layer": s.visit.layer,
            "pass": s.visit.pass_number,
            "step": s.visit.step,
        }

    table = []
    for r in lens["readouts"]:
        kind = r["readout"]
        if kind not in BOOK_READOUTS:
            raise ValueError(f"readout {kind!r} is not one this lens implements: {sorted(BOOK_READOUTS)}")
        if r["set"] not in sets:
            raise ValueError(f"readout names set {r['set']!r}, which the book does not define")
        position = r.get("position", "read")
        if position not in POSITIONS:
            raise ValueError(f"readout position {position!r} is not one this lens reads: {sorted(POSITIONS)}")
        s = one("@" + r["set"], sets)
        layer, pas = s.visit.layer, s.visit.pass_number
        visit = f"layers:{layer}" + (f" & passes:{pas}" if LOOP[0] <= layer < LOOP[1] else "")
        tag = f"L{layer}" + (f"p{pas}" if LOOP[0] <= layer < LOOP[1] else "")
        if kind == "plain":
            internal = {"state.active": "active", "mlp.mix": "moe_in", "message.gated.mlp": "moe_in"}.get(s.point)
            if internal is None:
                raise ValueError(
                    f"set {r['set']!r} addresses {s.point}; `plain` reads state.active, mlp.mix or message.gated.mlp"
                )
            reads = {("state.active" if internal == "active" else s.point): row(s, s.point)}
        else:
            if s.point != "state.active":
                raise ValueError(f"{kind} is derived at a visit: set {r['set']!r} must address state.active")
            internal = {"translator:final": "tl_final", "translator:near": "tl_near"}.get(str(kind), str(kind))
            reads = {pt: row(one(f"point:{pt} & {visit}"), pt) for pt in READOUT_POINTS[internal]}
        table.append(
            {
                "label": r.get("label", f"{tag}:{internal}"),
                "readout": internal,
                "layer": layer,
                "pass": pas,
                "reads": reads,
                "set": r["set"],
                "position": position,
            }
        )
    spec = {
        "name": getattr(book, "name", pathlib.Path(book_path).stem),
        "model": lens.get("model", "thinkingmachines/Inkling-2-Mini"),
        "K": lens["K"],
        "variant": lens.get("variant", "plain"),
        "filters": lens.get("filters", []),
        "artifacts": lens.get("artifacts", {}),
        "sites": [t["label"] for t in table],
        "book": str(book_path),
    }
    check_spec(spec)
    return {"spec": spec, "map": str(map_path), "sites": table}


def load_table(path: str | pathlib.Path, map_path: str | pathlib.Path | None = None) -> dict:
    """A site table from a resolved table (``.json``) or a lens book (``.yaml``, resolved against ``map_path``)."""
    path = pathlib.Path(path)
    if path.suffix in (".yaml", ".yml"):
        if map_path is None:
            raise ValueError(f"{path} is a lens book: resolving it needs venngram's Inkling 2 Mini map (map_path)")
        return resolve_book(path, map_path)
    table = json.loads(path.read_text())
    check_spec(table["spec"])
    return table


def check_spec(spec: dict) -> None:
    unknown = set(spec.get("filters", [])) - FILTERS
    if unknown or spec.get("variant", "plain") != "plain":
        raise ValueError(
            f"variant {spec.get('variant')!r} / filters {sorted(unknown)} are not implemented here "
            "(variant `plain`; filters `content`, `echo`)"
        )


def question_end(ids: list[int], read_position: int, tokenizer: Any, user_marker: str = "<|message_user|>") -> int:
    """The last token containing ``?`` after the last user-turn marker and before ``read_position``; else
    ``read_position`` (a plain prompt, or a turn with no question mark)."""
    user = tokenizer.convert_tokens_to_ids(user_marker)
    starts = [i for i, t in enumerate(ids) if t == user]
    if not starts:
        return read_position
    marks = [i for i in range(starts[-1] + 1, read_position) if "?" in tokenizer.decode([ids[i]])]
    return marks[-1] if marks else read_position


def site_positions(table: dict, ids: list[int], read_position: int, tokenizer: Any) -> dict[str, int]:
    """Where each site of ``table`` is read in one prompt."""
    q = None
    out = {}
    for s in table["sites"]:
        if s.get("position", "read") == "question_end":
            q = question_end(ids, read_position, tokenizer) if q is None else q
            out[s["label"]] = q
        else:
            out[s["label"]] = read_position
    return out


# ---------------------------------------------------------------------------------------------------------------------
# lists: the N x K labelled tokens
# ---------------------------------------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return s.strip().lower()


def build_lists(top: torch.Tensor, vocab: list[str], prompt: str, k: int, filters: set[str]) -> list[list[str]]:
    """Per site, its first ``k`` decoded tokens after the filters: ``content`` drops tokens with no word character,
    ``echo`` drops tokens the prompt contains as a word."""
    words = {_norm(w) for w in re.findall(r"\w+", prompt)}
    out = []
    for s in range(top.shape[0]):
        keep: list[str] = []
        for t in top[s].tolist():
            if t < 0:
                break
            tok = vocab[t]
            if "content" in filters and not WORD.search(tok):
                continue
            if "echo" in filters and _norm(tok) in words:
                continue
            keep.append(tok)
            if len(keep) == k:
                break
        out.append(keep)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# a store of reads -> the challenge's readouts
# ---------------------------------------------------------------------------------------------------------------------
_LABEL = re.compile(r"^L(\d+)(?:p([123]))?(?::|$)")
MAX_CHALLENGE_LAYERS, MAX_CHALLENGE_TOKENS = 10, 10


def visit_index(label: str) -> int:
    """The layer visit (0 to 47, in the order they run) a read-site label names: ``L15p3:coda`` is 39,
    ``L19:active`` 43.

    A core layer (6 to 17) runs three passes, so its label must carry one; a label that names no visit is refused.
    """
    m = _LABEL.match(label)
    if m is None:
        raise ValueError(f"{label!r} names no layer visit (expected L<layer>, or L<layer>p<pass> for a core layer)")
    layer, pas, (start, end) = int(m[1]), m[2], LOOP
    core = start <= layer < end
    if core != (pas is not None):
        raise ValueError(f"{label!r}: a core layer ({start} to {end - 1}) needs a pass, and no other layer has one")
    if layer >= 24:
        raise ValueError(f"{label!r}: layer {layer} is not a decoder layer (0 to 23)")
    if core:
        return start + (layer - start) + (end - start) * (int(pas) - 1)
    return layer if layer < start else layer + 2 * (end - start)


def challenge_readouts(rows: Any, ids: list[str], *, allow_several_positions: bool = False) -> list[dict]:
    """Stored ``looped_lens_read`` rows as the Looped Lens Challenge's readouts file, one dict per question.

    ``rows`` is anything that yields the op's rows: an :class:`AnalysisStore` (iterated, so a streamed one works) or a
    list of row dicts. Each row is one op call and may hold several prompts; ``ids`` are the questions' ids in prompt
    order, and their number must match. Each readout is ``{"layer": visit, "tokens": [...]}`` for one read site, at most
    ten sites and ten tokens per site, as the challenge allows.

    The challenge takes a prompt's readouts at a single token position, so a prompt whose sites were read at more than
    one (a book with ``question_end`` sites) is refused, unless ``allow_several_positions``.
    """
    out: list[dict] = []
    for row in rows:
        labels = [str(x) for x in row["looped_lens_labels"]]
        visits = [visit_index(label) for label in labels]
        if len(set(visits)) != len(visits) or len(visits) > MAX_CHALLENGE_LAYERS:
            raise ValueError(
                f"the challenge takes at most {MAX_CHALLENGE_LAYERS} layers per question, one readout each: {labels}"
            )
        if "looped_lens_site_positions" not in row:
            raise ValueError("the rows have no looped_lens_site_positions, so where their sites were read is unknown")
        prompts, where = row["looped_lens_tokens"], [[int(p) for p in w] for w in row["looped_lens_site_positions"]]
        if len(where) != len(prompts):
            raise ValueError(f"read positions for {len(where)} prompts, and token lists for {len(prompts)}")
        for prompt, at in zip(prompts, where):
            if len(prompt) != len(labels) or len(at) != len(labels):
                raise ValueError(f"{len(prompt)} token lists and {len(at)} positions for {len(labels)} read sites")
            if len(set(at)) > 1 and not allow_several_positions:
                raise ValueError(
                    f"a prompt's sites were read at {len(set(at))} token positions {dict(zip(labels, at))}; the "
                    "challenge takes a single position per prompt (allow_several_positions=True writes them anyway)"
                )
            sites = sorted(zip(visits, prompt), key=lambda item: item[0])
            if len(out) == len(ids):
                raise ValueError(f"the store holds more prompts than the {len(ids)} question ids given")
            readouts = [{"layer": v, "tokens": [str(t) for t in toks][:MAX_CHALLENGE_TOKENS]} for v, toks in sites]
            out.append({"id": ids[len(out)], "readouts": readouts})
    if len(out) != len(ids):
        raise ValueError(f"the store holds {len(out)} prompts for {len(ids)} question ids")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the coda mirror (a line-for-line mirror of the text model's coda, from a visit's state)
# ---------------------------------------------------------------------------------------------------------------------
def _mask_maps(tm, M, embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor):
    kw = {
        "config": tm.config,
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "cache": None,
        "position_ids": position_ids,
    }
    cmm = {
        "full_attention": M.create_causal_mask(**kw),
        "sliding_attention": M.create_sliding_window_causal_mask(**kw),
        "linear_attention": M.create_recurrent_attention_mask(**kw),
    }
    rdm = {
        "full_attention": M.get_relative_distance(embeds, None, is_sliding=False),
        "sliding_attention": M.get_relative_distance(embeds, None, is_sliding=True),
    }
    return cmm, rdm


def _step(layer, hidden, blocks, cmm, rdm, position_ids):
    """One layer as the model runs it: call, then (at a block boundary) append the ENTERING state."""
    kw = {
        "attention_mask": cmm[layer.attention_type],
        "relative_distance": rdm[layer.attention_type],
        "conv_mask": cmm["linear_attention"],
        "cache": None,
        "position_ids": position_ids,
    }
    new = layer(hidden, blocks, **kw)
    if layer.is_block_boundary:
        blocks = torch.cat([blocks.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
    return new, blocks


def coda_from(tm, M, hidden: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """Finish visit states (S, T, d) with their completed blocks (S, T, n, d) through the coda; the normed final
    input."""
    le = tm.config.loop_layer_index_end
    dev = next(tm.layers[le].parameters()).device
    hidden, blocks = hidden.to(dev), blocks.to(dev)
    am = torch.ones(hidden.shape[:2], dtype=torch.long, device=dev)
    pos = torch.arange(hidden.shape[1], device=dev).unsqueeze(0)
    cmm, rdm = _mask_maps(tm, M, hidden, am, pos)
    for layer in tm.layers[le:]:
        hidden, blocks = _step(layer, hidden, blocks, cmm, rdm, pos)
    return tm.norm(tm.final_input(blocks, hidden)[0]).to(hidden.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# local runtime
# ---------------------------------------------------------------------------------------------------------------------
def read_path(value, path: list):
    for p in path:
        value = value[p] if isinstance(p, int) else value.select(p["axis"], p["index"])
    return value


class LensRuntime:
    """Forward hooks at a site table's sites on a loaded Inkling 2 Mini, and the readouts."""

    def __init__(self, hf_model: Any, tokenizer: Any, table: dict, translators: dict | None = None) -> None:
        # Inkling 2 Mini's modeling code ships with transformers from 5.24
        from transformers.models.inkling2_loop import modeling_inkling2_loop as M

        self.hf, self.tok, self.table, self.M = hf_model, tokenizer, table, M
        spec = table["spec"]
        check_spec(spec)
        self.k, self.filters = int(spec.get("K", 3)), set(spec.get("filters", []))
        self.tm = hf_model.model.language_model
        self.dev = self.tm.norm.weight.device
        self.w_u = hf_model.lm_head.weight.detach()[:UNPADDED_VOCAB].to(self.dev)
        self.gain = self.tm.norm.weight.detach().float().to(self.dev)
        if any(s["readout"].startswith("tl_") for s in table["sites"]) and translators is None:
            raise ValueError("this lens reads a translator: pass the M2 translators (see ARTIFACTS.md)")
        self.translators = translators
        self._vocab: list[str] | None = None
        self.needs = collections.defaultdict(list)
        for s in table["sites"]:
            for point, r in s["reads"].items():
                self.needs[r["module"]].append((r["call_index"], r["hook"], r["path"], (s["label"], point)))
        self.handles = [hf_model.get_submodule(m).register_forward_hook(self._hook(m)) for m in self.needs]
        self.state: dict = {}

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    @property
    def vocab(self) -> list[str]:
        if self._vocab is None:
            self._vocab = [self.tok.decode([i]) for i in range(UNPADDED_VOCAB)]
        return self._vocab

    def _hook(self, module: str):
        def fn(mod, args, output):
            c = self.state["counts"][module]
            self.state["counts"][module] = c + 1
            for ci, hook, path, key in self.needs[module]:
                if ci == c:
                    self.state["got"][key] = read_path(args if hook == "input" else output, path)[0].detach()

        return fn

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.dev).float()
        x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return ((self.gain * x).to(torch.bfloat16) @ self.w_u.T).float()

    def readout(self, site: dict, pos: list[int]) -> torch.Tensor:
        g = {point: self.state["got"][(site["label"], point)] for point in site["reads"]}
        kind, layer, p = site["readout"], site["layer"], site["pass"]
        if kind == "active":
            return self.decode(g["state.active"][pos])
        if kind == "moe_in":
            return self.decode(next(iter(g.values()))[pos])
        if kind in ("tl_final", "tl_near"):
            assert self.translators is not None  # checked in __init__ for any translator site
            st, i = self.translators[kind[3:]], layer - LOOP[0]
            h = g["state.active"][pos].to(self.dev).float()
            t = h + h @ st["W"][i].to(self.dev).T + st["b"][i].to(self.dev) + st["c"][i, p - 1].to(self.dev)
            return self.decode(t)
        blocks, hidden, active = g["state.completed"], g["state.enter"], g["state.active"]
        if self.tm.layers[layer].is_block_boundary:  # the entering state completes a block, as the model does it
            blocks = torch.cat([blocks.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
        if kind == "final_input_mix":
            fin = self.tm.final_input
            fdev = fin.query.device
            mixed = fin(blocks[pos].unsqueeze(0).to(fdev), active[pos].unsqueeze(0).to(fdev))[0][0]
            return self.decode(mixed)
        if kind == "coda":
            normed = coda_from(self.tm, self.M, active.unsqueeze(0), blocks.unsqueeze(0))[0, pos]
            return (normed.to(self.dev).to(torch.bfloat16) @ self.w_u.T).float()
        raise ValueError(f"readout {kind!r} is not implemented")

    @torch.no_grad()
    def top_ids(self, input_ids: list[int], positions: list[int] | dict[str, int]) -> dict[str, torch.Tensor]:
        """Top 100 ids per site label, (len(positions), 100), from one forward pass of one prompt; ``positions`` is
        one list for every site, or one position per site label (:func:`site_positions`)."""
        self.state = {"counts": collections.Counter(), "got": {}}
        self.hf(input_ids=torch.tensor([input_ids], device=self.tm.embed_tokens.weight.device))
        missing = [key for reqs in self.needs.values() for *_, key in reqs if key not in self.state["got"]]
        if missing:
            raise RuntimeError(f"sites that never fired (a call index the forward did not reach): {missing[:4]}")

        def at(s):
            return [positions[s["label"]]] if isinstance(positions, dict) else positions

        return {s["label"]: self.readout(s, at(s)).topk(TOPK, dim=-1).indices.cpu() for s in self.table["sites"]}


# ---------------------------------------------------------------------------------------------------------------------
# remote runtime (NDIF, through the nnsight client)
# ---------------------------------------------------------------------------------------------------------------------
def check_remote(table: dict) -> None:
    for s in table["sites"]:
        if s["readout"] not in REMOTE_READOUTS:
            raise ValueError(f"{s['label']}: the {s['readout']} readout has no remote path")
        if s["readout"] == "moe_in" and "message.gated.mlp" in s["reads"]:
            raise ValueError(f"{s['label']}: the MoE input on a message visit has no remote path yet")
        if s["readout"] == "coda" and not 12 <= s["layer"] <= CODA_LAST:
            raise ValueError(
                f"{s['label']}: a remote coda readout needs a core visit at layers 12 to 17 (earlier "
                "visits complete fewer blocks than the coda receives)"
            )


def remote_collect(model, llm, table: dict, ids: list[int], pos: list[int] | dict[str, int]) -> dict[str, torch.Tensor]:
    """One NDIF trace of one prompt: each plain site's state at ``pos`` (float16), and each coda site's top-100 ids
    as an extra batch row patched from row 0 earlier in the same forward. ``pos`` is one list for every site, or one
    position per site label (:func:`site_positions`)."""
    check_remote(table)
    where = {s["label"]: ([pos[s["label"]]] if isinstance(pos, dict) else pos) for s in table["sites"]}
    sites = sorted(table["sites"], key=lambda s: next(iter(s["reads"].values()))["step"])
    coda = [s for s in sites if s["readout"] == "coda"]
    visits: dict[tuple[int, int], list[dict]] = {}
    for s in sites:
        if s["readout"] != "coda":
            visits.setdefault((s["layer"], s["pass"]), []).append(s)
    coda_vis = [(s["layer"], s["pass"]) for s in coda]
    keep = set(coda_vis) | {(CODA_THIRD, p) for _, p in coda_vis if p != 3}
    pre = sorted(v for v in visits if v[0] < LOOP[0])
    post = sorted(v for v in visits if v[0] >= LOOP[1])
    x = torch.tensor([ids] * (1 + len(coda)))
    with model.trace(x, remote=True) as tracer:
        got = list().save()  # pyright: ignore[reportAttributeAccessIssue]  # nnsight's saved list
        saved: dict = {}

        def visit(layer_i: int, p: int) -> None:
            v = (layer_i, p)
            ss = visits.get(v, [])
            patch_third = bool(coda) and p == 3 and layer_i == CODA_THIRD and any(q != 3 for _, q in coda_vis)
            patch_last = bool(coda) and p == 3 and layer_i == CODA_LAST
            if not ss and v not in keep and not patch_third and not patch_last:
                return
            kinds = {s["readout"] for s in ss}
            layer = llm.layers[layer_i]
            hidden = blocks = moe = None
            if "final_input_mix" in kinds:
                args = layer.inputs[0]
                hidden, blocks = args[0], args[1]
            if "moe_in" in kinds:
                moe = layer.mlp_input.output[0][0]
            out = layer.output
            for s in ss:
                at = where[s["label"]]
                if s["readout"] == "moe_in":
                    assert moe is not None  # read above whenever a site at this visit is a moe_in readout
                    val = moe[at]
                elif s["readout"] == "final_input_mix":
                    assert blocks is not None and hidden is not None
                    b = blocks
                    if layer_i % 6 == 0:  # a block boundary: the entering state completes a block
                        b = torch.cat([blocks.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
                    val = llm.final_input(b, out)[0][0, at]
                else:
                    val = out[0, at]
                assert val is not None
                got.append(val.detach().to("cpu", torch.float16))
            if v in keep:
                saved[v] = out[0].clone()
            if patch_third:
                for r, (_, q) in enumerate(coda_vis, start=1):
                    if q != 3:
                        out[r] = saved[(CODA_THIRD, q)]
            if patch_last:
                for r, cv in enumerate(coda_vis, start=1):
                    out[r] = saved[cv]

        for v in pre:
            visit(*v)
        for p in tracer.iter[:3]:
            for layer_i in range(LOOP[0], LOOP[1]):
                visit(layer_i, p + 1)
        for v in post:
            visit(*v)
        coda_top = None
        if coda:
            logits = model.output.logits
            coda_top = (
                torch.stack(
                    [
                        logits[r, where[s["label"]]].float().topk(TOPK, dim=-1).indices
                        for r, s in enumerate(coda, start=1)
                    ]
                )
                .cpu()
                .save()  # pyright: ignore[reportAttributeAccessIssue]  # nnsight's saved value
            )
    order = [
        s["label"] for v in pre + [(i, p) for p in (1, 2, 3) for i in range(*LOOP)] + post for s in visits.get(v, [])
    ]
    states = dict(zip(order, list(got)))
    for r, s in enumerate(coda):
        assert coda_top is not None
        states[s["label"]] = coda_top[r]
    return states


def remote_top_ids(
    model,
    llm,
    table: dict,
    ids: list[int],
    pos: list[int] | dict[str, int],
    w_u: torch.Tensor,
    gain: torch.Tensor,
    translators: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Top 100 ids per site from one NDIF trace, the states decoded locally with the checkpoint's head."""
    states = remote_collect(model, llm, table, ids, pos)
    out = {}
    for s in table["sites"]:
        if s["readout"] == "coda":
            out[s["label"]] = states[s["label"]].long()
            continue
        h = states[s["label"]].float()
        if s["readout"].startswith("tl_"):
            if translators is None:
                raise ValueError("this lens reads a translator: pass the M2 translators (see ARTIFACTS.md)")
            st, i = translators[s["readout"][3:]], s["layer"] - LOOP[0]
            h = h + h @ st["W"][i].T + st["b"][i] + st["c"][i, s["pass"] - 1]
        x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        out[s["label"]] = ((gain * x).to(torch.bfloat16).float() @ w_u.T).topk(TOPK, dim=-1).indices
    return out
