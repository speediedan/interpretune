# Looped lens for Inkling 2 Mini

A lens reads an activation inside a model and says, in tokens, what the model is computing there. Inkling 2 Mini runs
its 12 core layers (6 to 17) three times with shared weights, so a read location is a visit: a weight layer and, in
the core, a pass. This component reads a lens at one position per prompt and returns $N \times K$ tokens labelled by
visit, the output format of NDIF's Looped Lens Challenge.

## What is in it

| file | what it is |
| --- | --- |
| `books/*.yaml` | the lenses, as [venngram](https://github.com/speediedan/venngram) address books: `sets` address the read sites on venngram's Inkling 2 Mini map, and a `lens` section lists the readout at each site, $K$ and the list filters. A book is a plain document: anyone can resolve it against the map without this code |
| `looped_lens_runtime.py` | the runtime: book to site table (venngram), local reads by forward hooks (torch and transformers), remote reads on NDIF (the nnsight client) |
| `looped_lens_ops.py`, `looped_lens_ops.yaml` | the `looped_lens_read` op (alias `looped_lens`) |
| `ARTIFACTS.md` | the checkpoint head and the trained translators: what they are and where they come from |

## Readouts

At a visit: `active` (the active block, decoded as the model decodes its final state), `moe_in` (the MoE sublayer's
input), `final_input_mix` (the model's final block mix applied to the blocks completed at the visit), `coda` (the
visit's state finished through the coda layers 18 to 23), and `translator:final` / `translator:near` (a trained
per-visit affine translator, then the decode).

On NDIF, a coda readout from a core visit at layers 12 to 17 runs as an extra batch row of the same prompt. The coda's
entering state is layer 17's pass-3 output and its third completed block is layer 11's pass-3 output, so patching those
two to the visit's states reproduces the coda run from the visit inside the model's own forward. Coda readouts from
earlier visits complete fewer blocks than the coda receives and are refused remotely, by name.

## The lenses

Scores are on the render-gated WorkspaceBench single-token families (basic readout, multihop, multilingual, typo;
224 development and 33 held-out items), under a majority proxy for a blind judge: the gold must be named at more of
the lens's sites than any one distractor. Two selection criteria choose the configuration: item cross-validation
(every family present when sites are chosen) and leave one family out (an estimate of transfer to an unseen family).

| book | sites | $K$ | item CV | leave one family out | held out | runs on |
| --- | --- | --- | --- | --- | --- | --- |
| `inkling2_mini_icv_coda_remote_k10_n6` | coda at layers 15 and 14 (pass 3); active at 19, 22, 21; MoE input at 18 | 10 | 0.954 | 0.805 | 31 of 33 (30 of 33 end to end on NDIF) | NDIF (3 rows per prompt) or local |
| `inkling2_mini_icv_nocoda_k10_n6` | active at 21, 19 and layer 15 pass 3; MoE input at 19, 23, 18 | 10 | 0.928 | 0.854 | 29 of 33 | NDIF or local |
| `inkling2_mini_nocoda6` | active at 21, 19, 22 and layer 11 pass 2; MoE input at 18; final mix at 21 | 5 | 0.860 | 0.860 | 31 of 33 (31 end to end on NDIF) | NDIF or local |
| `inkling2_mini_icv_coda_k10_n8` | the first book's sites plus coda at layer 13 (pass 3) and layer 9 (pass 2) | 10 | 0.953 | 0.805 | 31 of 33 | local only |

Multihop items (where the hidden thing is a bridge the model never writes) separate the criteria: item
cross-validation picks mid-loop coda sites that only multihop needs (multihop 0.90 to 1.00 against 0.63 to 0.67 for the
leave-one-family-out picks).

## Use

```python
import interpretune as it  # with this directory on IT_ANALYSIS_OP_PATHS
from interpretune.analysis.ops.base import AnalysisBatch

# `module.model` is Inkling 2 Mini (HF, or an nnsight wrapper for NDIF); `batch` carries input_ids, attention_mask
out = it.looped_lens_read(
    module, AnalysisBatch(), batch, 0,
    looped_lens_table="books/inkling2_mini_icv_coda_remote_k10_n6.yaml",
    looped_lens_map="<venngram checkout>/inkling2_mini.yaml",
)
out.looped_lens_labels  # ['L15p3:coda', 'L19:active', ...]
out.looped_lens_tokens  # per prompt, per site: K tokens
```

For NDIF, pass `looped_lens_remote=True` and `looped_lens_head` (the checkpoint's final norm and unembedding; see
`ARTIFACTS.md`): states come back from one trace per prompt and are decoded locally.

## Limits

- Read position: one per prompt (default the last token).
- Locally the model is loaded in bf16; NDIF serves the same weights, and local and NDIF reads agree at top-1 about 0.95
  per site (the same agreement NDIF has between batched and single requests).
- The selection data is WorkspaceBench's single-token families gated for this model. On its poetry family (the rhyme
  word planned at the end of line one), every lens is near the floor: in plain text, Inkling 2 Mini's second line
  rarely settles on one rhyme.

License: Apache-2.0, as the model.
