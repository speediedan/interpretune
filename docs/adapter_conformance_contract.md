# Adapter conformance contract

An adapter composition is *conformant* when it runs Interpretune's conformance suite against itself and every
case its declarations select passes. The suite ships in the wheel as `interpretune.testing.conformance`, so a
repository runs it at the Interpretune version it targets, with no vendored copy and no peer backend.

## Why a suite, and why it is distributed

Interpretune's own backend tests compare bundled backends to each other. That shape cannot travel: a
repository testing one adapter has no peer installed, and comparing against whatever happens to be installed
cannot distinguish "both correct" from "both wrong in the same way". The suite therefore uses oracles that
need no peer, chosen per capability:

| oracle | proves | applies to |
| --- | --- | --- |
| the HuggingFace forward, hooked with plain PyTorch, never a participant | the backend observes and edits the same forward | forward family `hf_native` |
| causal and internal-consistency invariants, each with a positive control | the operations have the semantics their names claim | every family |
| structural conformance over `ITSession -> AnalysisRunner -> AnalysisStore` | the adapter is usable through the runner, not only correct in isolation | every family |

There are no published expected-result datasets. A live HuggingFace forward on `gpt2` costs seconds and
cannot be wrong in the same way as the thing it checks; a versioned artifact can.

## Installing

```bash
pip install "interpretune[conformance]"   # pytest, evaluate, scikit-learn: what the suite's session reaches
```

The suite refuses to build a session, by name, when those are missing; without that check a clean install
presents as a page of setup errors (measured on the first adopter) rather than as a dependency.

## What a repository writes

Two things: how to build a session config for its composition, and one class.

```python
# conftest.py
pytest_plugins = ["interpretune.testing.conformance.plugin"]

# tests/test_conformance.py
from interpretune.testing.conformance import ConformanceInputs, ConformanceTarget, ModelBackendConformance


def build_session_cfg(inputs: ConformanceInputs):
    from my_adapter import MyAdapterConfig

    return inputs.session_cfg(("core", "my_adapter"), module_cfg_extras={"my_adapter_cfg": MyAdapterConfig()})


class TestMyAdapterConformance(ModelBackendConformance):
    target = ConformanceTarget(
        composition=("core", "my_adapter"),
        session_cfg_factory=build_session_cfg,
        forward_family="hf_native",
        load=load_my_component,  # pull or stage the hub component and register it; None for a bundled adapter
    )
```

The suite owns the inputs (`gpt2`, a fixed prompt list, two batches of the `rte` seed dataset, the op set per
gate, device and precision, tolerances), the oracles, case selection and the report. The default datamodule
flavour (`"hf"`) is adapter-free: the seed's standalone datamodule plus a core-only module config, so it hydrates
on a bare core install; a hub adapter should not need TransformerLens to run the suite. The target owns its
session config, its runtime declarations, its forward family, and how its component is loaded.

## How cases are selected

The gate is what the composed module's backends **declare at runtime**: `capabilities`, and the support
records (`intervention_support`, `latent_model_support`). A declared surface runs its cases. An undeclared
surface runs the refusal case instead, so declaring less is never a way to pass more. A run in which no
gated case executed fails: a suite that proved nothing must not read as green.

Every case constructs its inputs the way a caller does and goes through the runner. Hand-built payloads
that reach a backend directly are the tempting isolation, and they are how a scope-dropping canonicalization
once survived a parity suite that tested the primitive and the engine but not the path between them.

## The cases

Always on:

| case | oracle | asserts |
| --- | --- | --- |
| session composes, declarations coherent | structural | a backend is attached; each declared capability satisfies its protocol; a support record is present iff its surface is declared |
| undeclared capabilities are refused by name | structural | the shared gate refuses every surface the backend does not claim, naming the backend and what it does claim |
| the runner produces the store schema | structural | `logit_diffs_base` yields the declared columns, one row per batch, declared dtypes |
| the cache op stores logits and every requested point | structural | `answer_logits` is not `None` from the cache path; every requested point is a tensor in the store |
| a block's output is the next block's input | causal | `blocks.L.hook_out` equals `blocks.L+1.hook_in` exactly; the layer index and slot mean what they say |
| capture converges on the forward | HF reference, `hf_native` | every captured point matches the HF module's tensor on real positions |
| answer logits converge on the forward | HF reference, `hf_native` | the store's logits match the HF forward on real positions |
| the scope discriminator tells the scopes apart | positive control | on the HF model alone, a last-token edit moves exactly the final position and a whole-prompt edit moves all of them |

Gate `INTERVENTION` (with `intervention_support`):

| case | oracle | asserts |
| --- | --- | --- |
| last-token scope moves exactly the final position | causal | the changed-position **set** from `intervention_position_effect` is `{seq_len - 1}` |
| all-positions scope moves every real position | causal | every real position is in the changed set |
| an undeclared scope is refused | negative | `NotImplementedError` naming the scope, before the backend is reached |
| undeclared modes are refused on the mode axis | negative | each mode not in `modes` raises, naming the axis |
| zero intervention is identity | causal | scale 0 leaves the logits unchanged |
| the baseline is an unsteered forward | HF reference, `hf_native` | the pre-intervention half equals the plain forward |
| steered logits converge on the forward | HF reference, `hf_native` | adding a vector at the last token matches an HF hook doing the same |

`LATENT_MODELS`, `GRADIENTS` and the analysis-backend gates (`ATTRIBUTION_GRAPH`, `FEATURE_INTERVENTION`)
have their cases listed on the tracking issue and land as the suite grows; a repository declaring them runs
whatever exists at its Interpretune version.

## Two rules for oracles, learned the expensive way

**Never coerce before comparing.** A parity test that normalized backend output with
`o.logits if hasattr(o, "logits") else o` accepted both shapes, and so concealed that a backend returned an
output object where the protocol declares a tensor, and `None` from a cache path that had never once produced
logits. A case asserts the declared return type and reads the value from it.

**No `getattr(..., default)` on a backend's return path.** The default is what converts a missing attribute
into a plausible value. A missing attribute is an error.

**The fixture is asserted before the sets are compared.** Below two real positions the two scopes both edit
`{0}`, so a one-token input would pass a backend that ignores scope entirely. The scope cases therefore fail
by name on any batch with fewer than two real positions rather than reporting a green that measured nothing.
The discriminating power lives in the input, and an adopter-supplied session config could shorten it.

## Single-prompt backends

Some engines take one prompt at a time and refuse a batch, deliberately. A loop over rows is **not** a
transparent implementation of the batched contract: in a batched forward, left-padded rows keep their padded
positions, so per-row unpadded forwards legitimately diverge from the batched reference. The contract therefore
carries the limit rather than hiding it: a target declares `batch_size=1`, every case then runs unpadded at one
row per batch, and one more case asserts that a batch above the declared limit is refused by name (never
processed as row 0, never looped silently).

Four facts about composing into a real session, learned by the first hub adapter on first contact with
`ITSession` (none of them reachable by a test that hosts the module class by hand):

- A backend wrapper that is not an `nn.Module` cannot take the `module.model` slot (assigning it raises inside
  `nn.Module`). Ops then hand the backend `module.model`, the raw model, so such a backend must carry its own
  wrapper handle rather than expect to receive it.
- At the time `post_auto_model_init` runs the tokenizer lives on the datamodule, not on the module.
- A composition must be registered for both component keys, `module` and `datamodule`; registering only the
  module half passes every module-level test and fails at `ITSession` with an error that names a missing adapter.
- Core resolves a `names_filter` list into a callable at setup, so a backend receives a predicate over hook
  names even when the caller wrote a list. A backend whose inventory is only partly expressible in that
  vocabulary applies the predicate over the expressible part; a predicate written in TL names cannot match a
  point that has no TL name, so that is complete rather than an under-capture.

## Four more facts from the first adoption at the default batch size

- **Pass the attention mask to the engine.** A forward run without it attends to pad positions; on gpt2 a
  two-row left-padded batch diverged from the masked HF forward by 80 in logits and 45 in a captured residual,
  while the unpadded row was correct either way. That is exactly why a single-row target cannot see it, and why
  the padded default is the stronger validation.
- **The oracle is HF batched with the mask, never per row.** A padded row batched does not equal that row run
  alone even with the mask (88 on gpt2), because absolute position embeddings shift under left padding. Every
  HF-native backend shares that behaviour.
- **Intervention specs may arrive as dicts through the store**, not only as `InterventionSpec` tuples; a
  backend must read both shapes (the bundled circuit-tracer backend does).
- **Enumerate hook inventories in Interpretune's vocabulary.** Core resolves a `names_filter` list into a
  predicate over Interpretune's hook names, and Interpretune prefers the bridge spellings (`hook_out`) where an
  engine may hold a legacy one (`hook_resid_post`). An adopter who enumerates from their own engine's spellings
  gets an empty selection, and nothing in the failure says "spelling". Enumerate
  `HookNameResolver(architecture).supported_hooks`, layer-expanded, and refuse a filter that selects a name the
  backend cannot map rather than capturing partially.
- **Refusals surface wrapped.** A refusal raised inside the runner reaches the caller as `datasets`'
  `DatasetGenerationError` with the refusal as its cause; the suite's refusal cases walk the cause chain.

## Four notes for adopters' own tests

**Measure in the environment your CI builds, not the one you developed in.** The first adopter's suite ran
13 of 14 in a development venv that had accumulated packages, and hit two undeclared-dependency walls of sixteen
setup errors each in a venv built the way its CI builds one. Neither was visible from the accumulated venv.
Build the clean environment once before calling the suite adopted.

**Derive complements; never transcribe the vocabulary.** A test that lists the capabilities a backend does
NOT claim by name goes red the moment the enum changes, with no opinion about the change. Derive the
complement (`set(BackendCapability) - backend.capabilities`) so a member added upstream is asserted absent
automatically and a removed one simply disappears. The suite's own refusal case is written that way.

**Select compositions by both keys, not by adapter set.** A helper that picks the composition class by matching
the adapter tuple alone starts returning the datamodule class the moment the datamodule composition is registered
too, and the failure points at `nn.Module` rather than at the selector. Match on `(component_key, adapters)`.

**Support records coerce.** `InterventionSupport(position_scopes={"last_token"}, modes={"add"})` is valid:
the record normalizes strings to the enums and refuses an empty set. The gates read the declared values
rather than checking the record's type, so a record from a second load of the enum module still passes.

## Padding

The bridge's activations on a left-padded batch match a plain HuggingFace forward given the attention mask
and **no** invented position ids (measured on `gpt2`: last token 9e-5, all real positions 4.6e-4). Pad
positions are undefined by contract and never compared. Value cases on padded dataset batches therefore use a
stated looser tolerance (relative 1e-3 plus absolute 1e-3) on real positions, because the deepest points (the
unembed input, the logits) accumulate platform-dependent drift: a Windows CPU runner exceeded 1e-3 absolute at
those two points while every shallower point passed. The single-prompt calibration runs at the tight one.
A failure reports the greatest absolute and relative differences so the next platform carries numbers.

## The report

Every run ends with the selection report: declared surfaces, cases run, cases skipped because their gate was
undeclared, cases skipped for any other reason, failures. Set `IT_CONFORMANCE_STRICT=1` in CI to turn the
"other reason" skips into a failure, so a missing dataset or model cannot quietly shrink the run.
