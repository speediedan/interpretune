# Publishing a Non-Bundled Op Collection

This walkthrough shows how a third-party author publishes an interpretune op
collection of their own: a standalone Hub repo with op definitions, an
implementation module, and tests — depending on interpretune's seams rather
than vendoring its code. It is written against the one collection built that
way in practice, `speediedan/jlens_steering_ops` (private until the Hub
registration flip): every pattern below names the file that demonstrates it.

For the op-authoring rules themselves (schemas, backends seam, persistence),
see the Custom Ops Composition Guide. For the trust posture consumers opt
into, see Hub Trust Posture. This page is the publishing half: repository
shape, manifest, tests, upload, and versioning.

## Why non-bundled matters

The seed collections (`speediedan/trivial_op_repo`,
`speediedan/concept_direction_ops`) mirror bundled ops: every op in them also
exists in the wheel. A mirror proves the machinery runs; it cannot show the
distinction a reviewer weighs — "a collection interpretune ships" versus "a
collection someone published". A genuinely non-bundled collection, whose ops
exist nowhere else, is the only exhibit that carries that claim.

## Repository shape

Five paths, each load-bearing:

```
jlens_steering_ops/
  it_component.yaml   # manifest: kinds, collection identity, op index
  jlens_ops.yaml      # op definitions: schemas, traits, implementations
  jlens_ops.py        # the implementation module
  README.md           # measured claims with their provenance, nothing pending
  tests/test_jlens_ops.py
```

Nothing else ships. In particular there is no packaging metadata, no
notebooks, and no vendored copy of anything interpretune already provides.

## The manifest

`it_component.yaml` declares `kinds: [ops]` and a `collection` block naming
the contract set:

```yaml
collection:
  name: jlens_steering_ops
  version: 0.3.0
```

The version tracks the op CONTRACT SET (names, schemas, traits), not code
churn: a consumer pins a revision for execution and reads the version to judge
compatibility. No `requires:` floor is declared, deliberately — a
`>=0.1`-style floor does not match a `0.1.0.devN+g<sha>` source install, so a
naive window silently skips the whole collection in any dev checkout.

## Op definitions

Each entry in `jlens_ops.yaml` carries four things: a `description` (plain
prose — quoted descriptions once rendered into an unparseable op cache, so the
file avoids double quotes by convention), an `implementation` naming a
`module.function` pair resolved repo-relatively by the Hub loader, an input
schema with dtypes and required flags, and the traits a consumer checks
WITHOUT executing anything. The worked example is
`required_intervention_modes: [patch]`: the op stages a patch-mode
intervention, and a backend that cannot express the mode refuses at staging
time rather than after every earlier op in a composition has already run.

## Implementation conventions

The impl module imports shared behavior from `interpretune.analysis.optools`
rather than reimplementing it: the unembed/norm seam, the readout helpers,
and — since the lens-layout work — Hub lens resolution itself, which probes
the repository layout and verifies the match instead of formatting one path.
Two rules earned the hard way travel with that dependency:

- A copy here would mean fixing bugs twice and drifting again: the module's
  own history records a private norm-folding copy that chose the wrong
  convention for two model lines, fixed once in the seam.
- Anything the module cannot vouch for is refused by name (unknown lens
  layouts, backends lacking a mode, empty token groups), never defaulted.

## Offline tests

`tests/test_jlens_ops.py` runs without network, weights, or a GPU. The
pattern: load the collection through `IT_ANALYSIS_OP_PATHS` — the path a user
actually takes — with the `interpretune` modules evicted from `sys.modules`
first so the test cannot pass on a stale import, stage synthetic fixtures
(a tiny lens checkpoint, a capturing backend), and assert staged values,
refusals, and algebraic identities (reject as the staged twin of patch,
scale behavior). Hub-layout variance (filenames that do not track the model
id, sparse checkpoints) is covered by construction through the shared
resolution seam plus a sparse-checkpoint unit test, not by network fixtures.

## Publishing

Publish with the op-collection preset, which excludes bytecode and tool
caches from the upload:

```python
from interpretune.hub.manager import HubAnalysisOpManager

manager = HubAnalysisOpManager(token=... )
manager.upload_ops(local_dir, "org/name", commit_message="...")
```

Publish private first. A private collection is reviewable (Hub reviewers see
it) without being consumable, which is the state the flip wants: the
registration PR cites a worked example whose contents are settled but whose
audience is not yet everyone.

## Versioning and pins

Consumers pin revisions, not versions:

```python
it.hub.pull_ops("speediedan/jlens_steering_ops", revision="<sha>")
```

Trusted Hub code must not change under the consumer, so demo notebooks pin
the revision they validated against and the trust opt-in (`IT_TRUST_REMOTE_CODE`)
is set explicitly at the pull site. Republishing is a pin workflow (publish,
re-pull at the revision, re-validate), and the README records measured claims
with the issue or commit that grounds them — never a pointer to an open
investigation, whose resolution would silently date the claim.
