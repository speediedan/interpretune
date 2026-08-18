# Trusting hub components: what you are consenting to run

Interpretune's point is running other people's analysis components. That makes "what am I
consenting to execute?" a question the framework owes you a clear answer to, before you load
anything.

**The short version:** Interpretune does not execute code from the Hub unless you say so. Data
travels freely; code does not. Opt in with `IT_TRUST_REMOTE_CODE=1` when you have decided a
publisher is worth trusting.

## What actually executes

Most of what you pull from the Hub is data, and data never triggers this gate: manifests
(`it_component.yaml`), configurations, cards, and artifact payloads (parquet, the
`it_artifact.json` envelope) are parsed, not run. You can pull a repo, read its manifest, and see
exactly which file it would execute without executing anything.

Two paths execute publisher-authored Python in your process, and both are gated:

| Path | What runs | Gate |
| --- | --- | --- |
| Analysis op collections | the op collection's module code, via the dynamic-module cache | discovery skips untrusted repos with a warning |
| Prompt-config entrypoints (`compose_ref`) | the entrypoint module the manifest names | `RemoteCodeNotTrustedError` |

The two behave differently on purpose. A session with no hub ops is still a working session, so op
discovery degrades to "load fewer things" rather than failing your first op access. A `compose_ref`
that cannot be resolved has no such fallback: the configuration you asked for cannot be built, so
it raises.

## The threat model

Once you opt in and a component's code runs, it is ordinary Python in your interpreter. It can:

- read and write anything your user account can, including your Hugging Face token and cached
  credentials, and send them anywhere your network allows;
- import anything installed in your environment, and patch or wrap it;
- persist beyond the load, since imported modules stay in `sys.modules` for the process lifetime.

There is no sandbox. The gate controls **whether** third-party code runs, not what it may do once
running. Treat trusting a component exactly as you would treat `pip install` from that publisher.

Two related surfaces worth knowing about, which the gate does not cover:

- **`class_path` in a hub configuration** names an importable object in *your* environment.
  Nothing new is downloaded, but a hostile configuration can select and call code you already have
  installed, with arguments it chooses. Read a configuration before instantiating from a publisher
  you do not trust.
- **Adapters** ([#125](https://github.com/speediedan/interpretune/issues/125)) are not shareable
  yet. When they are, they compose into your class hierarchy through the MRO, which is a stronger
  capability than an op collection's; whether they are held to a stricter bar than "the session
  opted in" is an open decision on that issue, deliberately not pre-empted here.

## The default, and how to opt in

`IT_TRUST_REMOTE_CODE` has three states, and the difference between "unset" and "off" matters:

| `IT_TRUST_REMOTE_CODE` | Meaning | Behavior |
| --- | --- | --- |
| unset | you have not decided | refuse, and explain how to decide |
| `1`, `true`, `yes` (any case) | opted in for this session | hub-resident code runs |
| any other value | opted out deliberately | refuse, quietly — the decision is made |

Opt in for a process:

```bash
export IT_TRUST_REMOTE_CODE=1
```

or from inside a running session, which is the common case in a notebook:

```python
import os

os.environ["IT_TRUST_REMOTE_CODE"] = "1"
```

The value is read each time the gate is consulted, never captured at import, so the notebook form
works even after `import interpretune`.

The opt-in is **session-scoped**: it covers every component you load in that process, not one repo.
Interpretune deliberately does not ship a per-repo allowlist yet — a list of trusted publishers
that is easy to append to and never reviewed is a worse guarantee than an explicit decision each
session. If you want per-repo granularity today, opt in around the specific load and unset it
afterward.

## Escape hatches

**Inspect before you trust.** Pulling caches a repo without executing anything, and the manifest
names the file that would run:

```python
import interpretune as it

manifest, commit = it.hub.pull("someorg/some-component")  # downloads; executes nothing
print(manifest["kinds"], manifest.get("promptconfigs", {}).get("entrypoint"))

# where the files landed, so you can read that entrypoint before opting in
_, snapshot, revision = it.hub.resolve_component_manifest("someorg/some-component")
print(snapshot)
```

**Pin a revision** so trusted code cannot change under you. Trust granted to a publisher is not
trust granted to every future commit they push; a pinned revision is the difference between
auditing something once and re-auditing it silently on every pull.

**Run without remote code at all.** Set `IT_TRUST_REMOTE_CODE=0`. Everything that does not require
executing publisher code keeps working: local modules and datamodules, hub-resident configurations
that name locally installed `class_path`s, artifact pull and hydration, and the whole analysis-op
surface that ships with interpretune. You lose hub op collections and `compose_ref` prompt configs.

## Bundled ops, hub ops, and which one you are running

Interpretune ships a set of **bundled** analysis ops, and bundled ops win their bare names. That
default is what makes a session behave the same offline as online, and the same for you as for a
collaborator who has pulled nothing. It also means **pulling an op collection cannot silently
re-point existing code**: a pull adds namespaced ops, it does not take over names.

Nothing is guessed, so nothing has to be inferred from behavior. Ask:

```python
import interpretune as it

print(it.hub.op_info("concept_direction"))
```

```text
'concept_direction' resolves to concept_direction [bundled, collection concept 0.1.0]
  also available:
    speediedan.concept_direction_ops.concept_direction [hub:speediedan.concept_direction_ops, collection concept_direction_ops 0.1.0, revision 16affe3811eb]
  precedence: none (bundled ops win bare names)
```

That reports the provenance of what is active, its declared collection and version, the cached
revision for a hub collection, every other definition sharing the bare name, and the precedence that
chose between them. The revision is read from the cache and never fetched: a lookup that could fetch
would change the answer while reporting it.

### Opting into a hub collection's bare names

```python
it.hub.prefer_ops("speediedan/concept_direction_ops")
```

Per namespace, explicit, and reversible — `it.hub.prefer_ops()` with no arguments clears it.
Precedence is applied when a name is resolved rather than by rewriting the op registry, so the
bundled definition is never evicted, only outranked. `IT_OP_PRECEDENCE="org/repo1,org/repo2"` is the
same opt-in, ordered, for runs with no place to call it.

**Fully-qualified names ignore precedence entirely**, in both directions. Explicit beats implicit,
so `speediedan.concept_direction_ops.concept_direction` addresses exactly that op whether or not its
collection is preferred. That is how you pin one specific copy in code that has to keep working
regardless of what precedence a session declares.

Precedence never bypasses the trust gate. An untrusted session has no hub collections loaded, so it
has none to prefer, and a preferred collection resolves at the revision its manifest pinned — so
"newer" is a deliberate pull, not drift.

## Collection versions and compatibility windows

An op collection declares its own identity and, optionally, one compatibility window against the
installed interpretune:

```yaml
collection:
  name: my_ops
  version: 0.3.0
  requires:
    interpretune: ">=0.1.0.dev0"
```

The version versions the op **contract set** — the names, schemas and traits the ops present to
callers — not a package. There is deliberately no cross-collection dependency resolution and no
solver: one window per collection, checked with the same `requires:` grammar and machinery component
manifests use. An incompatible collection is skipped **whole** with a warning, because compatibility
is declared once per collection and a partial load would present half a contract set. Set
`IT_STRICT_OP_LOAD=1` to make that a hard failure instead.

Two things worth knowing before you write a window:

- **Bundled families declare none.** They ship inside the wheel, so a window against the installed
  package is vacuous by construction.
- **A `>=0.1`-style floor does not match a source install.** `setuptools_scm` produces
  `0.1.0.devN+g<sha>` between tags, and a dev release sorts *before* its release under PEP 440, so
  that floor silently skips the whole collection in any checkout. Write `>=0.1.0.dev0` when you mean
  "0.1 or later, including pre-releases".

## For component authors

Publishing an entrypoint means asking your users for this consent, so keep the ask small: prefer
configuration over code, keep entrypoints self-contained and readable, and describe in your card
what the entrypoint does. A component whose executable surface a reader can check in a minute is
one they can actually decide about.

For an op collection specifically, `it_component.yaml` is what makes the repo well-formed:

```yaml
it_schema_version: 1
kinds: [ops]
ops:
  files:
    - my_ops.yaml
```

Op discovery is **manifest-routed**: the files listed there are the op definitions, and nothing else
in the repo is read as one. That is what lets a collection carry a card, a config sample or a
notebook without any of them reaching the op compiler. A repo with no manifest publishes fine and
then contributes no ops, and interpretune says so rather than guessing.
