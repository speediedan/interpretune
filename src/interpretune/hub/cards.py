"""Generated repo cards: the discovery sentinel, written at publish time, always.

Every repo interpretune publishes carries ``library_name: interpretune`` plus ``interpretune`` /
``interpretune-<kind>`` tags (and, for task components, a ``task:<name>`` tag and mirrored ``datasets:``
metadata). The card is generated — no publish path can produce a card-less repo, because
``library_name: interpretune`` is also the HF library-registration precondition and the tag is what
``list_models(filter="interpretune")`` discovery queries. (The pre-2026-08 op path shipped repos with no tags
at all, which made that discovery filter dead code; generation-at-publish fixes it by construction.)
"""

from __future__ import annotations

from huggingface_hub import ModelCard, ModelCardData

LIBRARY_NAME = "interpretune"


class ComponentCardError(ValueError):
    """A manifest cannot produce a coherent card (raised at publish, before anything is uploaded)."""


def component_card_metadata(manifest: dict, license_id: str = "apache-2.0") -> ModelCardData:
    """Build card metadata (library_name, tags, datasets) from a validated component manifest."""
    tags = [LIBRARY_NAME] + [f"{LIBRARY_NAME}-{kind}" for kind in manifest["kinds"]]
    datasets: list[str] = []
    task = (manifest.get("module") or {}).get("task") or {}
    if task.get("name"):
        tags.append(f"task:{task['name']}")
    for ds in task.get("datasets") or []:
        if ds.get("path"):
            datasets.append(ds["path"])
    return ModelCardData(
        license=license_id, library_name=LIBRARY_NAME, tags=tags, datasets=sorted(set(datasets)) or None
    )


def generate_artifact_card(envelope: dict, repo_id: str, summary: str | None = None) -> ModelCard:
    """Generate the card for an artifact (dataset) repo from its validated envelope.

    The card behavior is reimplemented once for BOTH repo types (§8): dataset repos carry the same
    ``library_name`` + tag sentinel so ``interpretune-analysis-store`` artifacts are discoverable.
    """
    kind = envelope["artifact_kind"]
    meta = ModelCardData(license="apache-2.0", library_name=LIBRARY_NAME, tags=[LIBRARY_NAME, f"{LIBRARY_NAME}-{kind}"])
    title = repo_id.split("/", 1)[-1]
    arts = envelope.get("artifacts") or {}
    prov = envelope.get("provenance") or {}
    lines = [f"# {title}", ""]
    lines.append(summary or f"An interpretune {kind} artifact ({title}).")
    lines.append("")
    lines.append(
        f"An [interpretune](https://github.com/speediedan/interpretune) artifact repo "
        f"(kind: {kind}; envelope schema v{envelope['schema']})."
    )
    lines += ["", "## Artifact", ""]
    lines.append(f"- split: `{arts.get('split')}`, rows: {arts.get('num_rows')}")
    if arts.get("columns"):
        lines.append(f"- columns: {', '.join(f'`{c}`' for c in arts['columns'])}")
    if prov.get("interpretune_version"):
        lines.append(f"- generated with interpretune `{prov['interpretune_version']}`")
    lines += [
        "",
        "Load with `interpretune.hub.pull_analysis_store(...)` — the interpretune formatter",
        "re-attaches from the `it_artifact.json` envelope; no pipeline re-run is required.",
    ]
    return ModelCard(f"---\n{meta.to_yaml()}\n---\n\n" + "\n".join(lines) + "\n")


def _adapter_card_sections(manifest: dict, source: str) -> list[str]:
    """The `adapters` kind's card sections: what it EXPOSES, declares, and composes.

    **Three blocks, not the five the design sketched, and the card now says so.** Capabilities and
    refusals live in the adapter's CODE, and the publisher never executes the entrypoint, so they are
    structurally unreachable from a manifest. Rendering them would require either executing hub-resident
    code at publish time -- which is what the trust gate exists to prevent -- or trusting an undeclared
    claim, which is strictly worse than an absent one: a manifest could assert a capability the code does
    not implement and the card would publish it unchallenged.

    Silently omitting them is its own failure, though. A card with no limits section reads as an adapter
    with no limits. So the card NAMES what it cannot report and points at where the answer actually lives.

    A model card describes weights, which are data. An adapter is code that runs in the caller's process,
    so this card carries a block a model card never needs — the trust posture — and it is the block this
    card most exists for. Capability is largely inferable from a manifest; exposure is not.

    **Provenance is stated rather than implied.** These sections render the VALIDATED MANIFEST. The
    publisher never executes the entrypoint (it stages and copies), so it cannot reconcile the declaration
    against what `register_adapter_ctx` actually registers — doing so would mean importing hub-resident code
    and its optional dependencies at publish time, which is precisely what the trust gate exists to prevent.
    The reconciliation that catches an overstating manifest is `load_hub_adapter`'s, at LOAD time, and it
    compares against the SATISFIABLE set rather than the declared one. The card says which of the two it is
    reporting so a reader is not left to assume the stronger one.
    """
    ad = manifest.get("adapters") or {}
    declares = ad.get("declares") or []
    comps = ad.get("compositions") or []
    lines: list[str] = ["", "## Adapters", ""]

    lines.append(
        "**This component executes code in your process.** Loading it runs the entrypoint "
        f"`{ad.get('entrypoint', '<entrypoint>')}` behind interpretune's trust gate "
        "(`IT_TRUST_REMOTE_CODE`), and an adapter — unlike an op collection or a prompt config — composes "
        "into the MRO of the module your session runs. Inspect it before opting in: "
        f'`interpretune.hub.pull("{source}")` caches the repo without executing anything.'
    )
    lines += [
        "",
        "**What this card cannot tell you:** the capabilities this adapter declares at runtime and the "
        "hook patterns it refuses. Those live in the code, and the publisher does not execute it, so they "
        "are not derivable from the manifest this card renders. Read the component's own documentation, "
        "or load it and ask the registered backend directly. An absent section here is not a claim that "
        "the adapter has no limits.",
    ]
    lines += ["", "### Declares", ""]
    lines += [f"- `{name}`" for name in declares] or ["- (none)"]

    if comps:
        lines += ["", "### Compositions", "", "| composition | component | available |", "| --- | --- | --- |"]
        for entry in comps:
            adapters = " + ".join(f"`{a}`" for a in entry.get("adapters") or [])
            req = entry.get("requires") or {}
            avail = (
                "always"
                if not req
                else "requires "
                + ", ".join(f"`{v}`" for vals in req.values() for v in (vals if isinstance(vals, list) else [vals]))
            )
            lines.append(f"| {adapters} | `{entry.get('component')}` | {avail} |")
        lines += [
            "",
            "A composition whose requirements this environment cannot satisfy is **skipped and reported**, "
            "not silently absent — the others still register, so one published component serves whatever "
            "compositions the installed environment supports.",
        ]

    lines += [
        "",
        "### Provenance of this card",
        "",
        "Generated at publish from the validated `it_component.yaml`. The publisher does not execute the "
        "entrypoint, so these are the component's DECLARATIONS. What it actually registers is reconciled "
        "against the satisfiable set when `interpretune.hub.load_hub_adapter` loads it, which is where an "
        "overstating manifest fails.",
    ]
    return lines


def _check_adapter_manifest_coherence(manifest: dict, source: str) -> None:
    """Cheap structural check the publisher CAN do without executing anything.

    Not the load-time reconciliation — that needs the code to run. This catches the subset visible in the
    manifest alone: a composition naming an adapter the component neither declares nor can expect to find
    bundled. That is a card promising a composition nothing could ever register.
    """
    from interpretune.protocol import Adapter

    ad = manifest.get("adapters") or {}
    known = set(ad.get("declares") or []) | set(Adapter.__members__)
    for entry in ad.get("compositions") or []:
        unknown = [a for a in (entry.get("adapters") or []) if a not in known]
        if unknown:
            raise ComponentCardError(
                f"{source}: composition {entry.get('adapters')!r} names adapter(s) {unknown!r} that this "
                f"component does not declare and interpretune does not provide. Nothing could register it, "
                f"so the card would advertise a composition no environment can supply."
            )


def generate_component_card(manifest: dict, repo_id: str, summary: str | None = None) -> ModelCard:
    """Generate the full card for a component repo from its manifest."""
    meta = component_card_metadata(manifest)
    task = (manifest.get("module") or {}).get("task") or {}
    title = repo_id.split("/", 1)[-1]
    lines = [f"# {title}", ""]
    lines.append(summary or task.get("description") or f"An interpretune component collection ({title}).")
    lines.append("")
    lines.append(
        f"An [interpretune](https://github.com/speediedan/interpretune) component repo (kinds: "
        f"{', '.join(manifest['kinds'])}; manifest schema v{manifest['it_schema_version']})."
    )
    configs = (manifest.get("module") or {}).get("configs") or {}
    if configs:
        lines += ["", "## Configurations", ""]
        lines += [f"- `{key}`" for key in sorted(configs)]
        lines += [
            "",
            "Resolution is manifest-first: fetch `it_component.yaml`, then only the configuration you need.",
        ]
    op_files = (manifest.get("ops") or {}).get("files") or []
    if op_files:
        lines += ["", "## Operations", ""]
        lines += [f"- declared in `{rel}`" for rel in op_files]
        lines += [
            "",
            'Fetch with `interpretune.hub.pull_ops("<org>/<repo>")` (manifest-first, revision-pinned). The',
            "collection's ops are then addressable by their namespaced names; `interpretune.hub.prefer_ops`",
            "opts into resolving their BARE names here instead of interpretune's bundled ops.",
        ]
    definitions = (manifest.get("promptconfigs") or {}).get("definitions") or {}
    if definitions:
        lines += ["", "## Prompt-config definitions", ""]
        for name in sorted(definitions):
            desc = (definitions[name] or {}).get("description")
            lines.append(f"- `{name}`" + (f" — {desc}" if desc else ""))
        lines += [
            "",
            "Reference a definition from a task configuration via `compose_ref: <org>/<repo>#<name>`.",
        ]
    if "adapters" in (manifest.get("kinds") or []):
        _check_adapter_manifest_coherence(manifest, repo_id)
        lines += _adapter_card_sections(manifest, repo_id)
    return ModelCard(f"---\n{meta.to_yaml()}\n---\n\n" + "\n".join(lines) + "\n")
