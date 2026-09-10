"""Generated repo cards: the discovery sentinel, written at publish time, always.

Every repo interpretune publishes carries ``library_name: interpretune`` plus ``interpretune`` /
``interpretune-<kind>`` tags (and, for task components, a ``task:<name>`` tag and mirrored ``datasets:``
metadata). The card is generated — no publish path can produce a card-less repo, because
``library_name: interpretune`` is also the HF library-registration precondition and the tag is what
``list_models(filter="interpretune")`` discovery queries. (The pre-2026-08 op path shipped repos with no tags
at all, which made that discovery filter dead code; generation-at-publish fixes it by construction.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


#: The conformance suite's report, published beside a component as a declared supplementary file.
CONFORMANCE_REPORT_FILE = "conformance_report.json"
CONFORMANCE_REPORT_FORMAT = "interpretune.conformance.report/1"

_CANNOT_TELL = (
    "**What this card cannot tell you:** the capabilities this adapter declares at runtime and the hook patterns "
    "it refuses. Those live in the code, and the publisher does not execute it, so they are not derivable from the "
    "manifest this card renders. Read the component's own documentation, or load it and ask the registered backend "
    "directly. An absent section here is not a claim that the adapter has no limits."
)


def _load_conformance_report(tree: Path) -> dict[str, Any] | None:
    """The staged report as a dict, or ``None`` when the tree carries none or it does not parse."""
    path = tree / CONFORMANCE_REPORT_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def measured_capabilities_lines(report: dict[str, Any] | None, source_revision: str | None) -> list[str]:
    """The measured-capabilities block for an adapter card, or the absence sentence with the reason, as lines.

    **A report is never trusted by its presence.** An artifact keyed to a revision the component has moved past is worse
    than none, because it reads as measured; a red run's artifact describes a declaration the suite did not accept. So
    the block renders only when the report's format is known, its git head equals the revision being published, and its
    run exited zero. Every other case renders the standing "cannot tell" sentence plus one line saying which report
    exists and why it is not shown, so a reader is never left inferring that nothing was ever measured.
    """
    if report is None:
        return ["", _CANNOT_TELL]
    prov = report.get("provenance") or {}
    head = prov.get("git_head")
    if report.get("format") != CONFORMANCE_REPORT_FORMAT:
        return [
            "",
            _CANNOT_TELL,
            "",
            f"A `{CONFORMANCE_REPORT_FILE}` is published but its format is not one this card reads; it is not shown.",
        ]
    if source_revision is None or not head:
        return [
            "",
            _CANNOT_TELL,
            "",
            f"A `{CONFORMANCE_REPORT_FILE}` is published, but the revision it measured or the revision being published "
            "could not be determined, so it is not shown.",
        ]
    if head != source_revision:
        return [
            "",
            _CANNOT_TELL,
            "",
            f"A `{CONFORMANCE_REPORT_FILE}` exists for component revision `{head[:12]}`, not the one published "
            f"(`{source_revision[:12]}`); it is not shown, because a measurement of an earlier revision would read as "
            "a measurement of this one.",
        ]
    if int(prov.get("exit_status", 1)) != 0:
        return [
            "",
            _CANNOT_TELL,
            "",
            f"A `{CONFORMANCE_REPORT_FILE}` exists for this revision, but the run that produced it did not pass, so it "
            "is not shown.",
        ]
    lines = ["", "### Measured capabilities", ""]
    lines.append(
        "Measured by interpretune's conformance suite on a composed session, not declared by the manifest: "
        f"interpretune {prov.get('interpretune_version', '?')}, component revision `{head[:12]}`, "
        f"run at {prov.get('measured_at', '?')}, exit status {prov.get('exit_status')}. "
        f"The report is `{CONFORMANCE_REPORT_FILE}` in this repo."
    )
    lines.append("")
    lines.append(
        "A point a backend declares it cannot capture, with a reason naming a spelling the vocabulary lacks, is a "
        "**vocabulary gap**: the model can produce the tensor and interpretune has no name to ask for it by. A "
        "surface that is not declared at all is a **capability gap**. The two read differently to someone deciding "
        "whether to build on this adapter, and only the record's own reason says which applies."
    )
    for target, decl in sorted((report.get("targets") or {}).items()):
        comp = " + ".join(f"`{a}`" for a in decl.get("composition") or []) or "(composition not recorded)"
        lines += ["", f"#### {target}", "", f"Composition {comp} on `{decl.get('model_id') or '?'}`.", ""]
        model_caps = decl.get("model_capabilities") or []
        analysis_caps = decl.get("analysis_capabilities") or []
        lines.append(f"- declares: {', '.join(f'`{c}`' for c in model_caps + analysis_caps) or '(nothing)'}")
        iv = decl.get("intervention")
        if iv:
            lines.append(
                f"- activation intervention: modes {', '.join(f'`{m}`' for m in iv.get('modes') or [])}; "
                f"position scopes {', '.join(f'`{m}`' for m in iv.get('position_scopes') or [])}"
            )
        lm = decl.get("latent_models")
        if lm:
            lines.append(f"- latent models: batched hooks {lm.get('batched_hooks')}")
        cap = decl.get("capture")
        if cap:
            capturable = cap.get("capturable") or []
            gaps = cap.get("uncapturable") or {}
            total = len(capturable) + len(gaps)
            lines.append(
                f"- capture: {len(capturable)} of {total} base points on `{cap.get('architecture')}`"
                f" ({cap.get('n_layers')} blocks)"
            )
            for point, why in sorted(gaps.items()):
                lines.append(f"  - cannot capture `{point}`: {why}")
        ag = decl.get("attribution_graph")
        if ag:
            lines.append(
                "- attribution graphs: "
                + (
                    "require the modeling module's own eager attention"
                    if ag.get("requires_own_eager_attention")
                    else "no model requirement"
                )
            )
        fi = decl.get("feature_intervention")
        if fi:
            lines.append(
                f"- feature intervention: value sources {', '.join(f'`{v}`' for v in fi.get('value_sources') or [])}; "
                f"constrainable layers {fi.get('constrainable_layers')};"
                f" returns activations {fi.get('returns_activations')}"
            )
    lines += [
        "",
        f"Cases: {len(report.get('ran') or [])} ran, {len(report.get('skipped_undeclared') or [])} skipped because the "
        f"surface is undeclared, {len(report.get('skipped_other') or [])} skipped for another reason, "
        f"{len(report.get('failed') or [])} failed.",
    ]
    return lines


def _adapter_card_sections(manifest: dict, source: str, measured: list[str] | None = None) -> list[str]:
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
    if measured:
        lines += measured
    else:
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


def generate_component_card(
    manifest: dict,
    repo_id: str,
    summary: str | None = None,
    *,
    tree: Path | None = None,
    source_revision: str | None = None,
) -> ModelCard:
    """Generate the full card for a component repo from its manifest.

    ``tree`` is the staged publish tree; when it carries :data:`CONFORMANCE_REPORT_FILE` (declared in the
    manifest's ``extra_files`` and written by the conformance suite in the component's own CI), the adapter
    sections gain a measured-capabilities block, guarded by :func:`measured_capabilities_lines` against a report
    that is stale or red. ``source_revision`` is the revision of the component source being published, the key
    that guard compares against; ``None`` means the publisher could not determine it, and the report is then
    treated as absent rather than trusted.
    """
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
    extra_files = manifest.get("extra_files") or []
    if extra_files:
        lines += ["", "## Supplementary files", ""]
        lines += [f"- `{rel}`" for rel in extra_files]
        lines += [
            "",
            "Declared in the manifest's `extra_files` and published by the same builder as the payloads. The",
            "published tree is the manifest's allowlist plus these; anything else present arrived out of band.",
        ]
    hookmap_files = (manifest.get("hookmaps") or {}).get("files") or []
    if hookmap_files:
        lines += ["", "## Component maps (hookmaps)", ""]
        lines += [f"- `{rel}`" for rel in hookmap_files]
        lines += [
            "",
            "One document per architecture, in the activation-point vocabulary's component-map schema. These are",
            'DATA (no code executes): fetch and register with `interpretune.hub.pull_hookmaps("<org>/<repo>")`, or',
            "`interpretune.hub.load_hookmaps` from the cache. A map for an architecture interpretune already",
            "bundles must agree with the bundled one, or be loaded with `replace=True` deliberately.",
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
        report = _load_conformance_report(tree) if tree is not None else None
        lines += _adapter_card_sections(
            manifest, repo_id, measured=measured_capabilities_lines(report, source_revision)
        )
    return ModelCard(f"---\n{meta.to_yaml()}\n---\n\n" + "\n".join(lines) + "\n")
