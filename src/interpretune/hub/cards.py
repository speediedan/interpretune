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
    return ModelCard(f"---\n{meta.to_yaml()}\n---\n\n" + "\n".join(lines) + "\n")
