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
    return ModelCard(f"---\n{meta.to_yaml()}\n---\n\n" + "\n".join(lines) + "\n")
