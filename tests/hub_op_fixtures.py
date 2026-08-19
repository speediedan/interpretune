"""Helpers for building cached hub op-collection fixtures in HF cache layout.

Op discovery is manifest-routed (#266 Phase 3): the op YAMLs of a cached snapshot are exactly the ones its
``it_component.yaml`` declares in ``ops.files``. A fixture that only drops a YAML into a snapshot therefore
describes a repo the dispatcher correctly refuses to read ops from, which is the contract, not a bug -- so
every fixture that means "a well-formed cached collection" declares its op files through here rather than
hand-rolling the manifest at a dozen call sites.

Deliberately-malformed layouts (no manifest, unsatisfiable ``requires:``) belong in
``tests/core/test_op_collection_manifest_routing.py``, which asserts how discovery reports them.
"""

from __future__ import annotations

from pathlib import Path

OP_YAML_SUFFIXES = (".yaml", ".yml")


def declare_cached_op_files(snapshot_dir: Path, *op_files: str, requires: str | None = None) -> Path:
    """Write the ``it_component.yaml`` that makes ``snapshot_dir`` a well-formed op collection.

    Args:
        snapshot_dir: an HF-layout snapshot directory that already contains the op YAMLs.
        op_files: repo-relative op YAML names to declare. Defaults to every YAML present in the snapshot
            (excluding the manifest itself), which keeps call sites a single line and self-maintaining.
        requires: optional ``requires:`` block body (e.g. ``'interpretune: ">=0.1.0.dev0"'``).

    Returns:
        The path to the written manifest.
    """
    snapshot_dir = Path(snapshot_dir)
    declared = list(op_files) or sorted(
        p.name for p in snapshot_dir.iterdir() if p.suffix in OP_YAML_SUFFIXES and p.name != "it_component.yaml"
    )
    lines = ["it_schema_version: 1", "kinds: [ops]", "ops:", "  files:"]
    lines += [f"    - {name}" for name in declared]
    if requires:
        lines += ["requires:", f"  {requires}"]
    manifest = snapshot_dir / "it_component.yaml"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_cached_op_collection(
    hub_cache: Path,
    repo_id: str = "testuser/test",
    revision: str = "abc123",
    op_files: dict[str, str] | None = None,
    requires: str | None = None,
) -> Path:
    """Materialize a complete well-formed cached op collection; returns the snapshot directory."""
    user, repo = repo_id.split("/", 1)
    snapshot = Path(hub_cache) / f"models--{user}--{repo}" / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, body in (op_files or {}).items():
        (snapshot / name).write_text(body, encoding="utf-8")
    declare_cached_op_files(snapshot, requires=requires)
    return snapshot
