"""Durable per-repo revision pins for op collections (interpretune#334).

The trust posture tells users to pin a revision "so trusted code cannot change under you", and
``pull_ops(revision=...)`` honors the pin for the DOWNLOAD — but until #334, op discovery ignored
pins entirely and loaded whatever revision the cache surfaced (``refs/main`` preferred). Pin ``A``,
let anyone republish ``B``, and the next session that resolved ``main`` would scan, trust-gate,
compile and execute ``B``: the precise thing the pinning advice promises cannot happen.

This module is the durable record that closes that gap: a revision-pinned pull writes a per-repo
pin file, and discovery consults it BEFORE any ref. The record deliberately lives in an
interpretune-owned sidecar directory beside the ops hub cache rather than inside it:

- A ``refs/it-pinned`` marker inside the HF cache layout parses for free via ``scan_cache_dir``,
  but the moment its snapshot is evicted the orphaned ref makes the scanner drop the ENTIRE repo
  as corrupted ("Reference(s) refer to missing commit hashes", measured against huggingface_hub
  1.28.0) — a failure mode worse than the one being fixed — and huggingface_hub's own GC deletes
  ref files, so the record's durability would rest on an upstream layout contract that upstream
  actively mutates. That is the same trust-what-you-don't-control failure class #334 exists to
  close.
- A stray file or directory at the scanned cache ROOT survives scanning but pollutes every scan
  with a per-entry warning.

The sidecar (``<cache>_pins/``) follows any ``IT_ANALYSIS_HUB_CACHE`` override automatically and
is invisible to the HF scanner. One JSON file per repo, written atomically, so concurrent pins of
different repos cannot clobber each other.

Pin SEMANTICS (enforced by the consumers, recorded here):

- A pin binds discovery strictly: a pinned-but-evicted revision is refused, never silently
  substituted with ``main`` (``cache_manager.discover_hub_yaml_files``).
- ``revision=None`` and ``revision="main"`` are not pins: both mean "the moving default", and
  freezing a user who explicitly asked for ``main`` would be the opposite surprise.
- A pin names the RESOLVED commit, not the requested spec: pinning a tag or branch freezes what
  it pointed at when trust was granted, which is what "audited once" means. The requested spec is
  kept alongside for reporting.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_PINS_DIR_SUFFIX = "_pins"


class OpPinError(ValueError):
    """A pin record is malformed or a pin operation received an invalid argument."""


def _default_cache_root() -> Path:
    # late import so tests monkeypatching `interpretune.analysis.IT_ANALYSIS_HUB_CACHE` (the value
    # discovery itself resolves) steer the pin default identically
    from interpretune.analysis import IT_ANALYSIS_HUB_CACHE

    return Path(IT_ANALYSIS_HUB_CACHE)


def op_pins_dir(cache_root: Path | str | None = None) -> Path:
    """The sidecar directory holding pin records for one ops hub cache."""
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    return root.parent / f"{root.name}{_PINS_DIR_SUFFIX}"


def _pin_path(repo_id: str, cache_root: Path | str | None = None) -> Path:
    user, _, repo = repo_id.partition("/")
    if not user or not repo or "/" in repo:
        raise OpPinError(f"expected a `user/repo` repo id, got {repo_id!r}")
    # mirror the HF cache dir naming so a pin record is greppable next to the repo it governs
    return op_pins_dir(cache_root) / f"models--{user}--{repo}.json"


def record_op_pin(repo_id: str, commit: str, requested_revision: str, cache_root: Path | str | None = None) -> Path:
    """Write (or overwrite) the pin record for one repo; returns the record path.

    Called by revision-pinned pulls with the commit the manifest resolved to. Overwriting IS the
    update verb: re-pulling at a new revision moves the pin to it.
    """
    if not commit:
        raise OpPinError(f"refusing to record an empty commit for {repo_id!r}")
    path = _pin_path(repo_id, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "repo_id": repo_id,
        "commit": commit,
        "requested_revision": requested_revision,
        "pinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_op_pin(repo_id: str, cache_root: Path | str | None = None) -> dict | None:
    """The pin record for one repo, or ``None`` when the repo is unpinned.

    A malformed record returns ``None`` with a warning rather than raising: discovery consults this
    on every load, and a corrupted sidecar file must degrade to today's unpinned behavior, not deny
    a session its ops.
    """
    path = _pin_path(repo_id, cache_root)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        record = json.loads(raw)
        if not isinstance(record, dict) or not record.get("commit"):
            raise ValueError("pin record is not a dict with a `commit`")
    except ValueError as err:
        from interpretune.utils.logging import rank_zero_warn

        rank_zero_warn(
            f"Ignoring malformed op-collection pin record {path}: {err}. The repo behaves as "
            f"unpinned until re-pinned (pull with an explicit revision) or the file is removed."
        )
        return None
    return record


def clear_op_pin(repo_id: str, cache_root: Path | str | None = None) -> bool:
    """Remove one repo's pin record; ``True`` when a record existed."""
    path = _pin_path(repo_id, cache_root)
    try:
        path.unlink()
        return True
    except OSError:
        return False


def list_op_pins(cache_root: Path | str | None = None) -> dict[str, dict]:
    """All pin records for one ops hub cache, keyed by repo id.

    Includes pins whose repo is no longer cached at all — an inert record, flagged by the caller
    (``it.hub.op_pins`` annotates cached-ness) rather than silently dropped here.
    """
    pins: dict[str, dict] = {}
    directory = op_pins_dir(cache_root)
    if not directory.is_dir():
        return pins
    for path in sorted(directory.glob("models--*.json")):
        record = _read_record_path(path)
        if record is not None:
            pins[record["repo_id"]] = record
    return pins


def _read_record_path(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and record.get("commit") and record.get("repo_id"):
            return record
    except (OSError, ValueError):
        pass
    from interpretune.utils.logging import rank_zero_warn

    rank_zero_warn(f"Skipping malformed op-collection pin record {path} while listing pins.")
    return None


__all__ = ["OpPinError", "clear_op_pin", "list_op_pins", "op_pins_dir", "read_op_pin", "record_op_pin"]
