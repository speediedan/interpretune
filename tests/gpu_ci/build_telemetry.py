"""Join one GPU build's plan, phase evidence, lease events and phase intervals into a per-build telemetry record.

The GPU job leaves its evidence on the template's evidence mount, published as the build's artifact:

- ``telemetry/plan.json``: the planner's record (``plan_gpu_run.py --record``): the build's identity, the mode and why,
  the areas and the change set it selected from;
- ``manifests/<phase>.json`` and ``reports/<phase>.json``: gpu-epsilon's phase manifest and report;
- ``denominator/manifests/<phase>.json``: on a selected run, the same phase collected without the selection file, at
  the same revision, which is the full phase a selected count is a fraction of;
- ``telemetry/events.jsonl``: lease and phase events (``.azure-pipelines/scripts/telemetry.sh``).

``assemble`` writes ``telemetry/build_telemetry.json`` from them. Two rules shape it:

- **Evidence is bound before it is counted.** A report is joined only to the manifest whose digest it records, at the
  same revision and selector, and the manifests only to the revision the plan says the job ran. A mismatch is refused
  by name: a count from one revision reported under another is the error this record exists to prevent.
- **What is missing is said, never zero.** A phase interval that started and never ended, a lease never released, a
  denominator that was not collected: each is recorded as unknown with the reason, and ``unknowns`` lists them all. A
  sum over incomplete intervals is not reported as the job's time.

Two quantities are easy to misread, so the record names them for what they are. ``raw_collection`` is the whole test
collection, not the GPU phase: the full GPU phase is ``full_phase``. A lease's ``wait_s`` is how long the job queued,
not how long it held the devices: that is ``held_s``, between acquisition and release.

Usage (from the repository root): ``python -m tests.gpu_ci.build_telemetry assemble --dir /evidence/it-phases``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "interpretune.gpu-ci.build-telemetry/1"
PLAN_SCHEMA = "interpretune.gpu-ci.plan/1"


class TelemetryError(ValueError):
    """Evidence that cannot be joined as it stands: named, never guessed around."""


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_events(path: Path) -> tuple[list[dict], int]:
    """The events in ``path``, in the order written, and how many lines could not be read (a torn write)."""
    if not path.is_file():
        return [], 0
    events, unreadable = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            unreadable += 1
    return events, unreadable


def _digest(manifest: dict) -> str:
    from gpu_epsilon.evidence import manifest_digest

    return manifest_digest(manifest)


def phase_entry(name: str, manifest: dict, report: dict | None, denominator: dict | None, unknowns: list[str]) -> dict:
    """One phase's counts, with its report and denominator bound to its manifest or refused."""
    digest = _digest(manifest)
    selected = manifest["selected_node_ids"]
    entry: dict = {
        "scope": manifest["scope"],
        "selector": manifest["selector"],
        "manifest_sha256": digest,
        "selected": len(selected),
        "raw_collection": len(manifest["raw_node_ids"]),
        "collection_errors": len(manifest.get("collection_errors", [])),
    }
    if manifest["scope"] == "full":
        entry.update(full_phase=len(selected), full_phase_source="this manifest, collected in the full scope")
    elif denominator is None:
        reason = f"phase {name}: no full-scope collection beside this {manifest['scope']} manifest"
        entry.update(full_phase=None, full_phase_unknown=reason)
        unknowns.append(reason)
    else:
        if denominator["revision"] != manifest["revision"]:
            raise TelemetryError(
                f"phase {name}: the denominator was collected at {denominator['revision']}, the manifest at "
                f"{manifest['revision']}; a denominator from another revision is not this phase's"
            )
        if denominator["scope"] != "full" or not manifest["selector"].startswith(denominator["selector"]):
            raise TelemetryError(
                f"phase {name}: the denominator's selector ({denominator['scope']}: {denominator['selector']!r}) is "
                f"not the manifest's without its selection ({manifest['selector']!r})"
            )
        full = denominator["selected_node_ids"]
        entry.update(full_phase=len(full), full_phase_source="a full-scope collection at the same revision")
        outside = sorted(set(selected) - set(full))
        if outside:
            entry["selected_outside_full_phase"] = outside
            unknowns.append(f"phase {name}: {len(outside)} selected node(s) are not in the full phase")
    if report is None:
        reason = f"phase {name}: no report (the job ended before its report step, or the phase never ran)"
        entry.update(outcomes=None, classification=None, report_unknown=reason)
        unknowns.append(reason)
        return entry
    if report.get("manifest_sha256") != digest:
        raise TelemetryError(
            f"phase {name}: the report counts against manifest {report.get('manifest_sha256')}, but the manifest "
            f"here digests to {digest}; the report does not describe this manifest"
        )
    for key in ("revision", "selector", "phase"):
        if report.get(key) != manifest.get(key):
            raise TelemetryError(f"phase {name}: the report's {key} {report.get(key)!r} is not the manifest's")
    entry.update(
        outcomes=report["outcomes"],
        classification=report.get("classification"),
        vendor=report.get("vendor"),
        lease_mode=report.get("lease_mode"),
        exclusions_reviewed=report.get("exclusions_reviewed"),
    )
    return entry


def phase_intervals(events: list[dict], unknowns: list[str]) -> dict:
    """Each phase class's start and end, paired in order; a start without an end is interrupted, not zero."""
    out: dict[str, dict] = {}
    for event in (e for e in events if e.get("kind") == "phase"):
        key = f"{event.get('phase')}/{event.get('class')}"
        if event.get("action") == "start":
            out[key] = {"start": event["ts"], "_epoch": int(event["epoch"]), "device": event.get("device")}
        elif event.get("action") == "end" and key in out and "end" not in out[key]:
            entry = out[key]
            entry.update(end=event["ts"], seconds=int(event["epoch"]) - entry["_epoch"])
            entry.update(rc=int(event.get("rc", -1)), ran=event.get("ran"))
    for key, entry in out.items():
        entry.pop("_epoch", None)
        if "end" not in entry:
            entry.update(end=None, seconds=None)
            unknowns.append(f"phase {key}: started at {entry['start']} and never recorded its end (interrupted)")
    return out


def lease_stages(events: list[dict], unknowns: list[str]) -> dict:
    """Each lease stage's waits, acquisition and release.

    ``wait_s`` is queueing; ``held_s`` is holding.
    """
    stages: dict[str, dict] = {}
    for event in (e for e in events if e.get("kind") == "lease"):
        stage = stages.setdefault(
            event.get("stage", "?"), {"waits": [], "acquired": None, "released": None, "unleased": False}
        )
        action, epoch = event.get("action"), int(event["epoch"])
        if action == "wait":
            stage["waits"].append({"at": event["ts"], "_epoch": epoch, "attempt": event.get("attempt")})
        elif action in ("timeout", "error") and stage["waits"]:
            stage["waits"][-1]["outcome"] = action
        elif action == "acquired" and stage["acquired"] is None:
            devices = event.get("devices", "unknown")
            stage["acquired"] = {
                "at": event["ts"],
                "_epoch": epoch,
                "grant": event.get("grant"),
                "devices": None if devices == "unknown" else devices.split(";"),
                "mode": event.get("mode"),
            }
            if stage["waits"]:
                stage["waits"][-1]["outcome"] = "acquired"
        elif action == "released" and stage["released"] is None:
            stage["released"] = {"at": event["ts"], "_epoch": epoch, "via": event.get("via", "release step")}
        elif action == "unleased":
            stage["unleased"] = True
    for name, stage in stages.items():
        first_wait = stage["waits"][0]["_epoch"] if stage["waits"] else None
        acquired, released = stage["acquired"], stage["released"]
        stage["wait_s"] = acquired["_epoch"] - first_wait if acquired and first_wait is not None else None
        stage["held_s"] = released["_epoch"] - acquired["_epoch"] if acquired and released else None
        if acquired and acquired["devices"] is None:
            unknowns.append(f"lease {name}: granted {acquired['grant']!r} but its devices could not be enumerated")
        if acquired and not released:
            unknowns.append(f"lease {name}: acquired at {acquired['at']} and no release was recorded (interrupted)")
        if stage["waits"] and not acquired:
            unknowns.append(f"lease {name}: waited {len(stage['waits'])} time(s) and never acquired")
        for record in (*stage["waits"], acquired, released):
            if record:
                record.pop("_epoch", None)
    return stages


def assemble(evidence_dir: Path) -> dict:
    """The build's telemetry record from its evidence directory; ``TelemetryError`` names what cannot be joined."""
    unknowns: list[str] = []
    plan = _read_json(evidence_dir / "telemetry" / "plan.json")
    if plan is None:
        unknowns.append("no plan record: identity, mode and change set are unknown")
    elif plan.get("schema") != PLAN_SCHEMA:
        raise TelemetryError(f"plan record schema {plan.get('schema')!r}; this reader knows {PLAN_SCHEMA!r}")
    identity = (plan or {}).get("identity", {})
    revision = identity.get("merge_revision")
    phases: dict[str, dict] = {}
    for path in sorted((evidence_dir / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        name = manifest["phase"]
        if revision and manifest["revision"] != revision:
            raise TelemetryError(
                f"phase {name}: manifest collected at {manifest['revision']}, but the plan says the job ran {revision}"
            )
        phases[name] = phase_entry(
            name,
            manifest,
            _read_json(evidence_dir / "reports" / f"{name}.json"),
            _read_json(evidence_dir / "denominator" / "manifests" / f"{name}.json"),
            unknowns,
        )
    if not phases and (plan or {}).get("mode") != "none":
        unknowns.append("no phase manifests: nothing was collected")
    events, unreadable = read_events(evidence_dir / "telemetry" / "events.jsonl")
    if unreadable:
        unknowns.append(f"{unreadable} event line(s) could not be read")
    if not events:
        unknowns.append("no lease or phase events were recorded")
    intervals = phase_intervals(events, unknowns)
    complete = [i["seconds"] for i in intervals.values() if i["seconds"] is not None]
    return {
        "schema": SCHEMA,
        "identity": identity or None,
        "plan": {
            k: plan[k]
            for k in ("mode", "why", "extended", "lease_policy", "selection", "changed_paths", "changed_unknown")
        }
        if plan
        else None,
        "phases": phases,
        "phase_intervals": intervals,
        # the sum is reported only when every interval it would sum is complete
        "phase_seconds_total": sum(complete) if intervals and len(complete) == len(intervals) else None,
        "leases": lease_stages(events, unknowns),
        "unknowns": unknowns,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    asm = sub.add_parser("assemble", help="write telemetry/build_telemetry.json from an evidence directory")
    asm.add_argument("--dir", type=Path, required=True, help="the phase-evidence directory (/evidence/it-phases)")
    asm.add_argument("--out", type=Path, default=None, help="default: <dir>/telemetry/build_telemetry.json")
    args = ap.parse_args(argv)
    try:
        record = assemble(args.dir)
    except TelemetryError as e:
        print(f"##vso[task.logissue type=error]build telemetry refused: {e}")
        return 1
    out = args.out or args.dir / "telemetry" / "build_telemetry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    for name, phase in record["phases"].items():
        full = phase["full_phase"] if phase["full_phase"] is not None else "unknown"
        print(f"phase={name} scope={phase['scope']} selected={phase['selected']} full_phase={full}")
    print(f"phase_seconds_total={record['phase_seconds_total']} unknowns={len(record['unknowns'])} -> {out}")
    for line in record["unknowns"]:
        print(f"  unknown: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
