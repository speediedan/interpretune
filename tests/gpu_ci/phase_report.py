"""Phase manifests and phase reports for the GPU CI phases (#703).

Ported from finetuning-scheduler's ``tests/gpu_ci/phase_manifest.py`` so both repositories report a GPU run the same
way, phase by phase. ``phase_manifest.py`` beside this keeps #794's build-day collection manifests and the ROCm triage
classifier; this module is what the pipeline runs. For each phase it writes two documents:

- a **manifest** (``collect``): the node ids the phase's paths collect (``raw_node_ids``) and the ones its selection
  keeps (``selected_node_ids``), with the repository, revision, phase and selector string. Collection only: it runs on
  the CPU with ``CUDA_VISIBLE_DEVICES`` emptied and executes nothing, so it is evidence about selection, never a pass.
  A selected run's ``IT_GPU_SELECTION_FILE`` narrows it as it narrows the phase, and no memory class is set, so a
  two-stage run's manifest is the union of both stages.
- a **report** (``report``): per-node outcomes read from the records ``phase_plugin.py`` wrote while the phase ran,
  counted against the manifest, plus the run's identity (vendor, backend, evidence class, image, lease mode, installed
  versions, devices) and the manifest's SHA-256.

The phases and their selectors are the GPU job's steps (``.azure-pipelines/scripts/gpu_phase.sh`` through
``.azure-pipelines/templates/gpu-job.yml``). A phase's ``selector`` is the selection-relevant part of its step (the
pytest paths and the selection variables), not what only reruns or reports (``--reruns``, ``--durations``): a phase is
reported node by node, and a node left unexecuted fails it anyway.

The manifest digest is SHA-256 over the manifest serialized as UTF-8 JSON with sorted keys and compact separators and
no trailing newline, the rule the release eligibility checker applies. A report never claims more than its evidence
class: ``parity-pre-agent`` marks a run made outside the pipeline (no image digest), which the release checker refuses.
Coverage is not yet part of interpretune's phase evidence: every report records ``coverage: {"measured": false}``.

Usage, from the repository root (``python -m tests.gpu_ci.phase_report --help`` for the details)::

    python -m tests.gpu_ci.phase_report list
    python -m tests.gpu_ci.phase_report collect --phase cuda --out-dir /tmp/it-phases
    python -m tests.gpu_ci.phase_report run-env --phase cuda --out-dir /tmp/it-phases
    python -m tests.gpu_ci.phase_report report --phase cuda --out-dir /tmp/it-phases \\
        --vendor amd --evidence-class parity-pre-agent --lease-mode tool

Keep ``--out-dir`` outside the repository: the manifest records the tree's untracked files, so an out-dir inside the
checkout marks every later collection dirty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

if __package__:
    from tests.gpu_ci.phase_plugin import RECORD_DIR_ENV
else:  # run as a script beside its siblings (release_eligibility.py on a host Python without the test dependencies)
    from phase_plugin import RECORD_DIR_ENV

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "speediedan/interpretune"
PLUGIN = "tests.gpu_ci.phase_plugin"
COLLECT_DIRS = ("tests", "src/it_examples/tests")
# Visible-device variables a report records: a run's device set is part of what makes two hosts comparable.
DEVICE_VARS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "IT_GPU_DEVICE")
# Variables tests/conftest.py selects on; stripped from the environment before every collection. The memory class goes
# too, so a two-stage phase's manifest covers both stages. IT_GPU_SELECTION_FILE stays: it narrows a selected run's
# phase exactly as it narrows the collection.
SELECTION_VARS = (
    "IT_RUN_CUDA_TESTS",
    "IT_RUN_STANDALONE_TESTS",
    "IT_RUN_PROFILING_TESTS",
    "IT_RUN_OPTIONAL_TESTS",
    "IT_RUN_BENCHMARK_TESTS",
    "IT_GPU_MEM_CLASS",
    "IT_GPU_SMALL_MAX_GB",
)
VENDORS = {"nvidia": "cuda", "amd": "hip"}
EVIDENCE_CLASSES = ("pipeline", "parity-pre-agent")
OUTCOMES = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
CLASSIFICATIONS = ("clean", "unreviewed-exclusions", "empty", "failing")


@dataclass(frozen=True)
class Phase:
    """One GPU CI phase: what it collects, how it selects, and how the pipeline runs it."""

    name: str
    selection_env: dict = field(default_factory=dict)
    paths: tuple[str, ...] = COLLECT_DIRS
    command: str = ""
    release: bool = True


PHASES = {
    p.name: p
    for p in (
        Phase(
            "cuda",
            {"IT_RUN_CUDA_TESTS": "1"},
            command="python -m pytest --cov=src/interpretune tests src/it_examples/tests -v --reruns 2",
        ),
        Phase(
            "standalone",
            {"IT_RUN_STANDALONE_TESTS": "1"},
            command="bash ./tests/special_tests.sh --mark_type=standalone --allow-failures --allow-empty",
        ),
        Phase(
            "profile_ci",
            {"IT_RUN_PROFILING_TESTS": "1"},
            command="bash ./tests/special_tests.sh --mark_type=profile_ci --allow-failures --allow-empty",
        ),
    )
}


def selector_string(phase: Phase) -> str:
    """The phase's selection, as one comparable string: selection variables, then the pytest paths."""
    env = " ".join(f"{k}={v}" for k, v in sorted(phase.selection_env.items()))
    selection = os.environ.get("IT_GPU_SELECTION_FILE")
    parts = [env, "; pytest", *phase.paths]
    if selection:
        parts.append(f"; select_file={selection}")
    return " ".join(parts).replace(" ; ", "; ")


def manifest_digest(manifest: dict) -> str:
    """SHA-256 of the manifest as sorted, compact UTF-8 JSON without a trailing newline (the release checker's
    rule)."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_records(record_dir: Path) -> list[list[dict]]:
    """Each session's records (one list per JSON-lines file ``phase_plugin.py`` wrote), in file-name order."""
    sessions = []
    for path in sorted(Path(record_dir).glob("records-*.jsonl")):
        sessions.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return sessions


def _collect(paths: tuple[str, ...], env_update: dict, root: Path) -> tuple[list[str], list, int]:
    """Collect once through the plugin: the selected node ids, the failed collectors and pytest's exit code."""
    env = {k: v for k, v in os.environ.items() if k not in SELECTION_VARS}
    env.update(env_update)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *filter(None, [env.get("PYTHONPATH")])])
    env.pop("PYTEST_ADDOPTS", None)
    record_dir = Path(tempfile.mkdtemp(prefix="it-phase-collect-"))
    env[RECORD_DIR_ENV] = str(record_dir)
    cmd = [sys.executable, "-m", "pytest", *paths, "--collect-only", "-q", "-p", "no:cacheprovider", "-p", PLUGIN]
    try:
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        records = [r for session in read_records(record_dir) for r in session]
    finally:
        shutil.rmtree(record_dir, ignore_errors=True)
    collected = [r["nodeids"] for r in records if r["event"] == "collected"]
    errors = sorted(r["nodeid"] for r in records if r["event"] == "collect_error")
    if len(collected) != 1:
        errors.append(f"collection recorded {len(collected)} times (pytest exit {proc.returncode})")
    return (collected[0] if collected else []), errors, proc.returncode


def git_state(root: Path = ROOT) -> tuple[str, list[str] | None]:
    """``HEAD`` and the paths that differ from it (untracked included); ``None`` when git cannot say."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True
    )
    dirty = None if status.returncode else sorted(line[3:] for line in status.stdout.splitlines() if line.strip())
    return (head.stdout.strip() if head.returncode == 0 else "unknown"), dirty


def build_manifest(phase_name: str, root: Path = ROOT) -> dict:
    """Collect ``phase_name`` twice on the CPU (all of its paths, then its selection) and describe both."""
    phase = PHASES[phase_name]
    raw, raw_errors, raw_exit = _collect(phase.paths, {}, root)
    selected, sel_errors, sel_exit = _collect(phase.paths, phase.selection_env, root)
    revision, dirty = git_state(root)
    return {
        "repository": REPOSITORY,
        "revision": revision,
        "phase": phase_name,
        "scope": "selected" if os.environ.get("IT_GPU_SELECTION_FILE") else "full",
        "selector": selector_string(phase),
        # A tree that differs from the revision makes the revision a wrong name for what was collected.
        "tree_dirty_paths": dirty,
        "raw_node_ids": raw,
        "selected_node_ids": selected,
        "collection_errors": sorted(set(raw_errors + sel_errors)),
        # 5 is pytest's "no tests collected": a selected run may legitimately select nothing in a phase
        "collect_exit_codes": {"raw": raw_exit, "selected": sel_exit},
    }


def _session_node_outcomes(records: list[dict]) -> dict[str, str]:
    """A session's verdict per node: a failure or error in any stage wins, else the deciding stage's outcome."""
    per_node: dict[str, list[str]] = {}
    for r in records:
        if r["event"] == "outcome":
            per_node.setdefault(r["nodeid"], []).append(r["outcome"])
    return {
        nodeid: next((o for o in ("failed", "errors") if o in outcomes), outcomes[0])
        for nodeid, outcomes in per_node.items()
    }


def build_report(
    manifest: dict,
    sessions: list[list[dict]],
    *,
    vendor: str,
    evidence_class: str,
    lease_mode: str | None = None,
    image: str | None = None,
    versions: dict | None = None,
    reviewed_exclusions: dict | None = None,
) -> dict:
    """Count one phase's per-node outcomes against its manifest and describe the run that produced them.

    ``tests/special_tests.sh`` runs each node in its own pytest process, and a two-stage run runs a phase once per
    memory class, so one phase has many sessions; a node's verdict is the one its sessions agree on, a failure or error
    in any of them winning. A selected node with no verdict is ``unexecuted``. A session that started but never
    finished is a missing completion, and fails every node it collected that it gave no verdict for. A node that ran but
    is not in the manifest's selection is a collection error: the run and the manifest disagree about what the phase is.
    """
    if vendor not in VENDORS:
        raise ValueError(f"vendor must be one of {sorted(VENDORS)}")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"evidence class must be one of {EVIDENCE_CLASSES}")
    reviewed = reviewed_exclusions or {}
    selected = manifest["selected_node_ids"]
    verdicts: dict[str, list[str]] = {}
    missing_completions = 0
    for records in sessions:
        if not records:
            continue
        session_verdicts = _session_node_outcomes(records)
        for nodeid, verdict in session_verdicts.items():
            verdicts.setdefault(nodeid, []).append(verdict)
        finished = any(r["event"] == "finish" for r in records)
        if not finished and any(r["event"] == "start" for r in records):
            missing_completions += 1
            collected = next((r["nodeids"] for r in records if r["event"] == "collected"), [])
            for nodeid in collected:
                if nodeid not in session_verdicts:
                    verdicts.setdefault(nodeid, []).append("incomplete")
    nodes, counts = {}, dict.fromkeys(OUTCOMES, 0)
    unexecuted = []
    for nodeid in selected:
        seen = verdicts.get(nodeid)
        if not seen:
            unexecuted.append(nodeid)
            nodes[nodeid] = {"outcome": "unexecuted", "sessions": 0}
            continue
        if "incomplete" in seen:
            outcome = "failed"
        else:
            outcome = next((o for o in ("failed", "errors") if o in seen), seen[-1])
        counts[outcome] += 1
        nodes[nodeid] = {"outcome": outcome, "sessions": len(seen)}
    unmanifested = sorted(set(verdicts) - set(selected))
    exclusions = [
        {"nodeid": n, "outcome": v["outcome"], "reason": reviewed.get(n)}
        for n, v in nodes.items()
        if v["outcome"] in ("skipped", "xfailed")
    ]
    report = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "phase": manifest["phase"],
        "scope": manifest["scope"],
        "selector": manifest["selector"],
        "evidence_class": evidence_class,
        "vendor": vendor,
        "backend": VENDORS[vendor],
        "lease_mode": lease_mode,
        "image": image,
        "versions": versions or {},
        "count_unit": "unique-node-id",
        "collection": {
            "raw": len(manifest["raw_node_ids"]),
            "deselected": len(manifest["raw_node_ids"]) - len(selected),
            "selected": len(selected),
            "errors": len(manifest.get("collection_errors", [])) + len(unmanifested),
        },
        "outcomes": {"attempted": len(selected) - len(unexecuted), **counts, "unexecuted": len(unexecuted)},
        "missing_completions": missing_completions,
        "exclusions_reviewed": all(e["reason"] for e in exclusions),
        "exclusions": exclusions,
        "unmanifested_nodes": unmanifested,
        "nodes": nodes,
        "manifest_sha256": manifest_digest(manifest),
        "coverage": {"measured": False},
    }
    report["classification"] = classify(report)
    return report


def classify(report: dict) -> str:
    """One word for the phase: ``clean``, ``unreviewed-exclusions``, ``empty`` or ``failing``."""
    o, c = report["outcomes"], report["collection"]
    if c["errors"] or o["failed"] or o["errors"] or o["xpassed"] or o["unexecuted"] or report["missing_completions"]:
        return "failing"
    if not c["selected"] or not o["passed"]:
        return "empty"
    return "clean" if report["exclusions_reviewed"] else "unreviewed-exclusions"


def gate(classification: str, scope: str, job_class: str) -> bool:
    """Whether a phase with ``classification`` passes the job.

    ``failing`` fails every job. ``empty`` fails a full run (a full phase that ran nothing means the collector broke)
    but not a selected one, which may legitimately select nothing in a phase. ``unreviewed-exclusions`` passes a gate
    with a warning and fails a release leg, whose evidence must name a reason for every skip.
    """
    if classification == "failing":
        return False
    if classification == "empty":
        return scope != "full"
    if classification == "unreviewed-exclusions":
        return job_class != "ci-release"
    return True


def installed_versions() -> dict:
    """The versions a phase report records: interpreter, torch (with its commit and backend), interpretune and the
    wrapped packages."""
    import platform

    versions: dict = {"python": platform.python_version()}
    try:
        import torch

        versions.update(
            torch=torch.__version__,
            torch_git_version=torch.version.git_version,
            torch_cuda=torch.version.cuda,
            torch_hip=getattr(torch.version, "hip", None),
        )
    except ImportError as exc:  # a report from a broken environment should say so, not crash
        versions["torch"] = f"unavailable: {exc}"
    from importlib import metadata

    for name in ("interpretune", "lightning", "finetuning-scheduler", "transformer-lens", "nnsight", "sae-lens"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_identity() -> dict:
    """The devices the run saw: the visible-device variables, the device count and each device's name."""
    ident: dict = {var: os.environ.get(var) for var in DEVICE_VARS}
    try:
        import torch

        count = torch.cuda.device_count()
        ident.update(device_count=count, devices=[torch.cuda.get_device_name(i) for i in range(count)])
    except (ImportError, RuntimeError, AssertionError) as exc:
        ident.update(device_count=None, devices=f"unavailable: {exc}")
    return ident


def run_env(phase_name: str, out_dir: Path) -> dict:
    """The environment a phase's pipeline command needs to be recorded (every pytest process it starts inherits
    it)."""
    PHASES[phase_name]  # an unknown phase is a KeyError, not an empty record directory
    addopts = " ".join(filter(None, [os.environ.get("PYTEST_ADDOPTS", ""), "-p", PLUGIN]))
    record_dir = Path(out_dir).resolve() / "records" / phase_name
    return {RECORD_DIR_ENV: str(record_dir), "PYTEST_ADDOPTS": addopts.strip()}


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _summary(doc: dict) -> str:
    o, c = doc["outcomes"], doc["collection"]
    return (
        f"phase={doc['phase']} {doc['evidence_class']} vendor={doc['vendor']} scope={doc['scope']}"
        f" collected={c['raw']} selected={c['selected']} passed={o['passed']} failed={o['failed']}"
        f" errors={o['errors']} skipped={o['skipped']} xfailed={o['xfailed']} xpassed={o['xpassed']}"
        f" unexecuted={o['unexecuted']} missing_completions={doc['missing_completions']} -> {doc['classification']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the phases, their selectors and the pipeline commands that run them")
    collect = sub.add_parser("collect", help="write <out-dir>/manifests/<phase>.json (CPU only, executes nothing)")
    collect.add_argument("--phase", action="append", choices=sorted(PHASES), required=True)
    collect.add_argument("--out-dir", type=Path, required=True)
    env = sub.add_parser("run-env", help="print the exports that record a phase")
    env.add_argument("--phase", choices=sorted(PHASES), required=True)
    env.add_argument("--out-dir", type=Path, required=True)
    report = sub.add_parser("report", help="write <out-dir>/reports/<phase>.json from the manifest and the records")
    report.add_argument("--phase", choices=sorted(PHASES), required=True)
    report.add_argument("--out-dir", type=Path, required=True)
    report.add_argument("--vendor", choices=sorted(VENDORS), required=True)
    report.add_argument("--evidence-class", choices=EVIDENCE_CLASSES, required=True)
    report.add_argument("--job-class", default="ci-any-vendor", help="the GPU job's class; ci-release is strictest")
    report.add_argument("--lease-mode", default=None)
    report.add_argument("--image", default=None)
    report.add_argument("--reviewed-exclusions", type=Path, default=None, help="JSON object: node id -> reason")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, phase in PHASES.items():
            print(f"{name}\trelease={phase.release}\tselector={selector_string(phase)}\t{phase.command}")
        return 0
    if args.cmd == "collect":
        status = 0
        for name in args.phase:
            manifest = build_manifest(name)
            _write_json(args.out_dir / "manifests" / f"{name}.json", manifest)
            # collected beside phase, always: a count without its phase reads as a suite result
            print(
                f"phase={name} scope={manifest['scope']} raw={len(manifest['raw_node_ids'])}"
                f" selected={len(manifest['selected_node_ids'])} errors={len(manifest['collection_errors'])}"
                f" (collection only) sha256={manifest_digest(manifest)}"
            )
            status |= bool(manifest["collection_errors"])
        return status
    if args.cmd == "run-env":
        for key, value in run_env(args.phase, args.out_dir).items():
            print(f"export {key}={json.dumps(value)}")
        return 0
    manifest = json.loads((args.out_dir / "manifests" / f"{args.phase}.json").read_text(encoding="utf-8"))
    reviewed = json.loads(args.reviewed_exclusions.read_text(encoding="utf-8")) if args.reviewed_exclusions else None
    doc = build_report(
        manifest,
        read_records(args.out_dir / "records" / args.phase),
        vendor=args.vendor,
        evidence_class=args.evidence_class,
        lease_mode=args.lease_mode,
        image=args.image,
        versions={**installed_versions(), **device_identity()},
        reviewed_exclusions=reviewed,
    )
    _write_json(args.out_dir / "reports" / f"{args.phase}.json", doc)
    print(_summary(doc))
    for e in doc["exclusions"]:
        if not e["reason"]:
            print(f"  unreviewed {e['outcome']}: {e['nodeid']}")
    return 0 if gate(doc["classification"], doc["scope"], args.job_class) else 1


if __name__ == "__main__":
    sys.exit(main())
