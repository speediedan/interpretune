"""Run Neuronpedia dashboard benchmark waves and package regenerable reviewer artifacts.

Wraps ``scripts/profile_neuronpedia_dashboard_generation.py`` to run either the accepted 3-way, 2-scenario
benchmark (detached preserved-baseline legacy, in-tree current legacy, current ``columnar_gpu``) or limited 2-way
scaling sweeps (current legacy + ``columnar_gpu`` only), then extracts primary/substage/import/parity tables,
regenerates the unified Mermaid flow diagram, executes the parameterized dashboard profiling notebook via
papermill, and assembles a self-contained reviewer artifact package.

Use ``--from-existing <artifact_root>`` to skip execution and package artifacts from a prior benchmark root.

See ``scripts/dashboard_benchmark_suite_usage.md`` for example commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dashboard_benchmark_artifacts import (
    DETACHED_BASELINE_LINEAGE_PREFIX,
    SCENARIOS,
    ParityResult,
    activation_row_parity,
    extract_root,
    render_mermaid_diagram,
    render_summary_markdown,
    select_parity_pair,
    variants_by_scenario,
    write_extracted_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = PROJECT_ROOT / "scripts" / "profile_neuronpedia_dashboard_generation.py"
NOTEBOOK_TEMPLATE = PROJECT_ROOT / "scripts" / "templates" / "dashboard_profiling_notebook_template.ipynb"
DEFAULT_JEMALLOC = Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so")
DEFAULT_DB_URL = "postgres://postgres:postgres@127.0.0.1:5433/postgres"
DEFAULT_REPO_ROOTS = {
    "SD": Path.home() / "repos" / "SAEDashboard",
    "SL": Path.home() / "repos" / "SAELens",
    "NP": Path.home() / "repos" / "neuronpedia",
    "IT": PROJECT_ROOT,
}

THREEWAY_LEGS = {
    "rte": (
        ("detached_legacy_rte", "detached-legacy-rte-pretokenized-reduced", "det"),
        ("legacy_rte", "legacy-rte-pretokenized-reduced", "legacy"),
        ("columnar_rte", "columnar-rte-pretokenized-reduced", "columnar"),
    ),
    "monology": (
        ("detached_legacy_monology", "detached-legacy-monology-pretokenized-reduced", "det"),
        ("legacy_monology", "legacy-monology-pretokenized-reduced", "legacy"),
        ("columnar_monology", "columnar-monology-pretokenized-reduced", "columnar"),
    ),
}
SCALING_LEGS = {scenario: tuple(leg for leg in legs if leg[2] != "det") for scenario, legs in THREEWAY_LEGS.items()}

# Conservative first-pass default scaling sweeps (features:prompts). Each doubles exactly one dimension of the
# accepted reviewer baseline (RTE 512x128, Monology 1024x256) so the eager current-legacy path stays well inside
# the host-RSS and GPU envelopes observed in prior waves; widen deliberately once these shapes are proven.
DEFAULT_SCALING_CONFIGS = {
    "rte": ("1024:128", "512:256"),
    "monology": ("2048:256", "1024:512"),
}
DEFAULT_PACKAGE_PARENT = Path("/tmp/dashboard_benchmark_packages")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=("threeway", "scaling", "full"),
        default="threeway",
        help=(
            "threeway: 3-way accepted benchmark legs; scaling: 2-way (current legacy + columnar_gpu) batch-shape "
            "sweep; full: both, with the default per-scenario scaling sweeps when --config is omitted."
        ),
    )
    parser.add_argument(
        "--from-existing",
        type=Path,
        default=None,
        help="Skip benchmark execution and package artifacts from this existing artifact root.",
    )
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--target-batches", type=int, default=4)
    parser.add_argument("--summary-warmup-batches", type=int, default=1)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Scaling batch-shape spec features:prompts (repeatable), applied to every swept scenario. When omitted "
            "in scaling/full mode, the conservative per-scenario DEFAULT_SCALING_CONFIGS sweep is used. Ignored in "
            "threeway mode."
        ),
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=None,
        help="Artifact root for new benchmark runs. Defaults to /tmp/np_dashboard_generation_profiles/<mode>_<ts>.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help=(
            "Output directory for the reviewer package. Defaults to "
            f"{DEFAULT_PACKAGE_PARENT}/<session_root_name> so packages accumulate in one reviewable location."
        ),
    )
    parser.add_argument("--local-db-url", default=DEFAULT_DB_URL)
    parser.add_argument("--run-tag", default=None, help="Suffix tag for run names. Defaults to <mode>-<timestamp>.")
    parser.add_argument("--timing-mode", choices=("steady-state", "all-batches"), default="steady-state")
    parser.add_argument(
        "--rolling-coefficient-substage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include deep rolling_* sub-substage aggregation in extracted data (when instrumented runs provide it).",
    )
    parser.add_argument(
        "--dashboard-extra-arg",
        action="append",
        default=[],
        help="Extra arg forwarded to every profiling leg (repeatable).",
    )
    parser.add_argument(
        "--rolling-threads",
        type=int,
        default=8,
        help="Value for --runner-rolling-coefficient-num-threads on every leg. Defaults to 8 (accepted benchmark).",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--jemalloc-path", type=Path, default=DEFAULT_JEMALLOC)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit packaging with dirty repo trees (diagnostic runs only; reviewer packages must be clean).",
    )
    parser.add_argument("--skip-notebook", action="store_true", help="Skip papermill notebook execution/export.")
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip the per-feature activation row parity pass (loads all legacy batch JSONs).",
    )
    return parser


def capture_lineage(repo_roots: dict[str, Path]) -> tuple[dict[str, str], list[str]]:
    """Return {abbr: short_sha} plus the list of dirty repo abbreviations."""

    shas: dict[str, str] = {}
    dirty: list[str] = []
    for abbr, root in repo_roots.items():
        try:
            sha = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sha = "unknown"
            status = ""
        shas[abbr] = sha
        if status:
            dirty.append(abbr)
    return shas, dirty


def lineage_strings(shas: dict[str, str]) -> dict[str, str]:
    current = f"SD-{shas['SD']}/SL-{shas['SL']}/NP-{shas['NP']}/IT-{shas['IT']}"
    detached = f"{DETACHED_BASELINE_LINEAGE_PREFIX}/IT-{shas['IT']}"
    return {"detached_legacy": detached, "current_legacy": current, "columnar_gpu": current}


def build_leg_command(
    args: argparse.Namespace,
    *,
    preset: str,
    variant_dir: str,
    suffix: str,
    session_root: Path,
    config_spec: str | None = None,
) -> list[str]:
    command = [
        args.python_executable,
        str(PROFILE_SCRIPT),
        "--no-py-spy",
        "--layer",
        str(args.layer),
        "--preset",
        preset,
        "--target-batches",
        str(args.target_batches),
        "--summary-warmup-batches",
        str(args.summary_warmup_batches),
        "--session-root",
        str(session_root / variant_dir),
        "--profile-import-stage",
        "--local-db-url",
        args.local_db_url,
        f"--dashboard-extra-arg=--run-name-suffix={suffix}",
        f"--dashboard-extra-arg=--start-layer={args.layer}",
        f"--dashboard-extra-arg=--end-layer={args.layer}",
        f"--dashboard-extra-arg=--end-batch={args.target_batches - 1}",
        "--dashboard-extra-arg=--runner-log-performance",
        f"--dashboard-extra-arg=--runner-rolling-coefficient-num-threads={args.rolling_threads}",
    ]
    if config_spec is not None:
        command.extend(["--config", config_spec])
    for extra in args.dashboard_extra_arg:
        command.append(f"--dashboard-extra-arg={extra}")
    return command


def legs_for_mode(args: argparse.Namespace, scenario: str) -> list[tuple[str, str, str, str | None]]:
    """Return (variant_dir, preset, path_tag, config_spec) legs for one scenario under the requested mode.

    Threeway legs run at the accepted preset shapes (``config_spec=None``); scaling legs sweep the explicit
    ``--config`` specs, falling back to the conservative per-scenario ``DEFAULT_SCALING_CONFIGS``.
    """

    legs: list[tuple[str, str, str, str | None]] = []
    if args.mode in ("threeway", "full"):
        legs.extend((variant_dir, preset, tag, None) for variant_dir, preset, tag in THREEWAY_LEGS[scenario])
    if args.mode in ("scaling", "full"):
        configs = list(args.config) or list(DEFAULT_SCALING_CONFIGS[scenario])
        legs.extend(
            (variant_dir, preset, tag, config_spec)
            for variant_dir, preset, tag in SCALING_LEGS[scenario]
            for config_spec in configs
        )
    return legs


def run_benchmark_legs(args: argparse.Namespace, session_root: Path, run_tag: str) -> None:
    """Run all requested benchmark legs sequentially (parallel runs OOM the single benchmark GPU)."""

    env = os.environ.copy()
    if args.jemalloc_path.exists():
        env["LD_PRELOAD"] = str(args.jemalloc_path)
    else:
        print(f"WARNING: jemalloc not found at {args.jemalloc_path}; running without LD_PRELOAD", flush=True)

    for scenario in args.scenarios:
        for variant_dir, preset, path_tag, config_spec in legs_for_mode(args, scenario):
            leg_variant_dir = variant_dir
            suffix = f"{path_tag}-{run_tag}"
            leg_config = None
            if config_spec is not None:
                config_label = config_spec.replace(":", "x")
                leg_variant_dir = f"{variant_dir}_{config_label}"
                suffix = f"{path_tag}-{config_label}-{run_tag}"
                leg_config = f"{preset}-{config_label}:{config_spec}"
            command = build_leg_command(
                args,
                preset=preset,
                variant_dir=leg_variant_dir,
                suffix=suffix,
                session_root=session_root,
                config_spec=leg_config,
            )
            print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] running leg: {leg_variant_dir}", flush=True)
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=False)
            if completed.returncode != 0:
                print(f"WARNING: leg {leg_variant_dir} exited with code {completed.returncode}", flush=True)


def build_package(args: argparse.Namespace, source_root: Path, package_dir: Path) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "tables").mkdir(exist_ok=True)
    (package_dir / "raw").mkdir(exist_ok=True)

    shas, dirty = capture_lineage(DEFAULT_REPO_ROOTS)
    if dirty and not args.allow_dirty:
        raise SystemExit(
            f"Refusing to package a reviewer benchmark with dirty repos: {', '.join(dirty)}. "
            "Commit first (see the benchmark regeneration reproducibility policy) or pass --allow-dirty "
            "for a disposable diagnostic package."
        )
    lineages = lineage_strings(shas)

    variants = extract_root(
        source_root,
        summary_warmup_batches=args.summary_warmup_batches,
        timing_mode=args.timing_mode,
    )
    if not variants:
        raise SystemExit(f"No successful benchmark variants found under {source_root}")

    parities: dict[str, ParityResult] = {}
    if not args.skip_parity:
        for scenario, group in variants_by_scenario(variants).items():
            pair = select_parity_pair(group)
            if pair is not None:
                print(f"Computing activation row parity for {scenario}...", flush=True)
                parities[scenario] = activation_row_parity(*pair)

    for variant in variants:
        variant.lineage = lineages.get(variant.path_key)
        raw_dir = package_dir / "raw" / (variant.variant_dir or variant.label)
        raw_dir.mkdir(parents=True, exist_ok=True)
        leaf = Path(variant.leaf_dir)
        for name in ("result.json", "runner_perf_events.json", "import_stage_profile.json", "stage_summary.md"):
            source = leaf / name
            if source.exists():
                shutil.copy2(source, raw_dir / name)
        session_results = leaf.parent / "results.json"
        if session_results.exists():
            shutil.copy2(session_results, raw_dir / "results.json")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "mode": args.mode,
        "timing_mode": args.timing_mode,
        "summary_warmup_batches": args.summary_warmup_batches,
        "target_batches": args.target_batches,
        "layer": args.layer,
        "repo_heads": shas,
        "dirty_repos": dirty,
        "lineages": lineages,
        "variants": [
            {"variant_dir": v.variant_dir, "label": v.label, "scenario": v.scenario, "path": v.path_key}
            for v in variants
        ],
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_extracted_data(package_dir, variants, parities)

    from dashboard_benchmark_artifacts import (  # local import to keep table renderers grouped
        render_import_table,
        render_parity_table,
        render_primary_table,
        render_resource_table,
        render_substage_table,
    )

    for scenario, group in variants_by_scenario(variants).items():
        tables_dir = package_dir / "tables"
        (tables_dir / f"primary_{scenario}.md").write_text(render_primary_table(group) + "\n", encoding="utf-8")
        (tables_dir / f"substage_{scenario}.md").write_text(render_substage_table(group) + "\n", encoding="utf-8")
        (tables_dir / f"import_{scenario}.md").write_text(render_import_table(group) + "\n", encoding="utf-8")
        (tables_dir / f"resource_{scenario}.md").write_text(render_resource_table(group) + "\n", encoding="utf-8")
        if scenario in parities:
            (tables_dir / f"parity_{scenario}.md").write_text(
                render_parity_table(parities[scenario]) + "\n", encoding="utf-8"
            )

    diagram = render_mermaid_diagram(variants, lineages=lineages)
    (package_dir / "dashboard_benchmark_diagram.mmd").write_text(diagram + "\n", encoding="utf-8")

    notebook_name = None
    if not args.skip_notebook:
        notebook_name = f"dashboard_profiling_{timestamp}.ipynb"
        notebook_path = package_dir / notebook_name
        print(f"Executing profiling notebook via papermill -> {notebook_path}", flush=True)
        import papermill  # type: ignore[import-untyped]

        papermill.execute_notebook(
            str(NOTEBOOK_TEMPLATE),
            str(notebook_path),
            parameters={"package_path": str(package_dir)},
            cwd=str(package_dir),
            progress_bar=False,
        )
        subprocess.run(
            [args.python_executable, "-m", "nbconvert", "--to", "html", str(notebook_path)],
            check=False,
        )

    summary = render_summary_markdown(variants, parities, manifest=manifest, notebook_name=notebook_name)
    (package_dir / "benchmark_summary.md").write_text(summary, encoding="utf-8")
    print(f"Reviewer artifact package written to {package_dir}", flush=True)
    return package_dir


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_tag or f"{args.mode}-{timestamp}"

    if args.from_existing is not None:
        source_root = args.from_existing
    else:
        source_root = args.session_root or Path(f"/tmp/np_dashboard_generation_profiles/{args.mode}_{timestamp}")
        source_root.mkdir(parents=True, exist_ok=True)
        run_benchmark_legs(args, source_root, run_tag)

    package_dir = args.package_root or DEFAULT_PACKAGE_PARENT / source_root.name
    build_package(args, source_root, package_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
