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
from typing import Any

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
    "SD": Path(os.getenv("SAEDASHBOARD_REPO_ROOT", str(Path.home() / "repos" / "SAEDashboard"))),
    "SL": Path(os.getenv("SAELENS_REPO_ROOT", str(Path.home() / "repos" / "SAELens"))),
    "NP": Path(os.getenv("NEURONPEDIA_REPO_ROOT", str(Path.home() / "repos" / "neuronpedia"))),
    "IT": PROJECT_ROOT,
}
# Coordination-PR reference stamped into the manifest/summary/notebook; override with
# --coordination-pr-url (re-render an existing package via --from-existing to update it).
DEFAULT_COORDINATION_PR_URL = "<COORDINATION_PR_URL — backfill at wave-open>"

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

# Default scaling sweeps. Config spec is features:prompts[:nprompts] (or label:features:prompts[:nprompts]),
# where `prompts` is the forward-pass minibatch and the optional `nprompts` is the TOTAL prompt count —
# the prompt-dimension sweep axis. The default sweeps below vary the FEATURE axis (columnar_gpu
# throughput scales there). NOTE: prompt-axis scaling is NOT throughput-neutral at scale — total-prompt
# scaling approaches a GPU-memory cliff whose position depends on per-layer activation density and the
# device memory available. To run the prompt-dimension sweep, pass explicit specs, e.g.
# `--config 4096:256:2490 --config 4096:256:4096` and run per-layer (a sparse and a dense layer bound
# the range). Per-point peak GPU memory is captured (cuda_reserved_gib) for the memory-vs-timing view.
DEFAULT_SCALING_CONFIGS = {
    "rte": ("1024:128", "2048:128"),
    "monology": ("2048:256", "4096:256"),
}
# Default prompt-dimension sweep (fixed 4096×256 batch shape, swept total prompts {2490, 4096, 24576}).
# Runs automatically in full mode on `--prompt-sweep-layer` (columnar path only, under the opt-in
# reduced-peak-memory flags below — the largest total-prompt points can exceed device memory on some
# layer/hardware combinations without them); also usable via explicit `--config` specs in scaling mode.
PROMPT_DIM_SWEEP_MONOLOGY = ("4096:256:2490", "4096:256:4096", "4096:256:24576")
# Opt-in reduced-peak-memory flags for the sweep legs (bit-identical outputs; they trade throughput for
# peak GPU memory so large total-prompt points fit that would otherwise OOM). Applied ONLY to the
# prompt-sweep legs so the main benchmark legs keep the default fast-path staging behavior. The
# row-chunk value bounds the packaging/selection per-chunk transients, which scale with per-layer
# candidate volume (denser layers need smaller chunks at a given total-prompt count); 16 leaves margin
# across layers at the default sweep sizes on the reference hardware.
PROMPT_SWEEP_MITIGATION_ARGS = (
    "--runner-columnar-max-staged-acts-bytes=0",
    "--runner-columnar-row-chunk-size=16",
)
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
    parser.add_argument(
        "--monology-sweep",
        choices=("pretok", "streaming"),
        default=None,
        help=(
            "Swap the default pretokenized Monology scaling presets for a prompt-dimension sweep variant, "
            "so total prompts can exceed the preset cap (e.g. --config 4096:256:4096 --config "
            "4096:256:8192). 'pretok' uses the largest available concat_<n> pretokenized set (reproducible, "
            "HF-independent for prompts; one set serves all smaller totals); 'streaming' uses load_dataset "
            "over monology/pile-uncopyrighted (for totals beyond the largest built concat set). Monology "
            "scaling legs only; build pretok sets with the command in docs/neuronpedia_dashboard_pipeline.md."
        ),
    )
    parser.add_argument(
        "--path-tags",
        nargs="+",
        choices=("det", "legacy", "columnar"),
        default=None,
        help=(
            "Restrict legs to these path tags (default: all for the mode). E.g. `--path-tags columnar` runs a "
            "clean-room columnar-only sweep — scaling mode otherwise runs legacy legs first, and a legacy OOM "
            "near the GPU memory ceiling can fragment the allocator for the following columnar legs, adding "
            "noise to their peak-memory and timing readings."
        ),
    )
    parser.add_argument(
        "--prompt-sweep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include the default Monology n-prompt scaling sweep legs ({2490, 4096, 24576} at 4096x256, "
            "columnar only, opt-in reduced-peak-memory flags) in the artifact. Default: ON in full mode, "
            "OFF otherwise."
        ),
    )
    parser.add_argument(
        "--prompt-sweep-layer",
        type=int,
        default=12,
        help=(
            "Layer for the n-prompt scaling sweep legs (default 12 — a representative middle layer). "
            "NOTE: a single-layer curve understates the OOM ceiling set by the densest layer."
        ),
    )
    parser.add_argument(
        "--prompt-sweep-config",
        action="append",
        default=[],
        help=(
            "n-prompt sweep spec features:prompts:nprompts (repeatable). Defaults to the canonical "
            f"{', '.join(PROMPT_DIM_SWEEP_MONOLOGY)} sweep."
        ),
    )
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
        "--coordination-pr-url",
        default=DEFAULT_COORDINATION_PR_URL,
        help="Scalable Dashboards coordination PR URL referenced by the packaged summary/notebook. "
        "Defaults to a backfill placeholder; update a shipped package with --from-existing + this flag.",
    )
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
    layer: int | None = None,
    extra_dashboard_args: tuple[str, ...] = (),
) -> list[str]:
    leg_layer = args.layer if layer is None else layer
    command = [
        args.python_executable,
        str(PROFILE_SCRIPT),
        "--no-py-spy",
        "--layer",
        str(leg_layer),
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
        f"--dashboard-extra-arg=--start-layer={leg_layer}",
        f"--dashboard-extra-arg=--end-layer={leg_layer}",
        f"--dashboard-extra-arg=--end-batch={args.target_batches - 1}",
        "--dashboard-extra-arg=--runner-log-performance",
        f"--dashboard-extra-arg=--runner-rolling-coefficient-num-threads={args.rolling_threads}",
    ]
    if config_spec is not None:
        command.extend(["--config", config_spec])
    for extra in args.dashboard_extra_arg:
        command.append(f"--dashboard-extra-arg={extra}")
    for extra in extra_dashboard_args:
        command.append(f"--dashboard-extra-arg={extra}")
    return command


# Prompt-dimension sweep: swap the default pretokenized (preset-capped) Monology scaling presets for a
# sweep-capable variant so total prompts can exceed the preset cap. --monology-sweep=pretok uses the
# largest available concat_<n> pretokenized set (reproducible, HF-independent for prompts; one set serves
# all smaller totals); --monology-sweep=streaming uses load_dataset (for totals beyond the largest built
# concat set).
MONOLOGY_SWEEP_PRESETS = {
    "pretok": {
        # legacy can't load_from_disk the pretokenized set, so it streams the same first-N prompts; only
        # columnar consumes the concat set directly (that is where the packaging cliff lives).
        "legacy-monology-pretokenized-reduced": "legacy-monology-streaming",
        "columnar-monology-pretokenized-reduced": "columnar-monology-pretok-sweep",
    },
    "streaming": {
        "legacy-monology-pretokenized-reduced": "legacy-monology-streaming",
        "columnar-monology-pretokenized-reduced": "columnar-monology-streaming",
    },
}


def legs_for_mode(args: argparse.Namespace, scenario: str) -> list[tuple[str, str, str, str | None]]:
    """Return (variant_dir, preset, path_tag, config_spec) legs for one scenario under the requested mode.

    Threeway legs run at the accepted preset shapes (``config_spec=None``); scaling legs sweep the explicit
    ``--config`` specs, falling back to the conservative per-scenario ``DEFAULT_SCALING_CONFIGS``. When
    ``--monology-sweep`` is set, the Monology scaling legs use the prompt-dimension sweep presets
    (``pretok`` = largest available concat_<n> pretokenized set, ``streaming`` = load_dataset) so the
    sweep (``features:prompts:nprompts``) can exceed the pretokenized preset cap.
    """

    legs: list[tuple[str, str, str, str | None]] = []
    if args.mode in ("threeway", "full"):
        legs.extend((variant_dir, preset, tag, None) for variant_dir, preset, tag in THREEWAY_LEGS[scenario])
    if args.mode in ("scaling", "full"):
        configs = list(args.config) or list(DEFAULT_SCALING_CONFIGS[scenario])
        sweep = getattr(args, "monology_sweep", None)
        preset_map = MONOLOGY_SWEEP_PRESETS.get(sweep, {}) if scenario == "monology" else {}
        for variant_dir, preset, tag in SCALING_LEGS[scenario]:
            resolved_preset = preset_map.get(preset, preset)
            legs.extend((variant_dir, resolved_preset, tag, config_spec) for config_spec in configs)
    path_tags = getattr(args, "path_tags", None)
    if path_tags:
        legs = [leg for leg in legs if leg[2] in path_tags]
    return legs


def prompt_sweep_enabled(args: argparse.Namespace) -> bool:
    """Prompt-sweep legs default ON in full mode, OFF otherwise; explicit flag wins."""
    if args.prompt_sweep is not None:
        return bool(args.prompt_sweep)
    return args.mode == "full"


def prompt_sweep_configs(args: argparse.Namespace) -> list[str]:
    return list(args.prompt_sweep_config) or list(PROMPT_DIM_SWEEP_MONOLOGY)


def resolve_gpu_device_total_mib() -> int | None:
    """Total memory (MiB) of the benchmark GPU, recorded so packaged charts can draw a data-driven device-memory
    ceiling instead of assuming a particular card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def resolve_pretokenization_record(args: argparse.Namespace | None = None) -> dict[str, Any] | None:
    """Read the pretokenization run record for a dataset THIS run actually used.

    Records are written by sae_dashboard.neuronpedia.prompt_pretokenization beside the dataset. The previous behavior
    always read the monology SWEEP dataset's record, which embedded a different run's prep (wrong cardinality +
    timestamp lineage) into the manifest/diagram whenever a mode used the reduced datasets instead. Now: consult the
    datasets for the modes' accepted presets in preference order and tag the returned record with its dataset so
    downstream renderers can attribute it; return None (renderers show TBD) when none of the run's datasets carry a
    record — never a record from a dataset the run didn't touch.
    """
    try:
        from profile_neuronpedia_dashboard_generation import (
            DEFAULT_MONOLOGY_PROMPT_SWEEP_PRETOKENIZED_DATASET,
            DEFAULT_PHASE3_MONOLOGY_PRETOKENIZED_DATASET,
            DEFAULT_PHASE3_RTE_PRETOKENIZED_DATASET,
        )
    except Exception:
        return None
    candidates: list[Path] = []
    mode = getattr(args, "mode", None) if args is not None else None
    if mode == "scaling":
        candidates.append(Path(DEFAULT_MONOLOGY_PROMPT_SWEEP_PRETOKENIZED_DATASET))
    # threeway/full accepted presets use the reduced monology + rte pretokenized datasets
    candidates.extend(
        [Path(DEFAULT_PHASE3_MONOLOGY_PRETOKENIZED_DATASET), Path(DEFAULT_PHASE3_RTE_PRETOKENIZED_DATASET)]
    )
    if mode == "full":
        candidates.append(Path(DEFAULT_MONOLOGY_PROMPT_SWEEP_PRETOKENIZED_DATASET))
    for dataset_path in candidates:
        record_path = dataset_path / "pretokenization_run.json"
        if not record_path.exists():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record["dataset"] = dataset_path.name
        return record
    return None


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

    if prompt_sweep_enabled(args) and "monology" in args.scenarios:
        sweep_preset = MONOLOGY_SWEEP_PRESETS["pretok"]["columnar-monology-pretokenized-reduced"]
        for config_spec in prompt_sweep_configs(args):
            config_label = config_spec.replace(":", "x")
            leg_variant_dir = f"columnar_monology_promptsweep_{config_label}"
            suffix = f"columnar-promptsweep-{config_label}-{run_tag}"
            command = build_leg_command(
                args,
                preset=sweep_preset,
                variant_dir=leg_variant_dir,
                suffix=suffix,
                session_root=session_root,
                config_spec=f"{sweep_preset}-{config_label}:{config_spec}",
                layer=args.prompt_sweep_layer,
                extra_dashboard_args=PROMPT_SWEEP_MITIGATION_ARGS,
            )
            print(
                f"[{datetime.now(UTC).isoformat(timespec='seconds')}] running leg: {leg_variant_dir} "
                f"(prompt sweep, layer {args.prompt_sweep_layer})",
                flush=True,
            )
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
        "scenarios": list(args.scenarios),
        "config": list(args.config),
        "monology_sweep": getattr(args, "monology_sweep", None),
        "path_tags": getattr(args, "path_tags", None),
        "prompt_sweep": prompt_sweep_enabled(args),
        "prompt_sweep_layer": args.prompt_sweep_layer,
        "prompt_sweep_configs": prompt_sweep_configs(args) if prompt_sweep_enabled(args) else [],
        "pretokenization_record": resolve_pretokenization_record(args),
        "prompt_sweep_mitigation_args": list(PROMPT_SWEEP_MITIGATION_ARGS) if prompt_sweep_enabled(args) else [],
        "gpu_device_total_mib": resolve_gpu_device_total_mib(),
        "invocation": "python scripts/run_dashboard_benchmark_suite.py " + " ".join(sys.argv[1:]),
        "coordination_pr_url": args.coordination_pr_url,
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

    diagram = render_mermaid_diagram(
        variants,
        lineages=lineages,
        pretokenization_record=manifest.get("pretokenization_record"),
    )
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
            parameters={"package_path": str(package_dir), "coordination_pr_url": args.coordination_pr_url},
            cwd=str(package_dir),
            progress_bar=False,
        )
        # --no-input: the HTML export is data/charts-only; sources stay in the .ipynb
        # (collapsed by default, expandable in JupyterLab).
        subprocess.run(
            [args.python_executable, "-m", "nbconvert", "--to", "html", "--no-input", str(notebook_path)],
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
