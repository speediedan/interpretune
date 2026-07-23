#!/usr/bin/env python3
"""Set up the full local environment for the Neuronpedia dashboard benchmark suite.

One guided, transparent, non-destructive flow that prepares everything
`scripts/run_dashboard_benchmark_suite.py` needs (Linux and macOS):

1. Locate (or clone) the source repos: interpretune, SAEDashboard, SAELens, neuronpedia,
   TransformerLens, nnsight, circuit-tracer. Existing checkouts are never switched or
   modified; dirty trees are surfaced with an offer to stash (or continue/abort).
2. Create the detached preserved-baseline worktrees (the pre-PR comparison lineage
   `SD-7886eaa+benchmark_patches / SL-3eea6552 / NP-5a33f17`) in a directory you choose,
   applying the audited benchmark patches from `scripts/benchmark_baseline_patches/`
   (see that directory's README for the per-patch classification/rationale) and
   verifying the resulting tree state against pinned expectations.
3. Check the local Neuronpedia Postgres is reachable (offering the docker compose
   bring-up from your neuronpedia checkout if it is not).
4. Build the integrated interpretune benchmark venv via `scripts/build_it_env.sh`
   (the canonical multi-repo from-source build). An existing venv is only cleared
   after explicit confirmation (or `--clear-existing-venv`).
5. Verify the benchmark prompt datasets exist under `$IT_NP_CACHE` (pointing at the
   regeneration docs when they do not).
6. Write `<worktrees-dir>/benchmark_env.sh` capturing every environment variable the
   suite needs, and print the exact command to run the benchmark.

Every mutating action is printed before it runs; use `--dry-run` to see the full plan
without executing anything, and `--yes` to accept defaults non-interactively (any
explicitly passed flag always wins over a prompt).

Requirements: git, docker (only if the local DB needs bring-up), `uv`, and bash >= 4.3
for the env build (macOS: `brew install bash`). Root is never required. Nothing is
pushed and no existing checkout is modified.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

SCRIPT_DIR = Path(__file__).resolve().parent
IT_ROOT = SCRIPT_DIR.parent
PATCHES_DIR = SCRIPT_DIR / "benchmark_baseline_patches"

WAVE_BRANCH = "streamlined-streamable-dashboard-generation-phase-1"
DEFAULT_DB_URL = "postgres://postgres:postgres@127.0.0.1:5433/postgres"

# Preserved pre-PR baseline pins (lineage `SD-7886eaa+benchmark_patches/SL-3eea6552/NP-5a33f17`).
SD_BASELINE_SHA = "7886eaa227398a52cd77a4483c94ecc74d204d34"
SL_BASELINE_SHA = "3eea65526345e0df384a7c89b3c7f9d6f541d687"
NP_BASELINE_SHA = "5a33f178e828ed5eb35e90a57b81807ee73d2153"

# Benchmark patches applied (in this order) on top of the clean SD baseline commit.
SD_BASELINE_PATCHES = (
    "saedashboard-7886eaa-profiling.patch",
    "saedashboard-7886eaa-activation-significance-floor.patch",
    "saedashboard-7886eaa-no-shuffle-tokens.patch",
    "saedashboard-7886eaa-shared-tokens-file.patch",
)
# Expected tracked-diff numstat (added, deleted, path) after all patches apply, plus the
# one new untracked file — pinned so drift in either the patches or the baseline is loud.
SD_BASELINE_EXPECTED_NUMSTAT = {
    ("193", "41", "sae_dashboard/feature_data_generator.py"),
    ("104", "21", "sae_dashboard/neuronpedia/neuronpedia_runner.py"),
    ("5", "0", "sae_dashboard/neuronpedia/neuronpedia_runner_config.py"),
    ("5", "0", "sae_dashboard/sae_vis_data.py"),
    ("163", "101", "sae_dashboard/sae_vis_runner.py"),
    ("108", "24", "sae_dashboard/utils_fns.py"),
}
SD_BASELINE_EXPECTED_NEW_FILE = "sae_dashboard/perf_logging.py"

# Datasets the accepted-shape benchmark presets require under $IT_NP_CACHE (threeway mode).
REQUIRED_DATASETS = (
    "pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts",
    "legacy_pretokenized/gemma-3-1b-it_rte_boolq_context319_fixed_pad_2490",
    "pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_2490",
    "legacy_pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_2490",
)
# Additionally required by --mode full (the prompt-dimension sweep).
FULL_MODE_DATASETS = ("pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_24576",)


@dataclass
class RepoSpec:
    key: str
    dirname: str
    url: str
    ref: str | None  # branch/tag checked out for FRESH clones only; existing checkouts untouched
    note: str = ""


REPOS: tuple[RepoSpec, ...] = (
    RepoSpec("interpretune", "interpretune", "https://github.com/speediedan/interpretune.git", WAVE_BRANCH),
    RepoSpec("sae_dashboard", "SAEDashboard", "https://github.com/speediedan/SAEDashboard.git", WAVE_BRANCH),
    RepoSpec("sae_lens", "SAELens", "https://github.com/speediedan/SAELens.git", WAVE_BRANCH),
    RepoSpec("neuronpedia", "neuronpedia", "https://github.com/speediedan/neuronpedia.git", WAVE_BRANCH),
    RepoSpec(
        "transformer_lens",
        "TransformerLens",
        "https://github.com/TransformerLensOrg/TransformerLens.git",
        "v3.5.1",
        note="pinned to the validated TL release (matches interpretune's override-dependencies pin)",
    ),
    RepoSpec(
        "nnsight",
        "nnsight",
        "https://github.com/ndif-team/nnsight.git",
        "0.7.0",
        note="pinned to the validated nnsight release",
    ),
    RepoSpec(
        "circuit_tracer",
        "circuit-tracer",
        "https://github.com/speediedan/circuit-tracer.git",
        None,
        note="not exercised by the dashboard benchmarks; any installable head of the fork works",
    ),
)


class Setup:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.actions_taken: list[str] = []
        self.warnings: list[str] = []
        self.repo_paths: dict[str, Path] = {}

    # ---------- io helpers ----------

    def say(self, msg: str) -> None:
        print(msg, flush=True)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"WARNING: {msg}", flush=True)

    def fail(self, msg: str) -> None:
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    def confirm(self, question: str, default: bool = True) -> bool:
        if self.args.yes:
            self.say(f"{question} [auto-{'yes' if default else 'no'} via --yes]")
            return default
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            resp = input(f"{question} {suffix} ").strip().lower()
            if not resp:
                return default
            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                return False

    def choose(self, question: str, choices: dict[str, str], default: str) -> str:
        """choices: {key: description}; returns the chosen key."""
        if self.args.yes:
            self.say(f"{question} [auto-'{default}' via --yes]")
            return default
        menu = ", ".join(f"[{k}] {v}" for k, v in choices.items())
        while True:
            resp = input(f"{question} ({menu}; default {default}): ").strip().lower()
            if not resp:
                return default
            if resp in choices:
                return resp

    def run(self, cmd: list[str], *, cwd: Path | None = None, mutating: bool = True, check: bool = True) -> str:
        loc = f" (in {cwd})" if cwd else ""
        if mutating:
            self.say(f"  $ {' '.join(cmd)}{loc}")
            if self.args.dry_run:
                self.actions_taken.append(f"[dry-run] {' '.join(cmd)}{loc}")
                return ""
            self.actions_taken.append(f"{' '.join(cmd)}{loc}")
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if check and result.returncode != 0:
            self.fail(f"command failed ({result.returncode}): {' '.join(cmd)}{loc}\n{result.stderr.strip()}")
        return result.stdout.strip()

    # ---------- steps ----------

    def resolve_repos(self) -> None:
        self.say("\n=== Step 1/6: source repositories ===")
        repos_root = Path(self.args.repos_root).expanduser()
        for spec in REPOS:
            override = getattr(self.args, spec.key, None)
            path = Path(override).expanduser() if override else repos_root / spec.dirname
            if spec.key == "interpretune" and override is None:
                path = IT_ROOT  # the checkout this script runs from
            if path.is_dir() and (path / ".git").exists():
                head = self.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], mutating=False)
                branch = self.run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], mutating=False)
                self.say(f"- {spec.dirname}: using existing checkout {path} ({branch}@{head})")
                if spec.ref and branch != spec.ref and spec.ref == WAVE_BRANCH:
                    self.warn(
                        f"{spec.dirname} is on '{branch}', not the expected wave branch '{spec.ref}'. "
                        "This script never switches branches for you; switch manually if intended."
                    )
                dirty = self.run(["git", "-C", str(path), "status", "--porcelain"], mutating=False)
                if dirty:
                    self.say(
                        f"  {spec.dirname} has uncommitted changes:\n    " + "\n    ".join(dirty.splitlines()[:12])
                    )
                    action = self.choose(
                        f"  {spec.dirname} is dirty — stash, continue as-is, or abort?",
                        {"s": "git stash push -u", "c": "continue with dirty tree", "a": "abort"},
                        "c",
                    )
                    if action == "a":
                        self.fail("aborted at user request (dirty repo)")
                    if action == "s":
                        self.run(
                            ["git", "-C", str(path), "stash", "push", "-u", "-m", "setup_dashboard_benchmark_env"],
                            cwd=None,
                        )
                    elif spec.key in ("interpretune", "sae_dashboard", "sae_lens", "neuronpedia"):
                        self.warn(
                            f"{spec.dirname} left dirty — reviewer packaging refuses dirty manifest repos "
                            "(run_dashboard_benchmark_suite.py needs clean SD/SL/NP/IT trees unless --allow-dirty)."
                        )
            else:
                if spec.note:
                    self.say(f"- {spec.dirname}: note: {spec.note}")
                if not self.confirm(f"- {spec.dirname}: missing at {path}; clone {spec.url} -> {path}?"):
                    self.fail(f"{spec.dirname} is required for the benchmark environment")
                self.run(["git", "clone", spec.url, str(path)])
                if spec.ref and not self.args.dry_run:
                    self.run(["git", "-C", str(path), "checkout", spec.ref])
            self.repo_paths[spec.key] = path

    def _ensure_commit(self, repo: Path, sha: str, fallback_ref: str | None) -> None:
        if (
            subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True
            ).returncode
            == 0
        ):
            return
        self.say(f"  commit {sha[:7]} not present locally; fetching from origin...")
        if fallback_ref:
            self.run(["git", "-C", str(repo), "fetch", "origin", fallback_ref])
        else:
            self.run(["git", "-C", str(repo), "fetch", "--all"])
        if (
            not self.args.dry_run
            and subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True
            ).returncode
            != 0
        ):
            self.fail(f"baseline commit {sha} unreachable in {repo} even after fetch")

    def create_worktrees(self) -> None:
        self.say("\n=== Step 2/6: detached preserved-baseline worktrees ===")
        wt_root = Path(self.args.worktrees_dir).expanduser()
        self.say(f"Baseline worktrees root: {wt_root}")
        if not self.args.dry_run:
            wt_root.mkdir(parents=True, exist_ok=True)
        plans = (
            ("sae_dashboard", "SAEDashboard-7886eaa", SD_BASELINE_SHA, "benchmark-baseline-7886eaa", True),
            ("sae_lens", "SAELens-3eea6552", SL_BASELINE_SHA, "benchmark-baseline-3eea6552", False),
            ("neuronpedia", "neuronpedia-5a33f17", NP_BASELINE_SHA, None, False),
        )
        for repo_key, wt_name, sha, fallback_ref, patched in plans:
            repo = self.repo_paths[repo_key]
            wt_path = wt_root / wt_name
            if wt_path.exists():
                self.say(f"- {wt_name}: already exists; verifying instead of recreating")
                self._verify_worktree(wt_path, sha, patched)
                continue
            self._ensure_commit(repo, sha, fallback_ref)
            self.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(wt_path), sha])
            if patched:
                self._apply_baseline_patches(wt_path)
            self._verify_worktree(wt_path, sha, patched)

    def _apply_baseline_patches(self, wt_path: Path) -> None:
        self.say(f"  applying {len(SD_BASELINE_PATCHES)} benchmark patches (see benchmark_baseline_patches/README.md)")
        for patch in SD_BASELINE_PATCHES:
            patch_path = PATCHES_DIR / patch
            if not patch_path.is_file():
                self.fail(f"missing vendored patch {patch_path}")
            if self.args.dry_run and not wt_path.exists():
                self.say(f"  $ git -C {wt_path} apply {patch_path}")
                continue
            self.run(["git", "-C", str(wt_path), "apply", "--check", str(patch_path)], mutating=False)
            self.run(["git", "-C", str(wt_path), "apply", str(patch_path)])

    def _verify_worktree(self, wt_path: Path, sha: str, patched: bool) -> None:
        if self.args.dry_run and not wt_path.exists():
            return
        head = self.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"], mutating=False)
        if head != sha:
            self.fail(f"{wt_path} HEAD is {head[:9]}, expected {sha[:9]}")
        if not patched:
            if self.run(["git", "-C", str(wt_path), "status", "--porcelain"], mutating=False):
                self.fail(f"{wt_path} should be a clean checkout of {sha[:9]} but has local changes")
            self.say(f"  OK: {wt_path.name} = clean {sha[:9]}")
            return
        numstat = {
            tuple(line.split("\t"))
            for line in self.run(["git", "-C", str(wt_path), "diff", "--numstat"], mutating=False).splitlines()
        }
        new_file_ok = (wt_path / SD_BASELINE_EXPECTED_NEW_FILE).is_file()
        if numstat != SD_BASELINE_EXPECTED_NUMSTAT or not new_file_ok:
            self.fail(
                f"{wt_path.name} post-patch state does not match the pinned expectation.\n"
                f"  expected numstat: {sorted(SD_BASELINE_EXPECTED_NUMSTAT)}\n"
                f"  actual numstat:   {sorted(numstat)}\n"
                f"  {SD_BASELINE_EXPECTED_NEW_FILE} present: {new_file_ok}"
            )
        self.say(f"  OK: {wt_path.name} = {sha[:9]} + benchmark patches (verified against pinned numstat)")

    def check_db(self) -> None:
        self.say("\n=== Step 3/6: local Neuronpedia Postgres ===")
        if self.args.skip_services_check:
            self.say("- skipped (--skip-services-check)")
            return
        parsed = urlsplit(self.args.local_db_url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 5432
        try:
            with socket.create_connection((host, port), timeout=3):
                self.say(f"- Postgres reachable at {host}:{port} — the DB-import benchmark legs can run.")
                return
        except OSError:
            pass
        self.warn(f"Postgres NOT reachable at {host}:{port}.")
        np_repo = self.repo_paths.get("neuronpedia")
        compose = np_repo / "docker" / "compose.yaml" if np_repo else None
        if compose and compose.is_file() and shutil.which("docker"):
            env_file = np_repo / ".env"
            if not env_file.exists():
                self.say(f"  creating empty {env_file} (gitignored; compose requires it to exist)")
                if not self.args.dry_run:
                    env_file.touch()
            cmd = [
                "docker",
                "compose",
                "-f",
                "docker/compose.yaml",
                "--env-file",
                ".env.localhost",
                "--env-file",
                ".env",
                "up",
                "-d",
                "db-init",
                "postgres",
            ]
            if self.confirm(f"  bring up the local Neuronpedia DB now? ({' '.join(cmd)} in {np_repo})"):
                self.run(cmd, cwd=np_repo)
            else:
                self.warn("continuing without a reachable DB — import legs will fail until it is up.")
        else:
            self.warn(
                "docker or the neuronpedia compose file is unavailable; start Postgres yourself "
                "(see neuronpedia Makefile target 'webapp-localhost-run') or pass --local-db-url."
            )

    def build_env(self) -> Path:
        self.say("\n=== Step 4/6: interpretune benchmark venv (build_it_env.sh) ===")
        venv_dir = Path(self.args.venv_dir).expanduser()
        venv_path = venv_dir / self.args.venv_name
        if self.args.skip_env_build:
            self.say(f"- skipped (--skip-env-build); assuming venv at {venv_path}")
            return venv_path
        if venv_path.exists():
            self.say(f"- venv already exists at {venv_path}; the build CLEARS and rebuilds it in place.")
            if not self.args.clear_existing_venv:
                action = self.choose(
                    "  rebuild (clears the existing venv), keep it and skip the build, or abort?",
                    {"r": "rebuild (clear)", "k": "keep + skip build", "a": "abort"},
                    "k",
                )
                if action == "a":
                    self.fail("aborted at user request (existing venv)")
                if action == "k":
                    return venv_path
        it, ov = self.repo_paths["interpretune"], "requirements/ci/overrides.txt"
        ex, slr = "requirements/ci/excludes.txt", "requirements/ci/sl_uv_requirements.txt"
        cmd = [
            str(it / "scripts" / "build_it_env.sh"),
            f"--repo-home={it}",
            f"--target-env-name={self.args.venv_name}",
            f"--venv-dir={venv_dir}",
            f"--torch-backend={self.args.torch_backend}",
            f"--from-source=nnsight:{self.repo_paths['nnsight']}:all:UV_OVERRIDE={it / ov}",
            (
                f"--from-source=sae_dashboard:{self.repo_paths['sae_dashboard']}:dev"
                f":UV_EXCLUDE={it / ex}:UV_OVERRIDE={it / ov}"
            ),
            f"--from-source=sae_lens:{self.repo_paths['sae_lens']}:dev:UV_OVERRIDE={it / ov}:FLAGS=-r {it / slr}",
            (
                f"--from-source=circuit_tracer:{self.repo_paths['circuit_tracer']}:dev"
                f":UV_EXCLUDE={it / ex}:UV_OVERRIDE={it / ov}"
            ),
            f"--from-source=transformer-lens:{self.repo_paths['transformer_lens']}",
        ]
        self.say("- canonical multi-repo from-source build (this can take a while; ~10-30 min):")
        if not self.confirm("  run the env build now?"):
            self.warn("env build skipped — activate/build a suitable venv before running the suite.")
            return venv_path
        self.say(f"  $ {' '.join(cmd)}")
        if not self.args.dry_run:
            self.actions_taken.append(" ".join(cmd))
            result = subprocess.run(cmd, cwd=it)
            if result.returncode != 0:
                self.fail(f"build_it_env.sh failed with exit code {result.returncode}")
        ct_override = self.repo_paths["circuit_tracer"] / "temp_hf_override.txt"
        if not ct_override.exists() and not self.args.dry_run:
            self.say(f"  recreating circuit-tracer's untracked {ct_override.name} (transformers-v5 floor)")
            ct_override.write_text("transformers>=5.0.0\nhuggingface_hub>=1.0.0\n", encoding="utf-8")
        # neuronpedia-utils (the Python local-DB import utilities) is not part of the canonical
        # build command — install it editable from the current neuronpedia checkout. --no-deps
        # because its pins would fight the integrated env; the only runtime dep the integrated
        # env does not already satisfy is pgpq (binary Arrow COPY for the columnar import lane).
        np_utils = self.repo_paths["neuronpedia"] / "utils" / "neuronpedia-utils"
        venv_python = str(venv_path / "bin" / "python")
        self.run(["uv", "pip", "install", "--python", venv_python, "-e", str(np_utils), "--no-deps"])
        self.run(["uv", "pip", "install", "--python", venv_python, "pgpq>=0.11,<0.12"])
        return venv_path

    def check_datasets(self) -> None:
        self.say("\n=== Step 5/6: benchmark prompt datasets ===")
        cache = Path(self.args.np_cache).expanduser()
        missing = [d for d in REQUIRED_DATASETS if not (cache / d).is_dir()]
        missing_full = [d for d in FULL_MODE_DATASETS if not (cache / d).is_dir()]
        for d in REQUIRED_DATASETS + FULL_MODE_DATASETS:
            status = "OK" if (cache / d).is_dir() else "MISSING"
            self.say(f"- [{status}] {cache / d}")
        if missing or missing_full:
            self.warn(
                "some benchmark prompt datasets are missing (they are not published to the HF Hub). "
                "Build them once per docs/neuronpedia_dashboard_pipeline.md § 'Regenerating the benchmark "
                "prompt datasets' (threeway mode needs the four accepted-shape sets; --mode full additionally "
                "needs the concat_24576 sweep set)."
            )
            if missing and not self.confirm("  continue anyway (you can build the datasets afterwards)?"):
                self.fail("aborted at user request (missing datasets)")

    def write_env_file(self, venv_path: Path) -> None:
        self.say("\n=== Step 6/6: benchmark_env.sh + next steps ===")
        wt_root = Path(self.args.worktrees_dir).expanduser()
        cache = Path(self.args.np_cache).expanduser()
        np_repo = self.repo_paths["neuronpedia"]
        lines = [
            "# Generated by scripts/setup_dashboard_benchmark_env.py "
            f"({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})",
            "# source this before running scripts/run_dashboard_benchmark_suite.py",
            f'export HF_HOME="{Path(self.args.hf_home).expanduser()}"',
            f'export IT_NP_CACHE="{cache}"',
            f'export IT_NP_BASELINE_WORKTREES="{wt_root}"',
            f'export SAEDASHBOARD_REPO_ROOT="{self.repo_paths["sae_dashboard"]}"',
            f'export SAELENS_REPO_ROOT="{self.repo_paths["sae_lens"]}"',
            f'export NEURONPEDIA_REPO_ROOT="{np_repo}"',
            f'export NEURONPEDIA_UTILS_ROOT="{np_repo / "utils" / "neuronpedia-utils"}"',
            f'export IT_BENCH_PYTHON="{venv_path / "bin" / "python"}"',
        ]
        env_file = wt_root / "benchmark_env.sh"
        self.say(f"- writing {env_file}")
        if not self.args.dry_run:
            env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.say(
            "\nSetup complete. To run the three-way benchmark (~25 min on the reference GPU):\n\n"
            f"  source {env_file}\n"
            f"  source {venv_path}/bin/activate\n"
            f"  cd {self.repo_paths['interpretune']}\n"
            "  python scripts/run_dashboard_benchmark_suite.py --mode threeway \\\n"
            f'    --local-db-url "{self.args.local_db_url}"\n\n'
            "Use --mode full for the batch-shape + n-prompts scaling sweeps (~3-5 h). "
            "See scripts/dashboard_benchmark_suite_usage.md for all modes and flags."
        )
        if self.warnings:
            self.say("\nWarnings raised during setup (review before running):")
            for w in self.warnings:
                self.say(f"  - {w}")

    def main(self) -> int:
        if self.args.dry_run:
            self.say("DRY RUN: printing the plan; no command below is executed.")
        self.resolve_repos()
        self.create_worktrees()
        self.check_db()
        venv_path = self.build_env()
        self.check_datasets()
        self.write_env_file(venv_path)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repos-root", default=str(Path.home() / "repos"), help="Parent dir for repo checkouts/clones."
    )
    for spec in REPOS:
        parser.add_argument(
            f"--{spec.key.replace('_', '-')}",
            default=None,
            help=f"Path to the {spec.dirname} checkout (default: <repos-root>/{spec.dirname}).",
        )
    parser.add_argument(
        "--worktrees-dir",
        required=True,
        help="Directory to create the detached preserved-baseline worktrees in (created if missing; "
        "existing worktrees inside are verified, never recreated or deleted).",
    )
    parser.add_argument(
        "--np-cache",
        default=os.getenv("IT_NP_CACHE")
        or str(
            Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "interpretune" / "neuronpedia"
        ),
        help="Neuronpedia cache root holding the benchmark prompt datasets ($IT_NP_CACHE).",
    )
    parser.add_argument(
        "--hf-home",
        default=os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        help="HF_HOME exported in benchmark_env.sh (model/dataset caches).",
    )
    parser.add_argument("--venv-dir", default=os.getenv("IT_VENV_BASE", str(Path.home() / ".venvs")))
    parser.add_argument("--venv-name", default="it_bench")
    parser.add_argument("--torch-backend", default="cu128", help="build_it_env.sh torch backend (cu128, cpu, auto).")
    parser.add_argument("--local-db-url", default=DEFAULT_DB_URL)
    parser.add_argument("--skip-env-build", action="store_true", help="Skip the venv build step entirely.")
    parser.add_argument(
        "--clear-existing-venv",
        action="store_true",
        help="Rebuild (clear) an existing venv without prompting.",
    )
    parser.add_argument("--skip-services-check", action="store_true", help="Skip the Postgres reachability check.")
    parser.add_argument("--yes", action="store_true", help="Accept defaults for every prompt (non-interactive).")
    parser.add_argument("--dry-run", action="store_true", help="Print the full plan without executing anything.")
    return parser


if __name__ == "__main__":
    sys.exit(Setup(build_parser().parse_args()).main())
