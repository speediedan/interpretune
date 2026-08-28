#!/usr/bin/env python3
"""Set up the full local environment for the Neuronpedia dashboard benchmark suite.

One guided, transparent, non-destructive flow that prepares everything
`scripts/run_dashboard_benchmark_suite.py` needs (Linux and macOS):

1. Check prerequisites: required tools and HuggingFace access (see PREREQUISITES below).
2. Locate (or clone) the four source repos: interpretune, SAEDashboard, SAELens, neuronpedia.
   TransformerLens, nnsight, and circuit-tracer are NOT needed as checkouts — interpretune's
   dependency pins install them at the validated versions (TransformerLens 3.5.1 and nnsight
   0.7.0 as release pins, circuit-tracer at the pinned fork SHA). Existing checkouts
   are never switched or modified; dirty trees are surfaced with an offer to stash.
3. Create the detached preserved-baseline worktrees (the pre-PR comparison lineage
   `SD-7886eaa+benchmark_patches / SL-3eea6552 / NP-5a33f17`) in a directory you choose,
   applying the audited benchmark patches from `scripts/benchmark_baseline_patches/`
   (see that directory's README for the per-patch classification/rationale) and
   verifying the resulting tree state against pinned expectations.
4. Ensure the neuronpedia local-stack env defaults (`.env`: Postgres host port/data dir, HF
   cache paths — appended only when missing) and check the local Postgres is reachable
   (reporting how to start it when it is not).
5. Build the integrated interpretune benchmark venv via `scripts/build_it_env.sh`
   (SAEDashboard editable from source; everything else, SAELens included, from interpretune's
   pins). SAELens is deliberately NOT built from source: its wave work merged upstream and
   released as 6.49.0, so the pins resolve the same code the artifacts were produced with,
   while the fork checkout would substitute a retired commit. See Step 2.
   An existing venv is only cleared after explicit confirmation (or `--clear-existing-venv`).
6. Verify the benchmark prompt datasets exist under `$IT_NP_CACHE`, offering to build any
   missing ones (tokenizer-only, CPU, a few minutes each; requires the gated-model access
   below because they tokenize with `google/gemma-3-1b-it`).
7. Write `<worktrees-dir>/benchmark_env.sh` capturing every environment variable the
   suite needs, and print the detected GPU + the exact command to run the benchmark.

PREREQUISITES (checked in Step 1):
- `git` and `uv` on PATH; a reachable Postgres (this script verifies, it does not start one);
  bash >= 4.3 for the env build (macOS: `brew install bash`).
- HuggingFace access to the GATED model `google/gemma-3-1b-it`: accept the license on the
  model page, then authenticate via `hf auth login` (stored token) or `export HF_TOKEN=...`
  (the pipeline also honors `HF_GATED_PUBLIC_REPO_AUTH_KEY` as a fallback). Only the
  tokenizer/weights are fetched; the benchmark prompt datasets' sources are not gated.
- Root is never required. Nothing is pushed and no existing checkout is modified.

Every mutating action is printed before it runs; use `--dry-run` to see the full plan
without executing anything, and `--yes` to accept defaults non-interactively (any
explicitly passed flag always wins over a prompt).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

SCRIPT_DIR = Path(__file__).resolve().parent
IT_ROOT = SCRIPT_DIR.parent
PATCHES_DIR = SCRIPT_DIR / "benchmark_baseline_patches"

# SAEDashboard and neuronpedia still carry the wave work on a topic branch, because their PRs are
# open upstream and their head refs are that branch. interpretune is different: its side of the wave
# lands on `main` (the dependency refresh as #234, the docs/examples as #232), so cloning it at the
# topic branch would hand reviewers a checkout that is behind `main` -- and would break outright once
# the topic branch is deleted. Keep the two axes separate rather than sharing one constant.
#
# SAELens is cloned at the wave branch too, but ONLY as the source of the preserved-baseline worktree
# (`SL_BASELINE_SHA`, materialised in `create_worktrees`). It is deliberately NOT installed from that
# checkout: SAELens#721 merged and released as 6.49.0, and sae-lens was retired from interpretune's
# git-deps on 2026-08-09, so the CI pins resolve the released code the artifacts were produced with.
# The wave branch head additionally carries the `pretrained_saes.yaml` entries retired as corrupt, so
# installing it would put a retired commit into the very environment used to verify a corpus.
WAVE_BRANCH = "streamlined-streamable-dashboard-generation-phase-1"
INTERPRETUNE_BRANCH = "main"
DEFAULT_DB_URL = "postgres://postgres:postgres@127.0.0.1:5433/postgres"
GATED_MODEL = "google/gemma-3-1b-it"
REFERENCE_GPU = "NVIDIA GeForce RTX 4090 (24 GiB)"

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
RTE_PRETOK = "pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts"
RTE_LEGACY = "legacy_pretokenized/gemma-3-1b-it_rte_boolq_context319_fixed_pad_2490"
MONOLOGY_PRETOK = "pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_2490"
MONOLOGY_LEGACY = "legacy_pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_2490"
REQUIRED_DATASETS = (RTE_PRETOK, RTE_LEGACY, MONOLOGY_PRETOK, MONOLOGY_LEGACY)
# Additionally required by --mode full (the prompt-dimension sweep).
MONOLOGY_SWEEP = "pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_24576"
FULL_MODE_DATASETS = (MONOLOGY_SWEEP,)

_PRETOK_MODULE = "sae_dashboard.neuronpedia.prompt_pretokenization"
_MONOLOGY_COMMON = (
    "--dataset-path",
    "monology/pile-uncopyrighted",
    "--dataset-split",
    "train",
    "--tokenizer-name",
    GATED_MODEL,
    "--context-size",
    "128",
    "--column-name",
    "text",
    "--use-chat-formatting",
    "--streaming",
    "--num-proc",
    "1",
    "--no-shuffle",
)


def dataset_build_groups(cache: Path) -> tuple[dict, ...]:
    """The pretokenization invocations of record (docs/neuronpedia_dashboard_pipeline.md § Regenerating the
    benchmark prompt datasets); each group is one CLI run."""

    return (
        {
            "name": "RTE example-aligned cache + legacy fixed-pad export",
            "produces": (RTE_PRETOK, RTE_LEGACY),
            "args": [
                "-m",
                _PRETOK_MODULE,
                "--dataset-path",
                "aps/super_glue",
                "--dataset-name",
                "rte",
                "--dataset-split",
                "train",
                "--tokenizer-name",
                GATED_MODEL,
                "--context-size",
                "128",
                "--custom-dataset-module",
                "it_examples.utils.dashboard_pretokenization_rte",
                "--windowing-mode",
                "max-prompt-pad",
                "--no-shuffle",
                "--output-dir",
                str(cache / RTE_PRETOK),
                "--legacy-output-dir",
                str(cache / RTE_LEGACY),
                "--force",
            ],
        },
        {
            "name": "Monology packed cache + legacy export (2490 rows)",
            "produces": (MONOLOGY_PRETOK, MONOLOGY_LEGACY),
            "args": [
                "-m",
                _PRETOK_MODULE,
                *_MONOLOGY_COMMON,
                "--max-tokenized-rows",
                "2490",
                "--output-dir",
                str(cache / MONOLOGY_PRETOK),
                "--legacy-output-dir",
                str(cache / MONOLOGY_LEGACY),
                "--force",
            ],
        },
        {
            "name": "Monology concat_24576 sweep set (needed by --mode full only)",
            "produces": (MONOLOGY_SWEEP,),
            "args": [
                "-m",
                _PRETOK_MODULE,
                *_MONOLOGY_COMMON,
                "--max-tokenized-rows",
                "24576",
                "--output-dir",
                str(cache / MONOLOGY_SWEEP),
                "--force",
            ],
        },
    )


@dataclass
class RepoSpec:
    key: str
    dirname: str
    url: str
    ref: str | None  # branch/tag checked out for FRESH clones only; existing checkouts untouched


REPOS: tuple[RepoSpec, ...] = (
    RepoSpec("interpretune", "interpretune", "https://github.com/speediedan/interpretune.git", INTERPRETUNE_BRANCH),
    RepoSpec("sae_dashboard", "SAEDashboard", "https://github.com/speediedan/SAEDashboard.git", WAVE_BRANCH),
    RepoSpec("sae_lens", "SAELens", "https://github.com/speediedan/SAELens.git", WAVE_BRANCH),
    RepoSpec("neuronpedia", "neuronpedia", "https://github.com/speediedan/neuronpedia.git", WAVE_BRANCH),
)
MANIFEST_REPO_KEYS = ("interpretune", "sae_dashboard", "sae_lens", "neuronpedia")


class Setup:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.actions_taken: list[str] = []
        self.warnings: list[str] = []
        self.repo_paths: dict[str, Path] = {}
        log_dir = Path(args.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"dashboard_bench_env_setup_{stamp}.log"
        self._log_fh = self.log_path.open("a", encoding="utf-8")

    # ---------- io helpers ----------

    def _log(self, text: str) -> None:
        self._log_fh.write(text if text.endswith("\n") else text + "\n")
        self._log_fh.flush()

    def say(self, msg: str) -> None:
        print(msg, flush=True)
        self._log(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"WARNING: {msg}", flush=True)
        self._log(f"WARNING: {msg}")

    def fail(self, msg: str) -> None:
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        self._log(f"ERROR: {msg}")
        raise SystemExit(1)

    def _run_streamed(self, cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
        """Run a long/verbose child, mirroring its output to the terminal AND the log file."""

        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            self._log(line)
        return proc.wait()

    def _auto_answer(self, question: str, shown: str) -> bool:
        """True when this run must not prompt, having already reported the answer it will use.

        `--dry-run` prints a plan and executes nothing, so there is nothing for an operator to decide; prompting there
        strands the run on a question whose answer cannot matter. A non-interactive stdin (CI, a pipe, `< /dev/null`) is
        handled at the `input()` call rather than here, because it is a property of the invocation and not of the
        arguments.
        """
        if self.args.yes:
            self.say(f"{question} [auto-{shown} via --yes]")
            return True
        if self.args.dry_run:
            self.say(f"{question} [auto-{shown} via --dry-run; nothing is executed]")
            return True
        return False

    def confirm(self, question: str, default: bool = True) -> bool:
        if self._auto_answer(question, "yes" if default else "no"):
            return default
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            try:
                resp = input(f"{question} {suffix} ").strip().lower()
            except EOFError:
                self.warn(f"no input available; taking the default ({'yes' if default else 'no'}) for: {question}")
                return default
            if not resp:
                self._log(f"{question} -> (default {'yes' if default else 'no'})")
                return default
            if resp in ("y", "yes"):
                self._log(f"{question} -> yes")
                return True
            if resp in ("n", "no"):
                self._log(f"{question} -> no")
                return False

    def choose(self, question: str, choices: dict[str, str], default: str) -> str:
        """choices: {key: description}; returns the chosen key."""
        if self._auto_answer(question, f"'{default}'"):
            return default
        menu = ", ".join(f"[{k}] {v}" for k, v in choices.items())
        while True:
            try:
                resp = input(f"{question} ({menu}; default {default}): ").strip().lower()
            except EOFError:
                self.warn(f"no input available; taking the default ('{default}') for: {question}")
                return default
            if not resp:
                self._log(f"{question} -> (default '{default}')")
                return default
            if resp in choices:
                self._log(f"{question} -> '{resp}'")
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

    def _hf_token_present(self) -> str | None:
        """Return a short description of how HF auth was found, or None."""
        if os.environ.get("HF_TOKEN"):
            return "HF_TOKEN environment variable"
        if os.environ.get("HF_GATED_PUBLIC_REPO_AUTH_KEY"):
            return "HF_GATED_PUBLIC_REPO_AUTH_KEY environment variable"
        token_file = Path(self.args.hf_home).expanduser() / "token"
        if token_file.is_file():
            return f"stored token at {token_file}"
        default_token = Path.home() / ".cache" / "huggingface" / "token"
        if default_token.is_file():
            return f"stored token at {default_token}"
        return None

    # ---------- steps ----------

    def check_prereqs(self) -> None:
        self.say("\n=== Step 1/7: prerequisites (tools + HuggingFace access) ===")
        self.say(
            "Required: git + uv on PATH; a Postgres this script can reach; bash >= 4.3 for the env "
            "build (macOS: `brew install bash`). Root is never required; nothing is pushed and no "
            "existing checkout is modified."
        )
        for tool, why, fatal in (
            ("git", "clones/worktrees", True),
            ("uv", "the env build and package installs", not self.args.skip_env_build),
        ):
            path = shutil.which(tool)
            if path:
                self.say(f"- [OK] {tool} ({path})")
            elif fatal:
                self.fail(f"`{tool}` not found on PATH — required for {why}.")
            else:
                self.warn(f"`{tool}` not found on PATH — needed for {why}.")
        auth = self._hf_token_present()
        if auth:
            self.say(f"- [OK] HuggingFace auth detected ({auth})")
        else:
            self.warn(
                f"no HuggingFace auth detected. `{GATED_MODEL}` is a GATED model: accept its license on "
                "the model page, then `hf auth login` or `export HF_TOKEN=...` (the benchmark pipeline "
                "also honors HF_GATED_PUBLIC_REPO_AUTH_KEY). Model download AND dataset pretokenization "
                "will fail without it."
            )
            if not self.confirm("  continue without HuggingFace auth?", default=False):
                self.fail("aborted at user request (missing HuggingFace auth)")

    def resolve_repos(self) -> None:
        self.say("\n=== Step 2/7: source repositories ===")
        self.say(
            "Only interpretune, SAEDashboard, SAELens, and neuronpedia are needed as checkouts — "
            "TransformerLens (v3.5.1 pinned SHA), nnsight (0.7.0), and circuit-tracer (pinned fork "
            "SHA) install from interpretune's dependency pins during the env build."
        )
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
                if spec.ref and branch != spec.ref:
                    self.warn(
                        f"{spec.dirname} is on '{branch}', not the expected wave branch '{spec.ref}'. "
                        "This script never switches branches for you; switch manually if intended."
                    )
                dirty = self.run(["git", "-C", str(path), "status", "--porcelain"], mutating=False)
                if dirty:
                    entries = dirty.splitlines()

                    # `??` is porcelain's untracked marker. Split it out because the stash offer below
                    # passes `-u`, which sweeps untracked files too: a maintainer's hand-written local
                    # config is exactly the kind of file that is untracked, valuable and unbacked-up.
                    # `run()` strips, so the first porcelain line has lost the leading space of its
                    # two-character status field; splitting on whitespace is correct for both shapes.
                    def _porcelain_path(entry: str) -> str:
                        parts = entry.split(None, 1)
                        return parts[1] if len(parts) > 1 else entry

                    untracked = [e for e in entries if e.startswith("??")]
                    tracked = [e for e in entries if not e.startswith("??")]
                    shown = (tracked + untracked)[:12]
                    elided = len(entries) - len(shown)
                    summary = f"  {spec.dirname} has uncommitted changes"
                    if untracked:
                        summary += f" ({len(tracked)} tracked, {len(untracked)} untracked)"
                    self.say(summary + ":\n    " + "\n    ".join(shown))
                    if elided > 0:
                        self.say(f"    ... and {elided} more (`git -C {path} status` for the full list)")
                    if untracked:
                        self.warn(
                            f"stashing {spec.dirname} would also remove its {len(untracked)} untracked "
                            "file(s) from the tree (`git stash push -u`). Recover with `git -C "
                            f"{path} stash pop`."
                        )
                    # A dirty `.env*` is usually the local-stack wiring (DB host/port, cache paths),
                    # i.e. the very settings the run about to be configured depends on. Stashing it
                    # reverts them, and the resulting failure surfaces as an unreachable database
                    # rather than as a stash, which is a long way from the cause.
                    envish = sorted(q for q in (_porcelain_path(e) for e in entries) if Path(q).name.startswith(".env"))
                    if envish:
                        self.warn(
                            f"{spec.dirname}'s pending changes include local-stack config "
                            f"({', '.join(envish)}). Stashing reverts the settings this run is about to "
                            "depend on; prefer [c] unless you know they are unrelated."
                        )
                    action = self.choose(
                        f"  {spec.dirname} is dirty: stash, continue as-is, or abort?",
                        {
                            "s": "git stash push -u (INCLUDING untracked files)" if untracked else "git stash push -u",
                            "c": "continue with dirty tree",
                            "a": "abort",
                        },
                        "c",
                    )
                    if action == "a":
                        self.fail("aborted at user request (dirty repo)")
                    if action == "s":
                        self.run(
                            ["git", "-C", str(path), "stash", "push", "-u", "-m", "setup_dashboard_benchmark_env"],
                            cwd=None,
                        )
                        self.say(f"    stashed; restore with `git -C {path} stash pop`")
                    elif spec.key in MANIFEST_REPO_KEYS:
                        self.warn(
                            f"{spec.dirname} left dirty — reviewer packaging refuses dirty manifest repos "
                            "(run_dashboard_benchmark_suite.py needs clean SD/SL/NP/IT trees unless --allow-dirty)."
                        )
            else:
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
        self.say("\n=== Step 3/7: detached preserved-baseline worktrees ===")
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

    def _ensure_np_env_defaults(self) -> None:
        """Append missing local-stack keys to the neuronpedia untracked `.env`.

        `HF_HOME` / `HF_HUB_CACHE` / `HF_DATASETS_CACHE` are the load-bearing ones: neuronpedia's
        `make inference-dev` sources this `.env` into the process environment, so a relocated HF
        cache is honoured with no mount and no flag.

        `POSTGRES_HOST_PORT` and `POSTGRES_DATA_DIR` are **vestigial** and kept only because
        removing keys from an operator's `.env` is not this script's business. They were read by
        `docker/compose.yaml`, which upstream deleted; nothing consumes them now. The port that
        matters today is the one inside `--local-db-url`, and the data directory belongs to
        whoever runs the server. Existing values are never modified here.
        """

        np_repo = self.repo_paths["neuronpedia"]
        env_file = np_repo / ".env"
        existing = env_file.read_text(encoding="utf-8") if env_file.is_file() else ""
        existing_keys = {line.split("=", 1)[0].strip() for line in existing.splitlines() if "=" in line}
        parsed = urlsplit(self.args.local_db_url)
        hf_home = Path(self.args.hf_home).expanduser()
        wanted = {
            "POSTGRES_HOST_PORT": str(parsed.port or 5432),
            "POSTGRES_DATA_DIR": str(Path(self.args.postgres_data_dir).expanduser()),
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        }
        missing = {k: v for k, v in wanted.items() if k not in existing_keys}
        if not missing:
            self.say(f"- {env_file}: all local-stack keys already present (values left untouched)")
            return
        self.say(f"- appending missing local-stack keys to {env_file} (existing values are never modified):")
        for k, v in missing.items():
            self.say(f"    {k}={v}")
        if not self.confirm("  append these keys?"):
            self.warn(
                f"{env_file} not updated — a relocated HF cache will not be picked up by "
                "neuronpedia's `make inference-dev`, which sources this file."
            )
            return
        if not self.args.dry_run:
            prefix = "" if (not existing or existing.endswith("\n")) else "\n"
            with env_file.open("a", encoding="utf-8") as fh:
                fh.write(prefix + "".join(f"{k}={v}\n" for k, v in missing.items()))
        self.actions_taken.append(f"appended {sorted(missing)} to {env_file}")

    def check_db(self) -> None:
        self.say("\n=== Step 4/7: neuronpedia local-stack env defaults + Postgres ===")
        if self.args.skip_services_check:
            self.say("- skipped (--skip-services-check)")
            return
        self._ensure_np_env_defaults()
        parsed = urlsplit(self.args.local_db_url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 5432
        try:
            with socket.create_connection((host, port), timeout=3):
                self.say(f"- Postgres reachable at {host}:{port} — the DB-import benchmark legs can run.")
                return
        except OSError:
            pass
        self.warn(f"Postgres NOT reachable at {host}:{port}.")
        # No bring-up is offered any more, and that is not a regression: upstream neuronpedia
        # removed containers from the repo entirely (no Dockerfiles, no compose files, no
        # `init.sh`). Nothing in its Makefile starts a server either, so there is nothing for
        # this script to launch, and pretending otherwise is what the previous compose branch
        # did: it failed soft behind `compose.is_file()`, silently stopping at some point after
        # upstream deleted the file, with no signal that a capability had gone.
        #
        # The `make db-check` hint is emitted only if that target is actually present in the
        # checkout, because whether it exists depends on which neuronpedia ref is cloned rather
        # than on anything this script controls: upstream's post-refactor tree has `db-check`,
        # `db-init`, `db-status` and `db-reset`, while the fork's pre-refactor branches have none
        # of them. A hint hard-coded either way is wrong half the time and wrong silently, since
        # a missing Make target only surfaces when the reader tries it. Ask the tree instead.
        np_repo = self.repo_paths.get("neuronpedia")
        hint = ""
        if np_repo and (Path(np_repo) / "Makefile").is_file():
            try:
                if re.search(r"^db-check:", (Path(np_repo) / "Makefile").read_text(), re.M):
                    hint = f" (`cd {np_repo} && make db-check` diagnoses it)"
            except OSError:
                pass
        self.warn(
            f"start Postgres yourself, then re-run{hint}, or pass --local-db-url to point at a "
            "server that is already up. The DB-import benchmark legs will fail until it is reachable."
        )

    def build_env(self) -> Path:
        self.say("\n=== Step 5/7: interpretune benchmark venv (build_it_env.sh) ===")
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
        ex = "requirements/ci/excludes.txt"
        # Only SAEDashboard builds from source, because its fix is unreleased and a maintainer needs the
        # checkout. Everything else comes from the locked pins: sae-lens (released as 6.49.0, retired
        # from git-deps 2026-08-09), TransformerLens/nnsight (override-dependencies + overrides.txt) and
        # circuit-tracer (the git-deps group).
        #
        # Do NOT re-add `--from-source=sae_lens:...`. Beyond substituting a retired commit for the
        # released pin, that path hands uv `requirements/ci/sl_uv_requirements.txt`, the vendored
        # SAELens Poetry export, which is how pytest-timeout/docstr-coverage/ruff reach environments
        # while being declared in neither pyproject.toml nor the CI lock. The README-flow lanes exist
        # to detect exactly that superset, so importing it here would make this env less like a user's
        # than the pins already make it.
        cmd = [
            str(it / "scripts" / "build_it_env.sh"),
            f"--repo-home={it}",
            f"--target-env-name={self.args.venv_name}",
            f"--venv-dir={venv_dir}",
            f"--torch-backend={self.args.torch_backend}",
            (
                f"--from-source=sae_dashboard:{self.repo_paths['sae_dashboard']}:dev"
                f":UV_EXCLUDE={it / ex}:UV_OVERRIDE={it / ov}"
            ),
        ]
        self.say(
            "- integrated env build (SAEDashboard from source, sae-lens from the pins; typically a few "
            "minutes with a warm uv cache; first-time torch/CUDA wheel downloads can add ~5-10 min):"
        )
        if not self.confirm("  run the env build now?"):
            self.warn("env build skipped — activate/build a suitable venv before running the suite.")
            return venv_path
        self.say(f"  $ {' '.join(cmd)}")
        if not self.args.dry_run:
            self.actions_taken.append(" ".join(cmd))
            rc = self._run_streamed(cmd, cwd=it)
            if rc != 0:
                self.fail(f"build_it_env.sh failed with exit code {rc}")
        # neuronpedia-utils (pinned git no-deps install) and pgpq (`test` dependency-group) are handled
        # by build_it_env.sh / the locked requirements — verify they landed rather than reinstalling.
        if not self.args.dry_run:
            probe = subprocess.run(
                [str(venv_path / "bin" / "python"), "-c", "import neuronpedia_utils, pgpq"], capture_output=True
            )
            if probe.returncode != 0:
                self.fail(
                    "neuronpedia_utils/pgpq missing from the built venv — expected via "
                    "requirements/ci/nodeps_git_requirements.txt + the `test` group in the CI lock."
                )
            self.say("- [OK] neuronpedia-utils + pgpq present (columnar local-DB import lane)")
        self._check_gated_model_access(venv_path)
        return venv_path

    def _check_gated_model_access(self, venv_path: Path) -> None:
        """Authoritative gated-repo access check via the built venv (non-fatal; needs network)."""

        if self.args.dry_run:
            return
        venv_python = venv_path / "bin" / "python"
        if not venv_python.exists():
            return
        probe = f"from huggingface_hub import model_info\nmodel_info('{GATED_MODEL}')\nprint('access OK')\n"
        env = {**os.environ, "HF_HOME": str(Path(self.args.hf_home).expanduser())}
        result = subprocess.run([str(venv_python), "-c", probe], capture_output=True, text=True, env=env)
        if result.returncode == 0:
            self.say(f"- [OK] gated-model access verified ({GATED_MODEL})")
        else:
            tail = (result.stderr or "").strip().splitlines()[-1:] or ["(no error output)"]
            self.warn(
                f"could not verify access to gated model {GATED_MODEL} ({tail[0]}). If you have not "
                "accepted its license / authenticated, model download and dataset builds will fail — "
                "see the PREREQUISITES section of --help."
            )

    def ensure_datasets(self, venv_path: Path) -> None:
        self.say("\n=== Step 6/7: benchmark prompt datasets ===")
        cache = Path(self.args.np_cache).expanduser()
        for d in REQUIRED_DATASETS + FULL_MODE_DATASETS:
            status = "OK" if (cache / d).is_dir() else "MISSING"
            self.say(f"- [{status}] {cache / d}")
        groups = [g for g in dataset_build_groups(cache) if any(not (cache / p).is_dir() for p in g["produces"])]
        if not groups:
            return
        venv_python = venv_path / "bin" / "python"
        self.say(
            "\nDatasets required to run the dashboard generation benchmarking were not found — building "
            "them now is recommended: a ONE-TIME task enabling the neuronpedia/sae_dashboard generation "
            "benchmarks (the commands of record from docs/neuronpedia_dashboard_pipeline.md § "
            "'Regenerating the benchmark prompt datasets'; tokenizer-only, CPU, a few minutes per set; "
            f"requires access to the gated `{GATED_MODEL}` tokenizer). All missing sets — including the "
            "--mode full sweep set — are built together:"
        )
        for group in groups:
            self.say(f"  - {group['name']}")
        if not venv_python.exists() and not self.args.dry_run:
            self.warn("benchmark venv missing — build it first (Step 5), then re-run to build the datasets.")
            return
        if not self.confirm("  build all missing datasets now?"):
            self.warn(
                "benchmark prompt datasets not built — the presets that consume them will fail "
                "(build later per docs/neuronpedia_dashboard_pipeline.md or by re-running this script)."
            )
            return
        env = {
            **os.environ,
            "HF_HOME": str(Path(self.args.hf_home).expanduser()),
            "IT_NP_CACHE": str(cache),
            # Silence known-benign warnings from the pretokenization runs:
            # - runpy RuntimeWarning: sae_dashboard.neuronpedia's __init__ imports
            #   prompt_pretokenization, so `-m` finds it pre-imported (harmless for this CLI)
            # - SAELens "use_chat_formatting ... strings" UserWarning: wrapping raw text rows as
            #   single user messages is exactly the intended behavior for these presets
            # - datasets' torch copy-construct UserWarning: third-party formatter internals, benign
            "PYTHONWARNINGS": (
                "ignore::RuntimeWarning:runpy,"
                "ignore:use_chat_formatting:UserWarning,"
                "ignore:To copy construct from a tensor:UserWarning"
            ),
        }
        for group in groups:
            cmd = [str(venv_python), *group["args"]]
            self.say(f"  $ {' '.join(cmd)}")
            if self.args.dry_run:
                self.actions_taken.append(f"[dry-run] dataset build: {group['name']}")
                continue
            self.actions_taken.append(f"dataset build: {group['name']}")
            rc = self._run_streamed(cmd, cwd=self.repo_paths["interpretune"], env=env)
            if rc != 0:
                self.fail(f"dataset build failed ({rc}): {group['name']}")
            still_missing = [p for p in group["produces"] if not (cache / p).is_dir()]
            if still_missing:
                self.fail(f"dataset build reported success but outputs are missing: {still_missing}")
            self.say(f"  OK: built {', '.join(group['produces'])}")

    def _detect_gpu(self) -> str | None:
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None
        try:
            result = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            return lines[0] if lines else None
        except Exception:
            return None

    def write_env_file(self, venv_path: Path) -> None:
        self.say("\n=== Step 7/7: benchmark_env.sh + next steps ===")
        wt_root = Path(self.args.worktrees_dir).expanduser()
        cache = Path(self.args.np_cache).expanduser()
        hf_home = Path(self.args.hf_home).expanduser()
        np_repo = self.repo_paths["neuronpedia"]
        lines = [
            "# Generated by scripts/setup_dashboard_benchmark_env.py "
            f"({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})",
            "# source this before running scripts/run_dashboard_benchmark_suite.py",
            f'export HF_HOME="{hf_home}"',
            f'export HF_HUB_CACHE="{hf_home / "hub"}"',
            f'export HF_DATASETS_CACHE="{hf_home / "datasets"}"',
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
        gpu = self._detect_gpu()
        gpu_line = (
            f"Detected GPU: {gpu}."
            if gpu
            else "No NVIDIA GPU detected (nvidia-smi unavailable) — the benchmark needs one."
        )
        self.say(
            f"\n{gpu_line}\nReference benchmark hardware: {REFERENCE_GPU} — three-way mode takes ~25 min "
            "and full mode ~2 h there; scale expectations to your GPU.\n"
            "\nSetup complete. To generate the full reviewer benchmark package (3-way + batch-shape "
            "scaling sweeps + the n-prompts scaling curve; ~2 h on the reference GPU). Because the run "
            "is long, background it with nohup so it survives a disconnect and keeps a console log:\n\n"
            f"  source {env_file}\n"
            f"  source {venv_path}/bin/activate\n"
            f"  cd {self.repo_paths['interpretune']}\n"
            "  nohup python scripts/run_dashboard_benchmark_suite.py --mode full \\\n"
            "    --session-root /tmp/np_dashboard_generation_profiles/full_$(date +%Y%m%d) \\\n"
            "    --package-root /tmp/dashboard_benchmark_packages/full_$(date +%Y%m%d) \\\n"
            "    --run-tag dashboard-bench \\\n"
            f'    --local-db-url "{self.args.local_db_url}" \\\n'
            "    > /tmp/full_wave.log 2>&1 &\n\n"
            "  tail -f /tmp/full_wave.log   # follow progress; the reviewer package lands in --package-root\n\n"
            "For a quicker check without the scaling-curve generation, use --mode threeway (~25 min) — short "
            "enough to run in the foreground:\n\n"
            f"  source {env_file}\n"
            f"  source {venv_path}/bin/activate\n"
            f"  cd {self.repo_paths['interpretune']}\n"
            "  python scripts/run_dashboard_benchmark_suite.py --mode threeway \\\n"
            "    --session-root /tmp/np_dashboard_generation_profiles/threeway_$(date +%Y%m%d) \\\n"
            "    --package-root /tmp/dashboard_benchmark_packages/threeway_$(date +%Y%m%d) \\\n"
            "    --run-tag dashboard-bench \\\n"
            f'    --local-db-url "{self.args.local_db_url}"\n\n'
            "See scripts/dashboard_benchmark_suite_usage.md for all modes and flags."
        )
        if self.warnings:
            self.say("\nWarnings raised during setup (review before running):")
            for w in self.warnings:
                self.say(f"  - {w}")
        self.say(f"\nFull setup log: {self.log_path}")

    def main(self) -> int:
        self.say(f"Logging this run to {self.log_path}")
        if self.args.dry_run:
            self.say("DRY RUN: printing the plan; no command below is executed.")
        self.check_prereqs()
        self.resolve_repos()
        self.create_worktrees()
        self.check_db()
        venv_path = self.build_env()
        self.ensure_datasets(venv_path)
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
        default=str(Path(tempfile.gettempdir()) / f"it_baseline_trees_{datetime.now().strftime('%Y%m%d')}"),
        help="Directory to create the detached preserved-baseline worktrees in (created if missing; "
        "existing worktrees inside are verified, never recreated or deleted). Defaults to a dated "
        "temp path (e.g. /tmp/it_baseline_trees_20260731) so the script can be run with no arguments "
        "at all; the date suffix keeps reruns on different days from colliding.",
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
        help="HF_HOME exported in benchmark_env.sh (model/dataset caches; "
        "HF_HUB_CACHE/HF_DATASETS_CACHE derive from it).",
    )
    parser.add_argument(
        "--postgres-data-dir",
        default=None,
        help="Host directory for the local Neuronpedia Postgres data volume (written to the neuronpedia "
        "untracked .env when missing; default: <np-cache>/neuronpedia_db).",
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
    parser.add_argument(
        "--log-dir",
        default=tempfile.gettempdir(),
        help="Directory for the full setup log (dashboard_bench_env_setup_<YYYYMMDD_HHMMSS>.log); "
        "defaults to the platform temporary directory.",
    )
    parser.add_argument("--yes", action="store_true", help="Accept defaults for every prompt (non-interactive).")
    parser.add_argument("--dry-run", action="store_true", help="Print the full plan without executing anything.")
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    if args.postgres_data_dir is None:
        args.postgres_data_dir = str(Path(args.np_cache).expanduser() / "neuronpedia_db")
    return args


if __name__ == "__main__":
    sys.exit(Setup(parse_args()).main())
