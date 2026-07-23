# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Initially based on: https://bit.ly/34f7gDv
"""Diagnose your system and show basic information.

This server mainly to get detail info for better bug reporting.
"""

import os
import platform

import argparse
import importlib
import importlib.metadata as md
import json
import re
import site
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch

LEVEL_OFFSET = "\t"
KEY_PADDING = 20


def _distribution_name_candidates(pkg_name: str) -> tuple[str, ...]:
    candidates = [pkg_name]
    dash_name = pkg_name.replace("_", "-")
    underscore_name = pkg_name.replace("-", "_")
    for candidate in (dash_name, underscore_name):
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _get_distribution(pkg_name: str) -> md.Distribution | None:
    for candidate in _distribution_name_candidates(pkg_name):
        try:
            return md.distribution(candidate)
        except md.PackageNotFoundError:
            continue
    return None


def _short_sha(sha: str | None) -> str | None:
    if not sha:
        return None
    return sha[:7]


def _looks_like_git_sha(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-f]{7,40}", value))


def _repo_slug_from_url(url: str | None) -> str | None:
    if not url:
        return None
    if ":" in url and "//" not in url and "@" in url:
        path = url.split(":", 1)[1]
    else:
        path = urlparse(url).path
    path = path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path or None


def _distribution_direct_url(pkg_name: str) -> dict | None:
    dist = _get_distribution(pkg_name)
    if dist is None:
        return None
    du_text = dist.read_text("direct_url.json")
    if not du_text:
        return None
    try:
        return json.loads(du_text)
    except json.JSONDecodeError:
        return None


def _editable_src_dir_from_direct_url(direct_url: dict | None) -> Path | None:
    if not direct_url:
        return None
    if not direct_url.get("dir_info", {}).get("editable", False):
        return None
    url = direct_url.get("url", "")
    if not url.startswith("file://"):
        return None
    parsed = urlparse(url)
    path = Path(unquote(parsed.path))
    return path if path.is_dir() else None


def _editable_src_dir_from_pth(pkg_name: str) -> Path | None:
    prefixes = tuple(f"__editable__.{candidate}" for candidate in _distribution_name_candidates(pkg_name))
    for sp in site.getsitepackages():
        site_packages_dir = Path(sp)
        if not site_packages_dir.is_dir():
            continue
        for entry in site_packages_dir.iterdir():
            if not any(entry.name.startswith(prefix) for prefix in prefixes) or entry.suffix != ".pth":
                continue
            src = entry.read_text().strip()
            src_path = Path(src)
            if not src_path.is_dir():
                continue
            for candidate in (src_path, src_path.parent):
                if (candidate / ".git").exists():
                    return candidate
    return None


def _editable_src_dir(pkg_name: str) -> Path | None:
    direct_url = _distribution_direct_url(pkg_name)
    return _editable_src_dir_from_direct_url(direct_url) or _editable_src_dir_from_pth(pkg_name)


def _run_git(repo_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _git_checkout_metadata(repo_dir: Path) -> dict[str, str] | None:
    sha = _short_sha(_run_git(repo_dir, "rev-parse", "HEAD"))
    branch = _run_git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    origin_url = _run_git(repo_dir, "remote", "get-url", "origin")

    metadata: dict[str, str] = {}
    fork = _repo_slug_from_url(origin_url)
    if fork:
        metadata["fork"] = fork
    if branch:
        metadata["branch"] = branch
    if sha:
        metadata["sha"] = sha
    return metadata or None


def _direct_url_git_metadata(direct_url: dict | None) -> dict[str, str] | None:
    if not direct_url:
        return None

    editable_src_dir = _editable_src_dir_from_direct_url(direct_url)
    if editable_src_dir is not None:
        return _git_checkout_metadata(editable_src_dir)

    vcs_info = direct_url.get("vcs_info") or {}
    if vcs_info.get("vcs") != "git":
        return None

    metadata: dict[str, str] = {}
    fork = _repo_slug_from_url(direct_url.get("url"))
    if fork:
        metadata["fork"] = fork

    requested_revision = vcs_info.get("requested_revision")
    if requested_revision and not _looks_like_git_sha(requested_revision):
        metadata["branch"] = requested_revision

    sha = _short_sha(vcs_info.get("commit_id") or requested_revision)
    if sha:
        metadata["sha"] = sha

    return metadata or None


def _package_git_metadata(pkg_name: str) -> dict[str, str] | None:
    direct_url_metadata = _direct_url_git_metadata(_distribution_direct_url(pkg_name))
    if direct_url_metadata is not None:
        return direct_url_metadata

    editable_src_dir = _editable_src_dir(pkg_name)
    if editable_src_dir is not None:
        return _git_checkout_metadata(editable_src_dir)
    return None


def _format_git_metadata(metadata: dict[str, str] | None) -> str | None:
    if not metadata:
        return None
    ordered_keys = ("fork", "branch", "sha")
    details = [f"{key}:{metadata[key]}" for key in ordered_keys if metadata.get(key)]
    return f"({', '.join(details)})" if details else None


def _pkg_version(pkg_name: str, module_name: str | None = None) -> str:
    """Return a best-effort version for a package.

    Tries to import the module and read __version__, then falls back to importlib.metadata.version(pkg_name).
    For editable installs or git-backed wheel installs, appends git provenance such as
    ``(fork:speediedan/circuit-tracer, sha:14cc3e8)``.
    Returns a readable failure string on error.
    """
    version = None
    try:
        module = importlib.import_module(module_name or pkg_name)
        version = getattr(module, "__version__", None)
    except Exception:
        pass
    if not version:
        for candidate in _distribution_name_candidates(pkg_name):
            try:
                version = md.version(candidate)
                break
            except md.PackageNotFoundError:
                continue
        if not version:
            return f"not found ({pkg_name})"
    git_metadata = _format_git_metadata(_package_git_metadata(pkg_name))
    return f"{version} {git_metadata}" if git_metadata else str(version)


def info_system():
    return {
        "OS": platform.system(),
        "architecture": platform.architecture(),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def info_cuda():
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "available": torch.cuda.is_available(),
        "version": getattr(getattr(torch, "version", object()), "cuda", None),
    }


def info_packages():
    packages = {
        "interpretune": _pkg_version("interpretune", "interpretune"),
        "lightning": _pkg_version("lightning", "lightning"),
        "transformer_lens": _pkg_version("transformer_lens", "transformer_lens"),
        "sae_lens": _pkg_version("sae_lens", "sae_lens"),
        "SAEDashboard": _pkg_version("SAEDashboard", "sae_dashboard"),
        "circuit_tracer": _pkg_version("circuit_tracer", "circuit_tracer"),
        "nnsight": _pkg_version("nnsight", "nnsight"),
        "neuronpedia": _pkg_version("neuronpedia", "neuronpedia"),
        "neuronpedia-utils": _pkg_version("neuronpedia-utils", "neuronpedia_utils"),
        "finetuning_scheduler": _pkg_version("finetuning_scheduler", "finetuning_scheduler"),
        "transformers": _pkg_version("transformers", "transformers"),
        "datasets": _pkg_version("datasets", "datasets"),
        "numpy": _pkg_version("numpy", "numpy"),
        "tqdm": _pkg_version("tqdm", "tqdm"),
    }

    # Add PyTorch-specific runtime info as separate keys to preserve previous fields
    packages.update(
        {
            "torch": getattr(torch, "__version__", packages.get("torch", "not found")),
            "torch_git_version": getattr(getattr(torch, "version", object()), "git_version", "not found"),
            "torch_debug_mode": getattr(getattr(torch, "version", object()), "debug", "not found"),
        }
    )

    return packages


def nice_print(details, level=0):
    lines = []
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose system and print environment/package details",
    )
    parser.add_argument(
        "--packages-only",
        action="store_true",
        help="Output only the package versions (JSON-like printed dict)",
    )
    args = parser.parse_args()

    packages = info_packages()
    if args.packages_only:
        # Print packages section only in a compact, human-readable form
        lines = nice_print({"Key Package Versions": packages})
        print(os.linesep.join(lines))
        return

    details = {"System": info_system(), "CUDA": info_cuda(), "Packages": packages}
    lines = nice_print(details)
    text = os.linesep.join(lines)
    print(text)


if __name__ == "__main__":
    main()
