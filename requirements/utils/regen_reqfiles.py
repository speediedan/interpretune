import argparse
import fnmatch
import os
import shlex
import subprocess
import shutil
import toml
import re
from dataclasses import dataclass
from typing import Dict, List

# Commit-based pin selection logic
# --------------------------------
# This module supports a small workflow for installing important helper
# packages either from a version bound declared in `pyproject.toml` (preferred)
# or from a specific commit SHA recorded in a pin file under
# `requirements/ci/<pin_filename>`. The selection rules are:
# 1. If the package-specific environment variable (e.g. `IT_USE_CT_COMMIT_PIN`)
#    is set to a truthy value ("1", "true", "yes"), the pin-file-based
#    commit installation is selected.
# 2. Otherwise, if the package appears in the pyproject location declared by
#    its mapping (for example the `examples` optional extra), the pyproject
#    requirement (including any version bounds) is used.
# 3. If neither of the above applies, the code falls back to the pin file (if
#    present) as a best-effort final option.
#
# The mapping `DEP_COMMIT_PINS` at the top of this file controls which
# packages participate in this flow; add new entries to enable the same
# behavior for other packages.

# Paths
REQ_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(REQ_DIR))
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")
CI_REQ_DIR = os.path.join(REPO_ROOT, "requirements", "ci")
POST_UPGRADES_PATH = os.path.join(CI_REQ_DIR, "post_upgrades.txt")


@dataclass
class DepCommitPin:
    package_name: str
    env_var: str
    dep_def_loc: str  # e.g. 'examples' for examples extra or 'dependencies' for a base dependency
    pin_filename: str
    repo_base_url: str


# Mapping for packages that support commit-pin installation. Add new entries
# here to enable the same behavior for other packages.
DEP_COMMIT_PINS: Dict[str, DepCommitPin] = {
    "circuit-tracer": DepCommitPin(
        package_name="circuit-tracer",
        env_var="IT_USE_CT_COMMIT_PIN",
        dep_def_loc="examples",
        pin_filename="circuit_tracer_pin.txt",
        repo_base_url="https://github.com/speediedan/circuit-tracer.git",
    ),
}

os.makedirs(REQ_DIR, exist_ok=True)
os.makedirs(CI_REQ_DIR, exist_ok=True)


def write_file(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def load_pyproject():
    with open(PYPROJECT_PATH, "r") as f:
        return toml.load(f)


def convert_pin_file(pin_file_path: str, repo_base_url: str, pkg_name: str) -> List[str]:
    """Read a pin file and convert lines into pip-installable requirement strings.

    Each non-empty, non-comment line may be:
      - a bare commit hash (40 or 64 hex chars) -> convert to git+{repo}@{hash}#egg={pkg}
      - a git+ URL or a name@rev entry -> return as-is
      - any other string -> returned as-is
    """
    if not os.path.exists(pin_file_path):
        return []
    out: List[str] = []
    with open(pin_file_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if all(c in "0123456789abcdef" for c in s.lower()) and len(s) in (40, 64):
                out.append(f"git+{repo_base_url}@{s}#egg={pkg_name}")
            elif s.startswith("git+") or "@" in s:
                out.append(s)
            else:
                out.append(s)
    return out


def generate_top_level_files(pyproject, output_dir=REQ_DIR):
    project = pyproject.get("project", {})
    core_reqs = project.get("dependencies", [])
    # Write top-level requirement files into the repository-level `requirements/` directory
    repo_requirements_dir = os.path.join(REPO_ROOT, "requirements")
    os.makedirs(repo_requirements_dir, exist_ok=True)

    write_file(os.path.join(repo_requirements_dir, "base.txt"), core_reqs)
    opt_deps = project.get("optional-dependencies", {})
    for group, reqs in opt_deps.items():
        write_file(os.path.join(repo_requirements_dir, f"{group}.txt"), reqs)


def generate_pip_compile_inputs(pyproject, ci_output_dir=CI_REQ_DIR):
    project = pyproject.get("project", {})
    tool_cfg = pyproject.get("tool", {}).get("ci_pinning", {})
    post_upgrades = tool_cfg.get("post_upgrades", {}) or {}
    platform_dependent = tool_cfg.get("platform_dependent", []) or []

    req_in_lines = []
    platform_dependent_lines = []
    direct_packages = []

    def normalize_package_name(name):
        return name.lower().replace("_", "-")

    def add_lines_from(list_or_none):
        if not list_or_none:
            return
        for r in list_or_none:
            parts = re.split(r"[\s\[\]=<>!;@]", r)
            pkg_name = parts[0].lower() if parts and parts[0] else ""
            if normalize_package_name(pkg_name) in {normalize_package_name(k) for k in post_upgrades}:
                continue
            is_platform_pkg = any(
                fnmatch.fnmatch(normalize_package_name(pkg_name), normalize_package_name(pattern))
                for pattern in platform_dependent
            )
            if is_platform_pkg:
                platform_dependent_lines.append(r)
                direct_packages.append(pkg_name)
                continue

            req_in_lines.append(r)
            direct_packages.append(pkg_name)

    add_lines_from(project.get("dependencies", []))

    opt_deps = project.get("optional-dependencies", {})
    groups_to_include_completely = ["test", "examples", "lightning"]

    for group, reqs in opt_deps.items():
        if not reqs:
            continue

        if group in groups_to_include_completely:
            add_lines_from(reqs)
        else:
            for req in reqs:
                parts = re.split(r"[\s\[\]=<>!;@]", req)
                pkg_name = parts[0].lower() if parts and parts[0] else ""
                post_upgrade_names = {normalize_package_name(k) for k in post_upgrades}
                if normalize_package_name(pkg_name) in post_upgrade_names:
                    continue
                is_platform_pkg = any(
                    fnmatch.fnmatch(normalize_package_name(pkg_name), normalize_package_name(pattern))
                    for pattern in platform_dependent
                )
                if is_platform_pkg:
                    platform_dependent_lines.append(req)
                    direct_packages.append(pkg_name)
                    continue
                continue

    def determine_dep_commit_lines(dep_key: str, pyproject: dict) -> List[str]:
        """Generalized selection logic for a dependency that supports commit-pin installs.

        Returns a list of requirement strings to add to requirements.in (may be empty).
        """
        if dep_key not in DEP_COMMIT_PINS:
            return []

        cfg = DEP_COMMIT_PINS[dep_key]
        env_flag = os.getenv(cfg.env_var, "").lower()
        if env_flag in ("1", "true", "yes"):
            print(f"{cfg.env_var} is set -> using pin file {cfg.pin_filename} for {cfg.package_name}")
            pin_path = os.path.join(CI_REQ_DIR, cfg.pin_filename)
            return convert_pin_file(pin_path, cfg.repo_base_url, cfg.package_name)

        # Look for the dependency in the declared pyproject extra/base location
        project = pyproject.get("project", {})
        if cfg.dep_def_loc == "dependencies":
            candidates = project.get("dependencies", []) or []
        else:
            opt_deps = project.get("optional-dependencies", {}) or {}
            candidates = opt_deps.get(cfg.dep_def_loc, []) or []

        for req in candidates:
            if cfg.package_name in req.lower() or cfg.package_name.replace("-", "_") in req.lower():
                print(
                    f"Found {cfg.package_name} in pyproject.{cfg.dep_def_loc} ->",
                    "using pyproject-specified requirement:",
                    req,
                )
                return [req]

        # fallback to pin file
        print(
            f"{cfg.package_name} not found in pyproject.{cfg.dep_def_loc};",
            f"falling back to pin file {cfg.pin_filename} if present",
        )
        pin_path = os.path.join(CI_REQ_DIR, cfg.pin_filename)
        return convert_pin_file(pin_path, cfg.repo_base_url, cfg.package_name)

    # Ascertain commit-pin dependencies via the DEP_COMMIT_PINS mapping.
    for dep_key, cfg in DEP_COMMIT_PINS.items():
        dep_lines = determine_dep_commit_lines(dep_key, pyproject)
        if not dep_lines:
            continue
        req_in_lines.extend(dep_lines)
        # Track the package as a direct package when appropriate
        for line in dep_lines:
            if cfg.package_name in line or cfg.package_name.replace("-", "_") in line:
                direct_packages.append(cfg.package_name)

    in_path = os.path.join(ci_output_dir, "requirements.in")
    write_file(in_path, req_in_lines)

    post_lines = []
    for pkg, spec in post_upgrades.items():
        spec_str = spec.strip()
        if re.match(r"^[<>=!].+", spec_str):
            post_lines.append(f"{pkg}{spec_str}")
        else:
            post_lines.append(f"{pkg}=={spec_str}")
    write_file(POST_UPGRADES_PATH, post_lines)

    platform_path = os.path.join(CI_REQ_DIR, "platform_dependent.txt")
    write_file(platform_path, platform_dependent_lines)

    return in_path, POST_UPGRADES_PATH, platform_path, direct_packages


def run_pip_compile(req_in_path, output_path):
    pip_compile = shutil.which("pip-compile")
    if not pip_compile:
        print("pip-compile not found in PATH; install pip-tools to generate full pinned requirements.txt")
        return False
    cmd = [pip_compile, "--output-file", output_path, req_in_path, "--upgrade"]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)
    return True


def post_process_pinned_requirements(requirements_path, platform_dependent_path, platform_patterns, direct_packages):
    if not os.path.exists(requirements_path):
        return

    with open(requirements_path, "r") as f:
        lines = f.readlines()

    requirements_lines = []
    platform_dependent_lines = []

    existing_platform_deps = []
    if os.path.exists(platform_dependent_path):
        with open(platform_dependent_path, "r") as f:
            existing_platform_deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def normalize_package_name(name):
        return name.lower().replace("_", "-")

    direct_packages_normalized = {normalize_package_name(pkg) for pkg in direct_packages}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            requirements_lines.append(lines[i])
            i += 1
            continue

        if " @ " in line:
            pkg_name = line.split(" @ ")[0].strip().lower()
        else:
            parts = re.split(r"[\[\]=<>!;]", line)
            pkg_name = parts[0].strip().lower() if parts else ""

        pkg_name_normalized = normalize_package_name(pkg_name)

        is_platform_dependent = any(
            fnmatch.fnmatch(pkg_name_normalized, pattern.replace("_", "-")) for pattern in platform_patterns
        )

        is_direct_dependency = pkg_name_normalized in direct_packages_normalized

        if is_platform_dependent:
            flexible_req = pkg_name_normalized
            platform_dependent_lines.append(flexible_req)
            i += 1
            while i < len(lines) and lines[i].strip().startswith("#"):
                i += 1
        elif is_direct_dependency:
            requirements_lines.append(lines[i])
            i += 1
            while i < len(lines) and lines[i].strip().startswith("#"):
                requirements_lines.append(lines[i])
                i += 1
        else:
            i += 1
            while i < len(lines) and lines[i].strip().startswith("#"):
                i += 1

    with open(requirements_path, "w") as f:
        for line in requirements_lines:
            f.write(line.rstrip() + "\n")

    all_platform_deps = list(set(existing_platform_deps + platform_dependent_lines))
    all_platform_deps.sort()

    with open(platform_dependent_path, "w") as f:
        for pkg in all_platform_deps:
            f.write(pkg.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Regenerate requirements files from pyproject.toml")
    parser.add_argument("--mode", choices=["top-level", "pip-compile"], default="top-level")
    parser.add_argument("--ci-output-dir", default=CI_REQ_DIR)
    args = parser.parse_args()

    pyproject = load_pyproject()

    generate_top_level_files(pyproject)

    if args.mode == "pip-compile":
        in_path, post_path, platform_path, direct_packages = generate_pip_compile_inputs(pyproject, args.ci_output_dir)
        out_path = os.path.join(args.ci_output_dir, "requirements.txt")
        try:
            success = run_pip_compile(in_path, out_path)
            if success:
                tool_cfg = pyproject.get("tool", {}).get("ci_pinning", {})
                platform_dependent = tool_cfg.get("platform_dependent", []) or []
                post_process_pinned_requirements(out_path, platform_path, platform_dependent, direct_packages)
                print(f"Generated pinned requirements at {out_path}")
                print(f"Generated post-upgrades at {post_path}")
                print(f"Generated platform-dependent packages at {platform_path}")
            else:
                print(f"Generated {in_path}, {post_path}, and {platform_path}.")
                print("To create a pinned requirements.txt, install pip-tools and run:")
                print(f"  pip-compile {in_path} --output-file {out_path}")
        except subprocess.CalledProcessError as e:
            print("pip-compile failed:", e)
            print(
                f"Generated inputs at {in_path}, post-upgrades at {post_path}, "
                f"and platform-dependent at {platform_path}"
            )
    else:
        print("Wrote top-level base and optional group requirement files in requirements/ (no pip-compile run).")


if __name__ == "__main__":
    main()
