import argparse
import fnmatch
import os
import shlex
import subprocess
import shutil
import toml
import re

# Paths
REQ_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REQ_DIR)
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")
CI_REQ_DIR = os.path.join(REPO_ROOT, "requirements", "ci")
POST_UPGRADES_PATH = os.path.join(REPO_ROOT, "requirements", "post_upgrades.txt")
CIRCUIT_TRACER_PIN = os.path.join(REPO_ROOT, "requirements", "circuit_tracer_pin.txt")

os.makedirs(REQ_DIR, exist_ok=True)
os.makedirs(CI_REQ_DIR, exist_ok=True)


def write_file(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def load_pyproject():
    with open(PYPROJECT_PATH, "r") as f:
        return toml.load(f)


def convert_circuit_tracer_pin():
    """Read requirements/circuit_tracer_pin.txt and return a list of VCS requirement lines.

    The file may contain a single commit SHA; translate that to a git+ URL usable by pip. If the file already contains a
    full VCS spec, return it as-is.
    """
    if not os.path.exists(CIRCUIT_TRACER_PIN):
        return []
    out = []
    with open(CIRCUIT_TRACER_PIN, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # if it looks like a 40-char SHA, translate
            if all(c in "0123456789abcdef" for c in s.lower()) and len(s) in (40, 64):
                out.append(f"git+https://github.com/speediedan/circuit-tracer.git@{s}#egg=circuit-tracer")
            elif s.startswith("git+") or "@" in s:
                out.append(s)
            else:
                out.append(s)
    return out


def generate_top_level_files(pyproject, output_dir=REQ_DIR):
    project = pyproject.get("project", {})
    core_reqs = project.get("dependencies", [])
    write_file(os.path.join(output_dir, "base.txt"), core_reqs)
    opt_deps = project.get("optional-dependencies", {})
    for group, reqs in opt_deps.items():
        write_file(os.path.join(output_dir, f"{group}.txt"), reqs)


def generate_pip_compile_inputs(pyproject, ci_output_dir=CI_REQ_DIR):
    project = pyproject.get("project", {})
    tool_cfg = pyproject.get("tool", {}).get("ci_pinning", {})
    post_upgrades = tool_cfg.get("post_upgrades", {}) or {}
    platform_dependent = tool_cfg.get("platform_dependent", []) or []

    # Build requirements.in lines from top-level dependencies and optional groups
    req_in_lines = []
    platform_dependent_lines = []

    def add_lines_from(list_or_none):
        if not list_or_none:
            return
        for r in list_or_none:
            # extract an approximate package name (handles extras and simple specifiers)
            parts = re.split(r"[\s\[\]=<>!;@]", r)
            pkg_name = parts[0].lower() if parts and parts[0] else ""

            # skip any packages that are declared in post_upgrades mapping
            if pkg_name in post_upgrades:
                continue

            # separate platform-dependent packages for special handling (supports glob patterns)
            if any(fnmatch.fnmatch(pkg_name, pattern) for pattern in platform_dependent):
                platform_dependent_lines.append(r)
                continue

            req_in_lines.append(r)

    # Only include direct dependencies that we want to constrain
    # Core dependencies - always include these as they are our main requirements
    add_lines_from(project.get("dependencies", []))

    # Only include specific optional dependencies that we want to constrain
    # Rather than including all optional dependencies, only include key packages
    # that we want to have stable/consistent versions across platforms
    opt_deps = project.get("optional-dependencies", {})

    # Key packages from optional dependencies that we want to constrain
    key_packages_to_constrain = [
        # From lightning group - these are important ML packages we want stable
        "finetuning-scheduler",
        "bitsandbytes",  # will be moved to platform_dependent
        "peft",

        # From examples group - key packages for functionality
        # Note: we exclude many examples packages to avoid constraining transitive deps
        # like triton that cause platform issues
    ]

    # Only add specific packages from optional dependencies
    for group, reqs in opt_deps.items():
        if reqs:
            for req in reqs:
                # Extract package name
                parts = re.split(r"[\s\[\]=<>!;@]", req)
                pkg_name = parts[0].lower() if parts and parts[0] else ""

                # Only include if it's in our key packages list
                if pkg_name in [p.lower() for p in key_packages_to_constrain]:
                    # Use the original req format with package name handling
                    parts = re.split(r"[\s\[\]=<>!;@]", req)
                    pkg_name = parts[0].lower() if parts and parts[0] else ""

                    # skip any packages that are declared in post_upgrades mapping
                    if pkg_name in post_upgrades:
                        continue

                    # separate platform-dependent packages for special handling (supports glob patterns)
                    if any(fnmatch.fnmatch(pkg_name, pattern) for pattern in platform_dependent):
                        platform_dependent_lines.append(req)
                        continue

                    req_in_lines.append(req)

    # include circuit-tracer pin(s) if present
    req_in_lines.extend(convert_circuit_tracer_pin())

    # write requirements.in
    in_path = os.path.join(ci_output_dir, "requirements.in")
    write_file(in_path, req_in_lines)

    # write post_upgrades.txt as exact pins
    post_lines = []
    for pkg, ver in post_upgrades.items():
        post_lines.append(f"{pkg}=={ver}")
    write_file(POST_UPGRADES_PATH, post_lines)

    # write platform_dependent.txt with flexible constraints
    platform_path = os.path.join(REQ_DIR, "platform_dependent.txt")
    write_file(platform_path, platform_dependent_lines)

    return in_path, POST_UPGRADES_PATH, platform_path


def run_pip_compile(req_in_path, output_path):
    pip_compile = shutil.which("pip-compile")
    if not pip_compile:
        print("pip-compile not found in PATH; install pip-tools to generate full pinned requirements.txt")
        return False
    cmd = [pip_compile, "--output-file", output_path, req_in_path]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)
    return True


def post_process_pinned_requirements(requirements_path, platform_dependent_path, platform_patterns):
    """Post-process the pinned requirements to move platform-dependent packages to separate file.

    This handles cases where transitive dependencies (like nvidia-* packages) get pinned but should be treated as
    platform-dependent.
    """
    if not os.path.exists(requirements_path):
        return

    with open(requirements_path, 'r') as f:
        lines = f.readlines()

    requirements_lines = []
    platform_dependent_lines = []

    # Load existing platform-dependent packages
    existing_platform_deps = []
    if os.path.exists(platform_dependent_path):
        with open(platform_dependent_path, 'r') as f:
            existing_platform_deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            requirements_lines.append(line)
            continue

        # Extract package name from pinned requirement (e.g., "nvidia-cublas-cu12==12.8.4.1")
        parts = re.split(r'[=<>!;]', line)
        pkg_name = parts[0].strip().lower() if parts else ""

        # Check if this package matches any platform-dependent pattern
        is_platform_dependent = any(fnmatch.fnmatch(pkg_name, pattern) for pattern in platform_patterns)

        if is_platform_dependent:
            # Convert pinned requirement back to flexible constraint
            # e.g., "nvidia-cublas-cu12==12.8.4.1" -> "nvidia-cublas-cu12"
            flexible_req = pkg_name
            platform_dependent_lines.append(flexible_req)
        else:
            requirements_lines.append(line)

    # Write back the filtered requirements.txt
    with open(requirements_path, 'w') as f:
        for line in requirements_lines:
            f.write(line.rstrip() + '\n')

    # Update platform_dependent.txt with both existing and newly found packages
    all_platform_deps = list(set(existing_platform_deps + platform_dependent_lines))
    all_platform_deps.sort()  # Keep consistent ordering

    with open(platform_dependent_path, 'w') as f:
        for pkg in all_platform_deps:
            f.write(pkg.rstrip() + '\n')


def main():
    parser = argparse.ArgumentParser(description="Regenerate requirements files from pyproject.toml")
    parser.add_argument("--mode", choices=["top-level", "pip-compile"], default="top-level")
    parser.add_argument("--ci-output-dir", default=CI_REQ_DIR)
    args = parser.parse_args()

    pyproject = load_pyproject()

    # always keep the simple top-level files for developer convenience
    generate_top_level_files(pyproject)

    if args.mode == "pip-compile":
        in_path, post_path, platform_path = generate_pip_compile_inputs(pyproject, args.ci_output_dir)
        # attempt to run pip-compile to produce a fully pinned requirements.txt
        out_path = os.path.join(args.ci_output_dir, "requirements.txt")
        try:
            success = run_pip_compile(in_path, out_path)
            if success:
                # Post-process to move platform-dependent packages from pinned requirements
                tool_cfg = pyproject.get("tool", {}).get("ci_pinning", {})
                platform_dependent = tool_cfg.get("platform_dependent", []) or []
                transitive_platform_issues = tool_cfg.get("transitive_platform_issues", []) or []
                # Combine both types of problematic packages for post-processing
                all_platform_patterns = platform_dependent + transitive_platform_issues
                post_process_pinned_requirements(out_path, platform_path, all_platform_patterns)
                print(f"Generated pinned requirements at {out_path}")
                print(f"Generated post-upgrades at {post_path}")
                print(f"Generated platform-dependent packages at {platform_path}")
            else:
                print(f"Generated {in_path}, {post_path}, and {platform_path}.")
                print("To create a pinned requirements.txt, install pip-tools and run:")
                print(f"  pip-compile {in_path} --output-file {out_path}")
        except subprocess.CalledProcessError as e:
            print("pip-compile failed:", e)
            print(f"Generated inputs at {in_path}, post-upgrades at {post_path}, "
                  f"and platform-dependent at {platform_path}")
    else:
        print("Wrote top-level base and optional group requirement files in requirements/ (no pip-compile run).")


if __name__ == "__main__":
    main()
