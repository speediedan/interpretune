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
    direct_packages = []  # Track all direct packages we're including

    def normalize_package_name(name):
        """Normalize package names to handle underscores vs dashes."""
        return name.lower().replace('_', '-')

    def add_lines_from(list_or_none):
        if not list_or_none:
            return
        for r in list_or_none:
            # extract an approximate package name (handles extras and simple specifiers)
            parts = re.split(r"[\s\[\]=<>!;@]", r)
            pkg_name = parts[0].lower() if parts and parts[0] else ""

            # skip any packages that are declared in post_upgrades mapping
            if normalize_package_name(pkg_name) in {normalize_package_name(k) for k in post_upgrades}:
                continue

            # separate platform-dependent packages for special handling (supports glob patterns)
            is_platform_pkg = any(
                fnmatch.fnmatch(normalize_package_name(pkg_name), normalize_package_name(pattern))
                for pattern in platform_dependent
            )
            if is_platform_pkg:
                platform_dependent_lines.append(r)
                direct_packages.append(pkg_name)  # Still track as direct package
                continue

            req_in_lines.append(r)
            direct_packages.append(pkg_name)  # Track as direct package

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
                if normalize_package_name(pkg_name) in [normalize_package_name(p) for p in key_packages_to_constrain]:
                    # Use the original req format with package name handling
                    parts = re.split(r"[\s\[\]=<>!;@]", req)
                    pkg_name = parts[0].lower() if parts and parts[0] else ""

                    # skip any packages that are declared in post_upgrades mapping
                    post_upgrade_names = {normalize_package_name(k) for k in post_upgrades}
                    if normalize_package_name(pkg_name) in post_upgrade_names:
                        continue

                    # separate platform-dependent packages for special handling (supports glob patterns)
                    is_platform_pkg = any(
                        fnmatch.fnmatch(normalize_package_name(pkg_name), normalize_package_name(pattern))
                        for pattern in platform_dependent
                    )
                    if is_platform_pkg:
                        platform_dependent_lines.append(req)
                        direct_packages.append(pkg_name)  # Still track as direct package
                        continue

                    req_in_lines.append(req)
                    direct_packages.append(pkg_name)  # Track as direct package

    # include circuit-tracer pin(s) if present
    circuit_tracer_lines = convert_circuit_tracer_pin()
    req_in_lines.extend(circuit_tracer_lines)
    # Add circuit-tracer to direct packages
    for line in circuit_tracer_lines:
        if "circuit-tracer" in line or "circuit_tracer" in line:
            direct_packages.append("circuit-tracer")

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

    return in_path, POST_UPGRADES_PATH, platform_path, direct_packages


def run_pip_compile(req_in_path, output_path):
    pip_compile = shutil.which("pip-compile")
    if not pip_compile:
        print("pip-compile not found in PATH; install pip-tools to generate full pinned requirements.txt")
        return False
    cmd = [pip_compile, "--output-file", output_path, req_in_path]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)
    return True


def post_process_pinned_requirements(requirements_path, platform_dependent_path, platform_patterns, direct_packages):
    """Post-process the pinned requirements to move platform-dependent packages to separate file and filter out
    transitive dependencies to only keep direct dependencies.

    This handles cases where:
    1. Platform-dependent direct dependencies need to be moved to separate file for flexible installation
    2. Transitive dependencies get pinned but we only want to constrain direct dependencies
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

    # Convert direct packages to normalized form (lowercase, replace underscores with dashes)
    # This handles the fact that pip normalizes package names
    def normalize_package_name(name):
        return name.lower().replace('_', '-')

    direct_packages_normalized = {normalize_package_name(pkg) for pkg in direct_packages}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Always include header comments and empty lines
        if not line or line.startswith('#'):
            requirements_lines.append(lines[i])
            i += 1
            continue

        # This is a package line (e.g., "accelerate==1.10.0")
        # Extract package name from pinned requirement

        # Handle special cases like git packages first
        if ' @ ' in line:
            # For git packages like "circuit-tracer @ git+...", extract the package name
            pkg_name = line.split(' @ ')[0].strip().lower()
        else:
            # For regular packages, extract name before any extras, version specifiers, etc.
            # Handle extras like "jsonargparse[signatures,typing-extensions]==4.40.2"
            parts = re.split(r'[\[\]=<>!;]', line)
            pkg_name = parts[0].strip().lower() if parts else ""

        # Normalize package name for comparison
        pkg_name_normalized = normalize_package_name(pkg_name)

        # Check if this package matches any platform-dependent pattern
        is_platform_dependent = any(
            fnmatch.fnmatch(pkg_name_normalized, pattern.replace('_', '-'))
            for pattern in platform_patterns
        )

        # Check if this is a direct dependency we want to constrain
        is_direct_dependency = pkg_name_normalized in direct_packages_normalized

        if is_platform_dependent:
            # Convert pinned requirement back to flexible constraint
            # e.g., "nvidia-cublas-cu12==12.8.4.1" -> "nvidia-cublas-cu12"
            flexible_req = pkg_name_normalized
            platform_dependent_lines.append(flexible_req)
            # Skip this package and its associated comment lines
            i += 1
            # Skip subsequent comment lines that belong to this package
            while i < len(lines) and lines[i].strip().startswith('#'):
                i += 1
        elif is_direct_dependency:
            # Keep direct dependencies that we explicitly want to constrain
            requirements_lines.append(lines[i])
            i += 1
            # Keep the associated comment lines too
            while i < len(lines) and lines[i].strip().startswith('#'):
                requirements_lines.append(lines[i])
                i += 1
        else:
            # Skip transitive dependencies that we don't want to constrain
            i += 1
            # Skip subsequent comment lines that belong to this package
            while i < len(lines) and lines[i].strip().startswith('#'):
                i += 1

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
        in_path, post_path, platform_path, direct_packages = generate_pip_compile_inputs(pyproject, args.ci_output_dir)
        # attempt to run pip-compile to produce a fully pinned requirements.txt
        out_path = os.path.join(args.ci_output_dir, "requirements.txt")
        try:
            success = run_pip_compile(in_path, out_path)
            if success:
                # Post-process to move platform-dependent packages from pinned requirements
                # and filter out transitive dependencies to only keep direct dependencies
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
            print(f"Generated inputs at {in_path}, post-upgrades at {post_path}, "
                  f"and platform-dependent at {platform_path}")
    else:
        print("Wrote top-level base and optional group requirement files in requirements/ (no pip-compile run).")


if __name__ == "__main__":
    main()
