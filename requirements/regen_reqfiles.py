import argparse
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

    # Build requirements.in lines from top-level dependencies and optional groups
    req_in_lines = []
    def add_lines_from(list_or_none):
        if not list_or_none:
            return
        for r in list_or_none:
            # skip any packages that are declared in post_upgrades mapping
            # extract an approximate package name (handles extras and simple specifiers)
            parts = re.split(r"[\s\[\]=<>!;@]", r)
            pkg_name = parts[0].lower() if parts and parts[0] else ""
            if pkg_name in post_upgrades:
                continue
            req_in_lines.append(r)

    add_lines_from(project.get("dependencies", []))
    opt_deps = project.get("optional-dependencies", {})
    for group, reqs in opt_deps.items():
        add_lines_from(reqs)

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

    return in_path, POST_UPGRADES_PATH


def run_pip_compile(req_in_path, output_path):
    pip_compile = shutil.which("pip-compile")
    if not pip_compile:
        print("pip-compile not found in PATH; install pip-tools to generate full pinned requirements.txt")
        return False
    cmd = [pip_compile, "--output-file", output_path, req_in_path]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)
    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate requirements files from pyproject.toml")
    parser.add_argument("--mode", choices=["top-level", "pip-compile"], default="top-level")
    parser.add_argument("--ci-output-dir", default=CI_REQ_DIR)
    args = parser.parse_args()

    pyproject = load_pyproject()

    # always keep the simple top-level files for developer convenience
    generate_top_level_files(pyproject)

    if args.mode == "pip-compile":
        in_path, post_path = generate_pip_compile_inputs(pyproject, args.ci_output_dir)
        # attempt to run pip-compile to produce a fully pinned requirements.txt
        out_path = os.path.join(args.ci_output_dir, "requirements.txt")
        try:
            success = run_pip_compile(in_path, out_path)
            if not success:
                print(f"Generated {in_path} and {post_path}.")
                print("To create a pinned requirements.txt, install pip-tools and run:")
                print(f"  pip-compile {in_path} --output-file {out_path}")
        except subprocess.CalledProcessError as e:
            print("pip-compile failed:", e)
            print(f"Generated inputs at {in_path} and post-upgrades at {post_path}")
    else:
        print("Wrote top-level base and optional group requirement files in requirements/ (no pip-compile run).")


if __name__ == "__main__":
    main()
