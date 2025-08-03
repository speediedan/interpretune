import os
import toml

# Set REQ_DIR to the directory containing this script
REQ_DIR = os.path.dirname(os.path.abspath(__file__))
# Set PYPROJECT_PATH to the parent directory's pyproject.toml
PYPROJECT_PATH = os.path.join(os.path.dirname(REQ_DIR), "pyproject.toml")

# Ensure requirements directory exists (should always exist)
os.makedirs(REQ_DIR, exist_ok=True)

def write_requirements(filename, reqs):
    path = os.path.join(REQ_DIR, filename)
    with open(path, "w") as f:
        for req in reqs:
            f.write(f"{req}\n")

def main():
    with open(PYPROJECT_PATH, "r") as f:
        pyproject = toml.load(f)

    project = pyproject.get("project", {})
    # Write core dependencies to base.txt
    core_reqs = project.get("dependencies", [])
    write_requirements("base.txt", core_reqs)

    # Write each optional dependency group to its own file
    opt_deps = project.get("optional-dependencies", {})
    for group, reqs in opt_deps.items():
        write_requirements(f"{group}.txt", reqs)

if __name__ == "__main__":
    main()
