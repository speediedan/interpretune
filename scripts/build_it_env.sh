#!/usr/bin/env bash
#
# Interpretune environment builder using uv
# Uses uv pip with traditional venv activation for maximum control
#
# Torch handling:
# - By default, installs stable torch with CUDA 12.8 from PyTorch stable channel (--torch-backend=cu128)
# - If requirements/ci/torch-pre.txt exists, installs torch prerelease (nightly or test)
# - Use --torch-backend=cpu for CPU-only environments (e.g., GitHub Actions runners)
# - Use --torch-backend=auto for automatic backend detection
#
# Usage examples:
#   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest
#   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler"
#   # With UV_OVERRIDE to prevent torch downgrade from from-source packages:
#   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest \
#     --from-source="nnsight:${HOME}/repos/nnsight:all:UV_OVERRIDE=${HOME}/repos/interpretune/requirements/ci/torch-override.txt"
set -eo pipefail

# This script uses associative arrays and namerefs, which require bash >= 4.3.
# macOS ships bash 3.2 at /bin/bash, so we check early and fail with a clear message.
# On macOS: `brew install bash` provides a modern bash that `/usr/bin/env bash` will
# resolve (Homebrew's bin directory precedes /bin on PATH in default brew setups).
if [[ ${BASH_VERSINFO[0]:-0} -lt 4 ]] || { [[ ${BASH_VERSINFO[0]} -eq 4 ]] && [[ ${BASH_VERSINFO[1]:-0} -lt 3 ]]; }; then
    echo "ERROR: $(basename "$0") requires bash >= 4.3 (found ${BASH_VERSION:-unknown})." >&2
    echo "On macOS, install a modern bash with 'brew install bash' and re-run this script" >&2
    echo "(the script resolves bash via '/usr/bin/env bash', which picks up the Homebrew bash)." >&2
    exit 1
fi

# Source shared infrastructure utilities
source "$(dirname "${BASH_SOURCE[0]}")/infra_utils.sh"

unset repo_home
unset target_env_name
unset uv_install_flags
unset from_source_spec
unset venv_dir
unset python_version
unset torch_backend
declare -a from_source_specs
declare -A from_source_packages

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo-home input]
   [ --target-env-name input ]
   [ --venv-dir input ]
   [ --python-version input ]  (uv --python spec, e.g. 3.12 or python3.12; default: 3.13)
   [ --torch-backend input ]  (cpu, cu128, auto; default: cu128 for CUDA 12.8)
   [ --from-source "package:path[:extras][:env_var=value...]" ] (can be specified multiple times)
   [ --uv-install-flags "flags" ]
   [ --help ]

   The --from-source flag can be specified multiple times for clarity, or use semicolons to separate specs.
   Format: "package:path[:extras][:env_var=value...]"
   - extras: optional, e.g., "all" or "dev,test"
   - env_var=value: optional, multiple env vars separated by colons
   Package names should use underscores (e.g., finetuning_scheduler, circuit_tracer, transformer_lens).
   Paths passed directly to --from-source and ~/ segments embedded in env-var values are expanded.
   Environment variables are set only during that package's installation and unset afterward.
   When splitting a long command across lines in bash, each continued line must end with '\'.

   Common environment variables for --from-source:
   - UV_OVERRIDE: Path to override file to prevent dependency downgrades
     Two override files are available:
     * torch-override.txt: Auto-generated, pins only torch. Use for simple cases.
     * overrides.txt: Manually maintained, pins torch AND triton. Use for packages
       like nnsight that constrain multiple dependencies.
     Example with torch-override.txt (torch only):
       --from-source="some_package:\${HOME}/repos/some-package:all:UV_OVERRIDE=\${HOME}/repos/interpretune/requirements/ci/torch-override.txt"
     Example with overrides.txt (torch + triton + transformer-lens):
       --from-source="nnsight:\${HOME}/repos/nnsight:all:UV_OVERRIDE=\${HOME}/repos/interpretune/requirements/ci/overrides.txt"
   - UV_EXCLUDE: Path to file listing packages to exclude from installation
     Useful when a from-source package would reinstall a package you want to keep:
     --from-source="circuit_tracer:\${HOME}/repos/circuit-tracer:dev:UV_EXCLUDE=\${HOME}/repos/interpretune/requirements/ci/excludes.txt"

   Torch Handling:
   - By default, uses stable torch with CUDA 12.8 (--torch-backend=cu128) to match Docker images
   - If requirements/ci/torch-pre.txt exists, installs torch prerelease from nightly or test channel
   - torch-pre.txt format (3 lines): version, CUDA target (e.g., cu128), channel (nightly or test)
   - Use --torch-backend=cpu to force CPU-only torch (e.g., for CI environments)
   - Use --torch-backend=auto for automatic backend detection

   Venv Directory:
   - Use --venv-dir to explicitly set the venv BASE directory (recommended when using with manage_standalone_processes.sh)
   - The venv will be created at: <venv-dir>/<target-env-name>
   - If --venv-dir not set, uses IT_VENV_BASE environment variable as base (default: ~/.venvs)
   - Place venvs on same filesystem as UV cache to avoid hardlink warnings and improve performance

   Environment Variables:
   - IT_VENV_BASE: Base directory for venvs when --venv-dir not specified (default: ~/.venvs)
     Example: export IT_VENV_BASE=/mnt/cache/username/.venvs

   Examples:
    # build latest (auto-selects CUDA or CPU torch):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest

    # build latest with CPU-only torch (for CI):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --torch-backend=cpu

    # build latest with single package from source (no extras):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"

    # build latest with package from source with extras:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all"

    # build latest with package from source with extras and env var:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1"

    # build latest with package from source with env var but no extras:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler::USE_CI_COMMIT_PIN=1"

    # build latest with multiple packages from source (using multiple --from-source flags):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest \\
    #     --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1" \\
    #     --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"

    # build latest with multiple packages from source (using semicolon separator):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1;circuit_tracer:${HOME}/repos/circuit-tracer"

    # build latest with transformer_lens from source:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="transformer_lens:${HOME}/repos/TransformerLens"

    # build latest with no cache:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --uv-install-flags="--no-cache"
EOF
exit 1
}

# Portable long-option parser. GNU getopt (util-linux) is unavailable on macOS/BSD,
# so we parse manually. Both "--flag value" and "--flag=value" forms are supported
# (GNU getopt previously normalized the latter into the former).
while [[ $# -gt 0 ]]; do
  opt="$1"
  opt_value=""
  has_inline_value=0
  case "${opt}" in
    --*=*)
      opt_value="${opt#*=}"
      opt="${opt%%=*}"
      has_inline_value=1
      ;;
  esac
  case "${opt}" in
    --help)    usage ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    --repo-home|--target-env-name|--venv-dir|--python-version|--torch-backend|--from-source|--uv-install-flags)
      if [[ ${has_inline_value} -eq 0 ]]; then
        if [[ $# -lt 2 ]]; then
          >&2 echo "Missing value for option: ${opt}"
          usage
        fi
        opt_value="$2"
        shift
      fi
      case "${opt}" in
        --repo-home)  repo_home=${opt_value} ;;
        --target-env-name)  target_env_name=${opt_value} ;;
        --venv-dir)  venv_dir=${opt_value} ;;
        --python-version)  python_version=${opt_value} ;;
        --torch-backend)   torch_backend=${opt_value} ;;
        --from-source)   from_source_specs+=("${opt_value}") ;;
        --uv-install-flags)   uv_install_flags=${opt_value} ;;
      esac
      ;;
    *) >&2 echo "Unsupported option: $1"
       usage ;;
  esac
  shift
done

# Combine multiple --from-source flags into single spec with semicolon separator
if [[ ${#from_source_specs[@]} -gt 0 ]]; then
    from_source_spec=$(IFS=';'; echo "${from_source_specs[*]}")
fi

# Parse from-source specifications using shared infra_utils function
if [[ -n ${from_source_spec} ]]; then
    from_source_spec=$(strip_quotes "${from_source_spec}")
    parse_from_source_specs "${from_source_spec}" from_source_packages || exit 1
fi

# Use uv_install_flags in uv pip commands
uv_install_flags=${uv_install_flags:-""}

# Expand leading ~ in common path arguments so users can pass --repo-home=~/repos/...
repo_home=$(expand_tilde "${repo_home}")
venv_dir=$(expand_tilde "${venv_dir}")

# Expand tilde in from_source_packages paths (which are in format "path|extras|env_vars")
for pkg in "${!from_source_packages[@]}"; do
    pkg_spec="${from_source_packages[$pkg]}"
    IFS='|' read -r pkg_path pkg_extras pkg_env_vars <<< "${pkg_spec}"
    pkg_path=$(expand_tilde "${pkg_path}")
    from_source_packages[$pkg]="${pkg_path}|${pkg_extras}|${pkg_env_vars}"
done

# Determine venv path using centralized function from infra_utils.sh
# Priority: 1) --venv-dir flag, 2) IT_VENV_BASE env var, 3) default ~/.venvs
# Placing venvs on same filesystem as UV cache avoids hardlink warnings and improves performance
venv_path=$(determine_venv_path "${venv_dir}" "${target_env_name}")

# Set default Python version if not specified
python_version=${python_version:-"3.13"}  # uv version spec (resolves/downloads a managed interpreter); explicit binary names (python3.12) still accepted

# Set default torch backend if not specified (auto = auto-detect CUDA/CPU)
torch_backend=${torch_backend:-"cu128"}

# Torch prerelease configuration file
torch_pre_file="${repo_home}/requirements/ci/torch-pre.txt"

clear_activate_env(){
    echo "Creating/clearing venv at ${venv_path} with ${python_version}"
    uv venv --clear "${venv_path}" --python ${python_version}
    source "${venv_path}/bin/activate"
    echo "Current venv prompt is now ${VIRTUAL_ENV_PROMPT}"
    uv pip install ${uv_install_flags} --upgrade pip setuptools wheel
}

base_env_build(){
    clear_activate_env

    # Read torch prerelease configuration if it exists
    read_torch_pre_config "${torch_pre_file}"

    if [[ -n "${TORCH_PRE_VERSION}" ]]; then
        # Install torch prerelease from nightly/test channel
        local cuda_target="${TORCH_PRE_CUDA:-cu128}"
        local index_url
        index_url=$(get_torch_index_url "${TORCH_PRE_CHANNEL}" "${cuda_target}")
        echo "Installing torch prerelease: ${TORCH_PRE_VERSION} from ${TORCH_PRE_CHANNEL}/${cuda_target}..."
        uv pip install ${uv_install_flags} --prerelease=if-necessary-or-explicit "torch==${TORCH_PRE_VERSION}" --index-url "${index_url}"
    else
        # Install stable torch - use --torch-backend for automatic backend selection
        echo "Installing stable torch with --torch-backend=${torch_backend}..."
        uv pip install ${uv_install_flags} torch --torch-backend=${torch_backend}
    fi
}

it_install(){
    source "${venv_path}/bin/activate"
    cd "${repo_home}"

    # installation strategy: locked CI reqs → git-deps → from-source
    ci_reqs_file="${repo_home}/requirements/ci/requirements.txt"

    if [[ ! -f "${ci_reqs_file}" ]]; then
        echo "⚠ ERROR: Locked CI requirements not found at ${ci_reqs_file}"
        echo "Please regenerate with: bash requirements/utils/lock_ci_requirements.sh"
        exit 1
    fi

    echo "Using locked CI requirements from ${ci_reqs_file}..."

    # 1. Install interpretune in editable mode + git-deps group (uv doesn't currently support url deps in locked reqs)
    echo "Installing interpretune in editable mode..."
    uv pip install ${uv_install_flags} -e . --group git-deps

    # 2. Install locked CI requirements (all PyPI packages)
    # Note: Torch is already installed in base_env_build, so we don't use --torch-backend here.
    # The lockfile excludes torch (generated with --no-emit-package torch) but includes packages
    # that depend on torch. Using --torch-backend would try to resolve those packages against
    # PyTorch's special index, which doesn't have all packages (e.g., torch-tb-profiler).
    echo "Installing locked dependencies..."
    uv pip install ${uv_install_flags} -r "${ci_reqs_file}"

    # 3. Install from-source packages (override any PyPI/git versions)
    if [[ ${#from_source_packages[@]} -gt 0 ]]; then
        echo "Installing from-source packages (these will override any PyPI/git versions)..."
        install_from_source_packages from_source_packages "${venv_path}" "${uv_install_flags}"
    fi

    # 3.5 Install no-deps git packages (their own pins would conflict with the integrated env;
    # their runtime deps are satisfied by the locked CI requirements — see the file's comments)
    nodeps_reqs_file="${repo_home}/requirements/ci/nodeps_git_requirements.txt"
    if [[ -f "${nodeps_reqs_file}" ]]; then
        echo "Installing no-deps git packages from ${nodeps_reqs_file}..."
        uv pip install ${uv_install_flags} --no-deps -r "${nodeps_reqs_file}"
    fi

    # 4. Setup git hooks and type checking
    cd "${repo_home}"
    echo "Setting up git hooks and running type checks..."
    pyright -p pyproject.toml || echo "⚠ pyright check had issues, continuing..."
    pre-commit install
    git lfs install

    # 5. Display environment info
    echo "Collecting environment details..."
    python ${repo_home}/requirements/utils/collect_env_details.py --packages-only
}

d=`date +%Y%m%d%H%M%S`
echo "IT env build executing at ${d} PT"
echo "Beginning env removal/update for ${target_env_name}"
maybe_deactivate
echo "Beginning IT base env install for ${target_env_name}"
base_env_build
echo "Beginning IT dev install for ${target_env_name}"
it_install
echo "IT env successfully built for ${target_env_name}!"
