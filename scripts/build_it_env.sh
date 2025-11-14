#!/bin/bash
#
# Interpretune environment builder using uv
# Uses uv pip with traditional venv activation for maximum control
#
# Usage examples:
#   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest
#   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --torch_dev_ver=dev20240201
#   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler"
set -eo pipefail

# Source shared infrastructure utilities
source "$(dirname "${BASH_SOURCE[0]}")/infra_utils.sh"

unset repo_home
unset target_env_name
unset torch_dev_ver
unset torch_test_channel
unset uv_install_flags
unset from_source_spec
unset venv_dir
declare -a from_source_specs
declare -A from_source_packages

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo-home input]
   [ --target-env-name input ]
   [ --venv-dir input ]
   [ --torch-dev-ver input ]
   [ --torch-test-channel ]
   [ --from-source "package:path[:extras][:env_var=value...]" ] (can be specified multiple times)
   [ --uv-install-flags "flags" ]
   [ --help ]

   The --from-source flag can be specified multiple times for clarity, or use semicolons to separate specs.
   Format: "package:path[:extras][:env_var=value...]"
   - extras: optional, e.g., "all" or "dev,test"
   - env_var=value: optional, multiple env vars separated by colons
   Package names should use underscores (e.g., finetuning_scheduler, circuit_tracer, transformer_lens).
   Paths will be expanded if they start with ~.
   Environment variables are set only during that package's installation and unset afterward.

   Venv Directory:
   - Use --venv-dir to explicitly set the venv BASE directory (recommended when using with manage_standalone_processes.sh)
   - The venv will be created at: <venv-dir>/<target-env-name>
   - If --venv-dir not set, uses IT_VENV_BASE environment variable as base (default: ~/.venvs)
   - Place venvs on same filesystem as UV cache to avoid hardlink warnings and improve performance

   Environment Variables:
   - IT_VENV_BASE: Base directory for venvs when --venv-dir not specified (default: ~/.venvs)
     Example: export IT_VENV_BASE=/mnt/cache/username/.venvs

   Examples:
    # build latest:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest

    # build latest with specific pytorch nightly:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --torch-dev-ver=dev20240201

    # build latest with torch test channel:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --torch-test-channel

    # build latest with single package from source (no extras):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"

    # build latest with package from source with extras:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all"

    # build latest with package from source with extras and env var:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1"

    # build latest with package from source with env var but no extras:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler::USE_CI_COMMIT_PIN=1"

    # build latest with multiple packages from source (using multiple --from-source flags):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1" --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"

    # build latest with multiple packages from source (using semicolon separator):
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1;circuit_tracer:${HOME}/repos/circuit-tracer"

    # build latest with transformer_lens from source:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="transformer_lens:${HOME}/repos/TransformerLens"

    # build latest with no cache:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --uv-install-flags="--no-cache"
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,target-env-name:,venv-dir:,torch-dev-ver:,torch-test-channel,from-source:,uv-install-flags:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home)  repo_home=$2    ; shift 2  ;;
    --target-env-name)  target_env_name=$2  ; shift 2 ;;
    --venv-dir)  venv_dir=$2  ; shift 2 ;;
    --torch-dev-ver)   torch_dev_ver=$2   ; shift 2 ;;
    --torch-test-channel)   torch_test_channel=1 ; shift  ;;
    --from-source)   from_source_specs+=("$2") ; shift 2 ;;
    --uv-install-flags)   uv_install_flags=$2 ; shift 2 ;;
    --help)    usage      ; shift   ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Combine multiple --from-source flags into single spec with semicolon separator
if [[ ${#from_source_specs[@]} -gt 0 ]]; then
    from_source_spec=$(IFS=';'; echo "${from_source_specs[*]}")
fi

# Parse from-source specifications using shared infra_utils function
if [[ -n ${from_source_spec} ]]; then
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

clear_activate_env(){
    local python_cmd=$1
    echo "Creating/clearing venv at ${venv_path} with ${python_cmd}"
    uv venv --clear "${venv_path}" --python ${python_cmd}
    source "${venv_path}/bin/activate"
    echo "Current venv prompt is now ${VIRTUAL_ENV_PROMPT}"
    uv pip install ${uv_install_flags} --upgrade pip setuptools wheel
}

base_env_build(){
    case ${target_env_name} in
        it_latest)
            clear_activate_env python3.12
            if [[ -n ${torch_dev_ver} ]]; then
                # temporarily remove torchvision until it supports cu128 in nightly binary
                uv pip install ${uv_install_flags} --pre torch==2.10.0.${torch_dev_ver} --index-url https://download.pytorch.org/whl/nightly/cu128
            elif [[ ${torch_test_channel} -eq 1 ]]; then
                uv pip install ${uv_install_flags} --pre torch==2.10.0 --index-url https://download.pytorch.org/whl/test/cu128
            else
                uv pip install ${uv_install_flags} torch --index-url https://download.pytorch.org/whl/cu128
            fi
            ;;
        it_release)
            clear_activate_env python3.12
            uv pip install ${uv_install_flags} torch --index-url https://download.pytorch.org/whl/cu128
            ;;
        *)
            echo "no matching environment found, exiting..."
            exit 1
            ;;
    esac
}

it_install(){
    source "${venv_path}/bin/activate"
    cd ${repo_home}

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
    echo "Installing locked dependencies..."
    uv pip install ${uv_install_flags} -r "${ci_reqs_file}"

    # 3. Install from-source packages (override any PyPI/git versions)
    if [[ ${#from_source_packages[@]} -gt 0 ]]; then
        echo "Installing from-source packages (these will override any PyPI/git versions)..."
        install_from_source_packages from_source_packages "${venv_path}" "${uv_install_flags}"
    fi

    # 4. Setup git hooks and type checking
    cd ${repo_home}
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
