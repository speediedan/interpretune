#!/bin/bash
#
# Utility script to build IT environments
# Usage examples:
# build latest:
#   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest
# build latest with specific pytorch nightly:
#   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --torch_dev_ver=dev20240201
# build latest with torch test channel:
#    ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --torch_test_channel
set -eo pipefail

unset repo_home
unset target_env_name
unset torch_dev_ver
unset torch_test_channel
unset fts_from_source
unset ct_from_source
unset pip_install_flags
unset ct_commit_pin
unset regen_with_pip_compile
unset apply_post_upgrades
unset no_ci_reqs

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo-home input]
   [ --target-env-name input ]
   [ --torch-dev-ver input ]
   [ --torch-test-channel ]
   [ --fts-from-source "path" ]
   [ --ct-from-source "path" ]
   [ --pip-install-flags "flags" ]
   [ --ct-commit-pin ]
   [ --no-ci-reqs ]
   [ --regen-with-pip-compile ]
   [ --apply-post-upgrades ]
   [ --help ]
   Examples:
    # build latest:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest
    # build latest with specific pytorch nightly:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --torch-dev-ver=dev20240201
    # build latest with torch test channel:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --torch-test-channel
    # build latest with FTS from source:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --fts-from-source=${HOME}/repos/finetuning-scheduler
    # build latest with circuit-tracer from source:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --ct-from-source=${HOME}/repos/circuit-tracer
    # build latest with no cache directory:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --pip-install-flags="--no-cache-dir"
    # build latest without using CT commit pinning:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest
    # build latest and regenerate CI pinned requirements:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --regen-with-pip-compile
    # build latest and apply post-upgrades:
    #   ./build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --apply-post-upgrades
EOF
exit 1
}

args=$(getopt -o '' --long repo-home:,target-env-name:,torch-dev-ver:,torch-test-channel,fts-from-source:,ct-from-source:,pip-install-flags:,ct-commit-pin,no-ci-reqs,regen-with-pip-compile,apply-post-upgrades,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo-home)  repo_home=$2    ; shift 2  ;;
    --target-env-name)  target_env_name=$2  ; shift 2 ;;
    --torch-dev-ver)   torch_dev_ver=$2   ; shift 2 ;;
    --torch-test-channel)   torch_test_channel=1 ; shift  ;;
    --fts-from-source)   fts_from_source=$2 ; shift 2 ;;
    --ct-from-source)   ct_from_source=$2 ; shift 2 ;;
    --pip-install-flags)   pip_install_flags=$2 ; shift 2 ;;
    --ct-commit-pin)   ct_commit_pin=1 ; shift  ;;
    --no-ci-reqs)      no_ci_reqs=1 ; shift ;;
    --regen-with-pip-compile) regen_with_pip_compile=1 ; shift ;;
    --apply-post-upgrades) apply_post_upgrades=1 ; shift ;;
    --help)    usage      ; shift   ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Use pip_install_flags in pip commands
pip_install_flags=${pip_install_flags:-""}

# Expand leading ~ in common path arguments so users can pass --repo-home=~/repos/...
expand_tilde(){
    local p="$1"
    if [[ -n "$p" ]] && [[ "$p" == ~* ]]; then
        # Use eval to expand ~ reliably
        eval echo "$p"
    else
        echo "$p"
    fi
}

repo_home=$(expand_tilde "${repo_home}")
fts_from_source=$(expand_tilde "${fts_from_source}")
ct_from_source=$(expand_tilde "${ct_from_source}")

# Source common utility functions
source ${repo_home}/scripts/infra_utils.sh

clear_activate_env(){
    $1 -m venv --clear ~/.venvs/${target_env_name}
    source ~/.venvs/${target_env_name}/bin/activate
    echo "Current venv prompt is now ${VIRTUAL_ENV_PROMPT}"
    pip install ${pip_install_flags} --upgrade pip
}

base_env_build(){
    case ${target_env_name} in
        it_latest)
            clear_activate_env python3.12
            if [[ -n ${torch_dev_ver} ]]; then
                # temporarily remove torchvision until it supports cu128 in nightly binary
                pip install ${pip_install_flags} --pre torch==2.9.0.${torch_dev_ver} --index-url https://download.pytorch.org/whl/nightly/cu128
            elif [[ $torch_test_channel -eq 1 ]]; then
                pip install ${pip_install_flags} --pre torch==2.9.0 --index-url https://download.pytorch.org/whl/test/cu128
            else
                pip install ${pip_install_flags} torch torchvision --index-url https://download.pytorch.org/whl/cu128
            fi
            ;;
        it_release)
            clear_activate_env python3.12
            pip install ${pip_install_flags} torch torchvision --index-url https://download.pytorch.org/whl/cu128
            ;;
        *)
            echo "no matching environment found, exiting..."
            exit 1
            ;;
    esac
}

it_install(){
    source ~/.venvs/${target_env_name}/bin/activate
    unset PACKAGE_NAME
    if [[ -n ${fts_from_source} ]]; then
        export USE_CI_COMMIT_PIN="1"
        echo "Installing FTS from source at ${fts_from_source}"
        cd ${fts_from_source}
        python -m pip install ${pip_install_flags} -e ".[all]" -r requirements/docs.txt
        unset USE_CI_COMMIT_PIN
    fi
    cd ${repo_home}

    # Optionally regenerate CI pinned requirements (pip-compile mode) if requested
    if [[ -n ${regen_with_pip_compile} ]]; then
        python -m pip install ${pip_install_flags} toml pip-tools
        echo "Regenerating CI pinned requirements (pip-compile mode)"
        python ${repo_home}/requirements/regen_reqfiles.py --mode pip-compile --ci-output-dir ${repo_home}/requirements/ci
    fi

    # If CI pinned requirements don't exist and user did not disable ci-reqs, regenerate them
    if [[ -z ${no_ci_reqs} ]] && [[ ! -f ${repo_home}/requirements/ci/requirements.txt ]]; then
        python -m pip install ${pip_install_flags} toml pip-tools
        echo "CI pinned requirements not found; regenerating requirements.in and post_upgrades."
        python ${repo_home}/requirements/regen_reqfiles.py --mode pip-compile --ci-output-dir ${repo_home}/requirements/ci
    fi

    # Set IT_USE_CT_COMMIT_PIN if --ct_commit_pin is specified
    if [[ -n ${ct_commit_pin} ]]; then
        export IT_USE_CT_COMMIT_PIN="1"
        echo "Using IT_USE_CT_COMMIT_PIN for circuit-tracer installation"
    else
        unset IT_USE_CT_COMMIT_PIN
        echo "Using version-based circuit-tracer installation"
    fi

    # Install project and extras; prefer CI pinned requirements if available
    if [[ -f ${repo_home}/requirements/ci/requirements.txt ]] && [[ -z ${no_ci_reqs} ]]; then
        # Install pinned requirements, then install editable package so CLI modules (interpretune.*) are importable
        python -m pip install ${pip_install_flags} -r ${repo_home}/requirements/ci/requirements.txt -r requirements/docs.txt || true
        # Ensure interpretune package is installed (editable install recommended during dev)
        python -m pip install ${pip_install_flags} -e ".[test,examples,lightning,profiling]"
    else
        python -m pip install ${pip_install_flags} -e ".[test,examples,lightning,profiling]" -r requirements/docs.txt
    fi

    cd ${repo_home}
    # Optionally apply post-upgrades if requested and file exists
    if [[ -n ${apply_post_upgrades} ]] && [[ -s ${repo_home}/requirements/ci/post_upgrades.txt ]]; then
        echo "Applying post-upgrades from requirements/ci/post_upgrades.txt"
        pip install --upgrade -r ${repo_home}/requirements/ci/post_upgrades.txt || true
    else
        echo "Skipping post-upgrades (flag not set or file empty)."
    fi

    if [[ -n ${ct_from_source} ]]; then
        echo "Installing circuit-tracer from source at ${ct_from_source}"
        cd ${ct_from_source}
        python -m pip install ${pip_install_flags} -e .
    else
        # Use the interpretune CLI tool to install circuit-tracer
        if [[ -n ${IT_USE_CT_COMMIT_PIN} ]]; then
            echo "Installing circuit-tracer via interpretune CLI tool"
            python -m interpretune.tools.install_circuit_tracer --ct-commit-pin
        fi
    fi

    pyright -p pyproject.toml
    pre-commit install
    git lfs install
    python ${repo_home}/requirements/collect_env_details.py
    # Print environment/package diagnostics using the CI helper script
    python ${repo_home}/scripts/ci_print_env_versions.py ${repo_home}
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
