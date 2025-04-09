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
unset pip_install_flags

usage(){
>&2 cat << EOF
Usage: $0
   [ --repo_home input]
   [ --target_env_name input ]
   [ --torch_dev_ver input ]
   [ --torch_test_channel ]
   [ --fts_from_source ]
   [ --pip_install_flags "flags" ]
   [ --help ]
   Examples:
    # build latest:
    #   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest
    # build latest with specific pytorch nightly:
    #   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --torch_dev_ver=dev20240201
    # build latest with torch test channel:
    #   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --torch_test_channel
    # build latest with FTS from source:
    #   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --fts_from_source
    # build latest with no cache directory:
    #   ./build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --pip_install_flags="--no-cache-dir"
EOF
exit 1
}

args=$(getopt -o '' --long repo_home:,target_env_name:,torch_dev_ver:,torch_test_channel,fts_from_source,pip_install_flags:,help -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}
while :
do
  case $1 in
    --repo_home)  repo_home=$2    ; shift 2  ;;
    --target_env_name)  target_env_name=$2  ; shift 2 ;;
    --torch_dev_ver)   torch_dev_ver=$2   ; shift 2 ;;
    --torch_test_channel)   torch_test_channel=1 ; shift  ;;
    --fts_from_source)   fts_from_source=1 ; shift  ;;
    --pip_install_flags)   pip_install_flags=$2 ; shift 2 ;;
    --help)    usage      ; shift   ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

# Use pip_install_flags in pip commands
pip_install_flags=${pip_install_flags:-""}

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
                pip install ${pip_install_flags} --pre torch==2.7.0.${torch_dev_ver} --index-url https://download.pytorch.org/whl/nightly/cu128
            elif [[ $torch_test_channel -eq 1 ]]; then
                pip install ${pip_install_flags} --pre torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/test/cu128
            else
                pip install ${pip_install_flags} torch torchvision --index-url https://download.pytorch.org/whl/cu126
            fi
            ;;
        it_latest_pt_2_4)
            clear_activate_env python3.11
            pip install ${pip_install_flags} torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu118
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
    if [[ $fts_from_source -eq 1 ]]; then
        echo "Installing FTS from source"
        cd ~/repos/finetuning-scheduler  # TODO: make fts source path configurable
        python -m pip install ${pip_install_flags} -e ".[all]" -r requirements/docs.txt
    fi
    cd ${repo_home}

    python -m pip install ${pip_install_flags} -e ".[test,examples,lightning]" -r requirements/docs.txt
    pip install ${pip_install_flags} circuitsvis --no-deps
    rm -rf .mypy_cache
    mypy --install-types --non-interactive
    pre-commit install
    git lfs install
    python -c "import importlib.metadata; import torch; import lightning.pytorch; import transformer_lens; import finetuning_scheduler; import interpretune;
for package in ['torch', 'lightning', 'transformer_lens', 'finetuning_scheduler', 'interpretune']:
    print(f'{package} version: {importlib.metadata.distribution(package).version}');"
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
