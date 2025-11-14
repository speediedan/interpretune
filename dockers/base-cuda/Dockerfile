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
# initially based on https://bit.ly/3pdAf1G

ARG CUDA_VERSION=12.8.1
ARG OS_VER=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VER}

ARG PYTHON_VERSION=3.12
ARG PYTORCH_VERSION=2.9.1
ARG CUST_BUILD=0
ARG MKL_THREADING_LAYER=GNU

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    CUST_BUILD=${CUST_BUILD} \
    DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0" \
    MKL_THREADING_LAYER=${MKL_THREADING_LAYER} \
    MAKEFLAGS="-j2" \
    UV_NO_CACHE=1
    RUN apt-get update -qq --fix-missing  && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        cmake \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        software-properties-common \
        libopenmpi-dev \
        libpq5 \
        libpq-dev \
    && \
# Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    # Install uv for fast, reliable package management (system-wide)
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements/ ./requirements/

ENV PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages

RUN \
    # Create virtual environment with uv
    uv venv /tmp/venvs/it_dev --python python${PYTHON_VERSION} && \
    . /tmp/venvs/it_dev/bin/activate && \
    # set particular PyTorch version by default
    if [[ "${CUST_BUILD}" -eq 0 ]]; then \
        CUDA_VERSION_MM=${CUDA_VERSION%.*}; \
        uv pip install torch --no-cache \
        --find-links="https://download.pytorch.org/whl/cu${CUDA_VERSION_MM//'.'/''}/torch_stable.html" \
        --find-links="https://download.pytorch.org/whl/test/cu${CUDA_VERSION_MM//'.'/''}/torch" \
        --find-links="https://download.pytorch.org/whl/test/cu${CUDA_VERSION_MM//'.'/''}/pytorch-triton"; \
    else \
        # or target a specific cuda build, by specifying a particular index url w/...
        # ... default channel
        uv pip install torch --index-url https://download.pytorch.org/whl/cu128; \
        # ... pytorch nightly dev version
        # uv pip install --pre torch==2.9.0.dev20250811 --index-url https://download.pytorch.org/whl/nightly/cu128; \
        # ... test channel
        # uv pip install --pre torch==2.9.0 --index-url https://download.pytorch.org/whl/test/cu128; \
    fi && \
    # We avoid installing Lightning and other dependencies here as they are usually upgraded anyway later in
    # CI but we may re-enable in the future.
    # LIGHTNING_COMMIT=$(cat ./requirements/lightning_pin.txt) && \
    # uv pip install "lightning @ git+https://github.com/Lightning-AI/lightning.git@${LIGHTNING_COMMIT}#egg=lightning" --no-cache && \
    # Install all requirements
    # uv pip install -r requirements/devel.txt --no-cache && \
    # Ensure venv is fully writable by all users and clean up __pycache__ to avoid permission issues
    find /tmp/venvs/it_dev -type d -name __pycache__ -exec chmod -R 777 {} + 2>/dev/null || true && \
    chmod -R 777 /tmp/venvs/it_dev && \
    rm -rf requirements.* requirements/

RUN \
    set -x && \
    . /tmp/venvs/it_dev/bin/activate && \
    echo "Checking CUDA version:" && \
    CUDA_VERSION_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])") && \
    echo "CUDA Version Major: ${CUDA_VERSION_MAJOR}" && \
    echo "Checking Python version compatibility:" && \
    py_ver=$(python -c "print(int('$PYTHON_VERSION'.split('.') >= '3.12'.split('.')))") && \
    echo "Python version check result: ${py_ver}"

RUN \
    set -x && \
    echo "============= Environment Information =============" && \
    . /tmp/venvs/it_dev/bin/activate && \
    echo "UV version:" && \
    uv --version && \
    echo "Installed packages:" && \
    uv pip list && \
    echo "Verifying Python version:" && \
    python -c "import sys; ver = sys.version_info; print(f'Python {ver.major}.{ver.minor} detected'); assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    echo "Verifying PyTorch version:" && \
    python -c "import torch; print(f'PyTorch {torch.__version__} detected'); assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__" && \
    echo "============= Environment Verification Complete ============="
