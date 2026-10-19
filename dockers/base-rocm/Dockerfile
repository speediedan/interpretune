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
#
# ROCm twin of dockers/base-cuda (#703). Same OS, Python, venv path and permissions contract, so dockers/it-az-base
# builds on it unchanged with CUST_BASE=rocm<version>-. The torch wheel is PyTorch's own ROCm build of the exact
# PYTORCH_VERSION, from download.pytorch.org/whl/rocm<major.minor>, never AMD's rocm/pytorch image (whose torch and
# Python versions are its own) and never a default-index torch, which would silently be a CUDA build.
#
# DRAFT: no image built from this file has run on AMD hardware. Choosing the ROCm version that build B's cards need,
# recording the image digest and proving gfx code objects plus two-device visibility are hardware-time steps.

# torch 2.13.0 publishes rocm7.1 and rocm7.2 wheels for cp313 (read from download.pytorch.org, 2026-10-19).
ARG ROCM_VERSION=7.2
ARG OS_VER=ubuntu-22.04

FROM rocm/dev-${OS_VER}:${ROCM_VERSION}

ARG ROCM_VERSION
ARG PYTHON_VERSION=3.13
ARG PYTORCH_VERSION=2.13.0
ARG MKL_THREADING_LAYER=GNU
# gfx targets for source-built extensions only (the torch wheel ships its own code objects). Empty means the
# toolchain default; set it to build B's targets once the cards are known.
ARG PYTORCH_ROCM_ARCH=""

SHELL ["/bin/bash", "-c"]
ENV \
    DEBIAN_FRONTEND=noninteractive \
    TZ=US/Pacific \
    PATH="$PATH:/root/.local/bin:/opt/rocm/bin" \
    ROCM_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    MKL_THREADING_LAYER=${MKL_THREADING_LAYER} \
    MAKEFLAGS="-j2" \
    UV_NO_CACHE=1
RUN apt-get update -qq --fix-missing && \
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
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements/ ./requirements/

ENV PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages

RUN \
    umask 000 && \
    uv venv /tmp/venvs/it_dev --python python${PYTHON_VERSION} && \
    . /tmp/venvs/it_dev/bin/activate && \
    uv pip install "torch==${PYTORCH_VERSION}+rocm${ROCM_VERSION}" \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" && \
    rm -rf requirements/

ENV VIRTUAL_ENV=/tmp/venvs/it_dev
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Backend truth is torch.version.hip, not the device API (the HIP build keeps torch.cuda): a CUDA wheel slipping in
# here fails the build rather than a test run on build B. No device is needed for this check.
RUN \
    set -x && \
    . /tmp/venvs/it_dev/bin/activate && \
    uv --version && \
    uv pip list && \
    python -c "import sys; ver = sys.version_info; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; v = torch.__version__; assert v.startswith('$PYTORCH_VERSION'), v; \
assert torch.version.hip and torch.version.cuda is None, (torch.version.hip, torch.version.cuda); \
print('torch', v, 'hip', torch.version.hip)" && \
    chmod -R 777 /tmp/venvs/it_dev
