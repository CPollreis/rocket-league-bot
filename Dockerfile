# syntax=docker/dockerfile:1

# MVP training + sim image for rocket-league-bot.
#
# Base: the official PyTorch runtime image. torch + CUDA + cuDNN are already
# installed and version-matched, which is the single biggest source of GPU
# setup pain. RocketSim (the environment) is CPU physics and runs fine in
# here; only the RLViser *window* does not (it needs a host display, see
# "Watching a containerized run" in the README).
#
# Pick a tag whose torch matches requirements.txt from:
#   https://hub.docker.com/r/pytorch/pytorch/tags
ARG PYTORCH_TAG=2.13.0-cuda12.6-cudnn9-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1

# Native build deps. rocketsim / rlviser-py build through CMake (cmeel) and
# rlgym-learn's pyany_serde is a Rust (maturin) extension. Only used when pip
# has no prebuilt wheel for linux/amd64.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain, for source builds of rlgym-learn when no wheel is available.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install the RL stack but let the base image own torch: its build is matched to
# the bundled CUDA, and reinstalling from requirements.txt could downgrade or
# swap that build.
COPY requirements.txt .
RUN grep -v '^torch==' requirements.txt > /tmp/reqs.docker.txt \
    && pip install --upgrade pip \
    && pip install -r /tmp/reqs.docker.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Fallback copy for `docker run` with no bind mount. During normal use the
# working tree is mounted over this at run time (see README).
COPY . .

# Override with `speed_test.py` (throughput benchmark) as needed.
CMD ["python", "quick_start.py"]
