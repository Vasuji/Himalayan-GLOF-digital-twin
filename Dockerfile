# Production-tier image: CUDA + PhysicsNeMo + Earth-2.
# Built on the CUDA devel image because torch-scatter compiles against torch and
# needs nvcc and a C++ toolchain present at install time.
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # physicsnemo.models triggers warp.init(), which needs a writable kernel cache.
    WARP_CACHE_PATH=/workspace/.warp_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git build-essential \
        libgl1 libglib2.0-0 libeccodes0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

WORKDIR /workspace

# torch from the CUDA index first, so pip cannot resolve a CPU-only wheel while
# satisfying PhysicsNeMo's torch>=2.10 requirement.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt requirements-nvidia.txt ./
RUN python -m pip install -r requirements-nvidia.txt

COPY pyproject.toml README.md ./
COPY glof_pipeline ./glof_pipeline
COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
COPY docs ./docs
RUN python -m pip install -e . && mkdir -p "$WARP_CACHE_PATH" data outputs checkpoints

# Fails the build if the NVIDIA stack cannot be imported, rather than at run time
# on a GPU instance that is already costing money.
RUN python -c "import physicsnemo, earth2studio; print(physicsnemo.__version__, earth2studio.__version__)"

ENTRYPOINT ["python", "-m", "glof_pipeline"]
CMD ["info"]
