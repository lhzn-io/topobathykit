# syntax=docker/dockerfile:1
# Tested on: NVIDIA Jetson AGX Orin (JetPack 6.x / l4t r36), x86_64 (Ubuntu 24.04 + CUDA 12.x),
#            and macOS (Apple Silicon / OrbStack)
# Override BASE_IMAGE at build time for your target platform:
#   Jetson Orin:       --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.2.0
#   x86_64/WSL2 GPU:   --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04  (default)
#   macOS / no GPU:    --build-arg BASE_IMAGE=ubuntu:24.04  (multi-arch; pulls arm64 on Apple Silicon)
ARG BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${BASE_IMAGE}

WORKDIR /app

# Minimal apt: only curl + bzip2 to bootstrap micromamba.
# All geo libs (GDAL/NetCDF4 via rasterio manylinux wheel, PDAL via conda-forge),
# Python, cmake, and ninja come from conda-forge or pip — no universe repo needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Alias python to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install micromamba
RUN curl -Ls "https://micro.mamba.pm/api/micromamba/$(uname -m | sed 's/x86_64/linux-64/;s/aarch64/linux-aarch64/;s/arm64/linux-aarch64/')/latest" | \
    tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# Create the Python environment via micromamba. Includes:
#   - python 3.12 with headers (for building C extensions)
#   - pdal + python-pdal (only available pre-built via conda-forge, not PyPI)
#   - cmake + ninja (build tools, avoids needing apt universe)
RUN micromamba create -y -p /app/.venv -c conda-forge \
    "python=3.12" \
    "pdal>=2.7" \
    "python-pdal" \
    "cmake" \
    "ninja" && \
    micromamba clean -q --all

ENV PATH="/app/.venv/bin:$PATH"

# Pre-install build requirements to avoid build-isolation issues (like NumPy 2.0)
RUN pip install --upgrade pip setuptools wheel
RUN pip install Cython hatchling scikit-build py-cpuinfo setuptools_scm

# Copy Library and install it
# We assume the build context is the root of topobathykit/
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-build-isolation .

# Copy Service dependencies and install service
COPY service ./service
RUN pip install ./service --no-deps && pip install fastapi uvicorn[standard] pydantic-settings fastapi-cache2[redis] click

# Default port 9595
EXPOSE 9595

# Set pythonpath so it finds topobathyserve
ENV PYTHONPATH="/app:/app/src:/app/service"

# Fix ownership so non-root users can write to logs
ARG UID=1000
ARG GID=1000
RUN chown -R ${UID}:${GID} /app

CMD ["uvicorn", "topobathyserve.main:app", "--host", "0.0.0.0", "--port", "9595", "--workers", "2"]
