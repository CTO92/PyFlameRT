# PyFlameRT Model Server Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Build
# =============================================================================
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python build dependencies
RUN pip3 install --no-cache-dir pybind11

# Set working directory
WORKDIR /build

# Copy source code
COPY . .

# Build PyFlameRT
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYFLAME_RT_BUILD_TESTS=OFF \
        -DPYFLAME_RT_BUILD_PYTHON=ON \
        -DPYFLAME_RT_BUILD_SERVING=ON && \
    cmake --build . --config Release -j$(nproc)

# Install Python package
RUN pip3 install --no-cache-dir -e .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 pyflame && \
    mkdir -p /models /cache && \
    chown -R pyflame:pyflame /models /cache

# Copy built artifacts
COPY --from=builder /build/build/lib*.so /usr/local/lib/
COPY --from=builder /build/python/pyflame_rt /usr/local/lib/python3/dist-packages/pyflame_rt
COPY --from=builder /usr/local/lib/python3/dist-packages/pyflame_rt*.egg-info /usr/local/lib/python3/dist-packages/

# Install Python runtime dependencies
RUN pip3 install --no-cache-dir numpy aiohttp

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:$PYTHONPATH

# Switch to non-root user
USER pyflame
WORKDIR /home/pyflame

# Default configuration
ENV HTTP_HOST=0.0.0.0
ENV HTTP_PORT=8080
ENV GRPC_PORT=9090
ENV METRICS_PORT=9091
ENV MODEL_DIR=/models
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8080 9090 9091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health/live')" || exit 1

# Entry point script
COPY --chown=pyflame:pyflame deploy/docker/entrypoint.py /home/pyflame/entrypoint.py

ENTRYPOINT ["python3", "/home/pyflame/entrypoint.py"]
