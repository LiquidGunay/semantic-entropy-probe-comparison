# uv-based image for Railway / container deploys
FROM python:3.12-slim

WORKDIR /app

# Writable cache and conservative threading for joblib/numba
ENV UV_CACHE_DIR=/tmp/.uv-cache \
    UV_LINK_MODE=copy \
    NUMBA_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    JOBLIB_TEMP_FOLDER=/tmp

# System deps for build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy lockfiles first for caching
COPY pyproject.toml uv.lock ./

# Install project dependencies (no dev), honoring the lockfile (frozen = fail if lock mismatch)
RUN uv sync --frozen --no-dev

# Use the project venv by default for runtime binaries (marimo, python, etc.)
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy the rest
COPY . .

# Ensure launcher is executable
RUN chmod +x scripts/serve_probe_analysis.sh

# Expose default port (Railway will override with $PORT)
EXPOSE 6780
ENV PORT=6780

# Shared entrypoint for local and Railway deploys
ENTRYPOINT ["./scripts/serve_probe_analysis.sh"]
