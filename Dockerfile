# Python base with uv installed via pip (ghcr tag unavailable on Railway builder)
FROM python:3.12-slim

WORKDIR /app

# System deps for build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy lockfiles first for caching
COPY pyproject.toml uv.lock ./

# Install project dependencies (no dev)
RUN uv sync --locked --no-dev

# Copy the rest
COPY . .

EXPOSE 8080
ENV PORT=8080

CMD ["sh", "-c", "uv run marimo run notebooks/probe_analysis.py --host 0.0.0.0 --port ${PORT:-8080} --no-token --allow-origins=*"]
