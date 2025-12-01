# uv-based image for Railway / container deploys
FROM ghcr.io/astral-sh/uv:python3.12-slim

WORKDIR /app

# Copy lockfiles first for layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies (no dev)
RUN uv sync --locked --no-dev

# Copy the rest of the repo
COPY . .

# Expose port (Railway supplies $PORT)
EXPOSE 8080

# Use $PORT if provided (Railway); default to 8080 locally
ENV PORT=8080

CMD ["sh", "-c", "uv run marimo run notebooks/probe_analysis.py --host 0.0.0.0 --port ${PORT:-8080} --no-token --allow-origins=*" ]
