#!/usr/bin/env bash
set -euo pipefail

# Prefer the project venv if present so the same command works locally and in Docker.
if [ -d "/app/.venv/bin" ]; then
  export PATH="/app/.venv/bin:$PATH"
elif [ -d ".venv/bin" ]; then
  export PATH="$(pwd)/.venv/bin:$PATH"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -d "${VIRTUAL_ENV}/bin" ]; then
  export PATH="${VIRTUAL_ENV}/bin:$PATH"
fi

PORT="${PORT:-6780}"
ALLOW_ORIGINS="${ALLOW_ORIGINS:-*}"
ANALYSIS_PARQUET="${ANALYSIS_PARQUET:-artifacts_clean/analysis/analysis.parquet}"
METRICS_JSON="${METRICS_JSON:-artifacts_clean/models/probe_eval.json}"
# Use an app-local tmpdir to avoid small /tmp or shm limits on hosts like Railway.
APP_TMP="${APP_TMP:-/app/tmp}"
mkdir -p "${APP_TMP}"
export TMPDIR="${TMPDIR:-${APP_TMP}}"
export MARIMO_TMPDIR="${MARIMO_TMPDIR:-${APP_TMP}}"
export ARROW_TMPDIR="${ARROW_TMPDIR:-${APP_TMP}}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${APP_TMP}}"

# Helpful warnings so deploy logs surface missing assets instead of silently showing an empty UI.
[ -f "$ANALYSIS_PARQUET" ] || echo "Warning: analysis parquet missing at $ANALYSIS_PARQUET"
[ -f "$METRICS_JSON" ] || echo "Warning: metrics JSON missing at $METRICS_JSON"

echo "Starting marimo on port ${PORT} (origins=${ALLOW_ORIGINS})"
exec marimo run notebooks/probe_analysis.py \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --no-token \
  --allow-origins="${ALLOW_ORIGINS}" \
  --no-skew-protection
