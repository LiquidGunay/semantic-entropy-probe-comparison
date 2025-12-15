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

MARIMO_MODE="${MARIMO_MODE:-run}"
MARIMO_NOTEBOOK="${MARIMO_NOTEBOOK:-notebooks/probe_analysis.py}"
export MARIMO_NO_SHM="${MARIMO_NO_SHM:-1}"
export JOBLIB_MULTIPROCESSING="${JOBLIB_MULTIPROCESSING:-0}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PORT="${PORT:-6780}"
ALLOW_ORIGINS="${ALLOW_ORIGINS:-*}"
ANALYSIS_PARQUET="${ANALYSIS_PARQUET:-artifacts_clean/analysis/analysis.parquet}"
METRICS_JSON="${METRICS_JSON:-artifacts_clean/models/probe_eval.json}"
# Keep all temp/caches out of /dev/shm and within a writable directory.
APP_TMP="${APP_TMP:-/tmp/sep-marimo}"
mkdir -p "${APP_TMP}"
export TMPDIR="${TMPDIR:-${APP_TMP}}"
export MARIMO_TMPDIR="${MARIMO_TMPDIR:-${APP_TMP}}"
export ARROW_TMPDIR="${ARROW_TMPDIR:-${APP_TMP}}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${APP_TMP}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${APP_TMP}/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# If we can bind-mount a fake shm into a writable path, point Python there.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${APP_TMP}/mpl}"
mkdir -p "${MPLCONFIGDIR}"

# Use a private XDG cache/data to stay inside the volume.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${APP_TMP}/xdg-cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${APP_TMP}/xdg-data}"

# Helpful warnings so deploy logs surface missing assets instead of silently showing an empty UI.
[ -f "$ANALYSIS_PARQUET" ] || echo "Warning: analysis parquet missing at $ANALYSIS_PARQUET"
[ -f "$METRICS_JSON" ] || echo "Warning: metrics JSON missing at $METRICS_JSON"

echo "Starting marimo (${MARIMO_MODE}) on port ${PORT} (origins=${ALLOW_ORIGINS})"
exec marimo "${MARIMO_MODE}" "${MARIMO_NOTEBOOK}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --no-token \
  --allow-origins="${ALLOW_ORIGINS}" \
  --no-skew-protection
