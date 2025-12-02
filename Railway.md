# Deploying the marimo app on Railway (iframe-ready)

## What this serves
- Read-only marimo app: `notebooks/probe_analysis.py`
- Uses cleaned data artifacts: `artifacts_clean/analysis/analysis.parquet` and `artifacts_clean/models/probe_eval.json`.
- CORS is open (`--allow-origins="*"`) and tokens are disabled for iframe embedding.

## One-time setup on Railway (new entrypoint)
1. Create a Railway project and connect this repo.
2. Choose **Deploy from repo → Dockerfile** (or keep Nixpacks + Procfile; both now share the same entrypoint).
3. No env vars are required; Railway injects `PORT`. Optional overrides:
   - `ALLOW_ORIGINS` (default `*`)
   - `ANALYSIS_PARQUET` (default `artifacts_clean/analysis/analysis.parquet`)
   - `METRICS_JSON` (default `artifacts_clean/models/probe_eval.json`)
4. Ensure cleaned artifacts (`artifacts_clean/**`) are present. If your clone skipped LFS, run `git lfs pull` locally before pushing so the image bakes in the files.

## Fresh deployment approach
- **Shared launcher**: `scripts/serve_probe_analysis.sh` sets up PATH for the project venv, checks that the parquet/metrics exist, and starts marimo on `$PORT` with open CORS and no token.
- **Dockerfile**: installs `uv`, runs `uv sync --frozen --no-dev`, sets `PATH` to the project `.venv`, disables joblib multiprocessing (`JOBLIB_MULTIPROCESSING=0`, `LOKY_MAX_CPU_COUNT=1`) to avoid semaphore/disk warnings, marks the launcher executable, and uses it as `ENTRYPOINT`. Listens on `$PORT` (defaults to 6780).
- **Procfile**: `web: ./scripts/serve_probe_analysis.sh` — works if you prefer Railway’s Nixpacks instead of Docker.
- Threading/caches: `UV_CACHE_DIR=/tmp/.uv-cache`, `UV_LINK_MODE=copy`, `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `JOBLIB_TEMP_FOLDER=/tmp`.

## Local test (mirrors Railway)
```
PORT=7860 ./scripts/serve_probe_analysis.sh
```
Visit http://localhost:7860. Stop with Ctrl+C. If you only want to confirm assets are found, run `ANALYSIS_PARQUET=missing ./scripts/serve_probe_analysis.sh` to see the warning.

If Railway logs show `joblib ... No space left on device`, that means the platform denies new POSIX semaphores or /dev/shm is tiny. The image already forces serial joblib; the warning should disappear after rebuilding with this Dockerfile.

## Embedding snippet
Replace `YOUR_APP_URL` with your Railway domain (e.g., `https://your-app.up.railway.app`).
```html
<iframe
  src="https://YOUR_APP_URL"
  style="width:100%;height:900px;border:0;"
  allow="clipboard-read; clipboard-write"
></iframe>
```

## Notes
- App is read-only (`marimo run`), so notebook code isn’t editable.
- If you need authentication, remove `--no-token` and handle the token in your site. For frictionless embeds, keep it tokenless and deploy behind your own auth/reverse proxy if needed.
- To regenerate cleaned data on Railway, you’d need to bring the raw runs/hidden states or precompute artifacts; current Procfile assumes artifacts already exist in the image.
