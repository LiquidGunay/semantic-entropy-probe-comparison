# Deploying the marimo app on Railway (iframe-ready)

## What this serves
- Read-only marimo app: `notebooks/probe_analysis.py`
- Uses cleaned data artifacts: `artifacts_clean/analysis/analysis.parquet` and `artifacts_clean/models/probe_eval.json`.
- CORS is open (`--allow-origins="*"`) and tokens are disabled for iframe embedding.

## One-time setup on Railway
1. Create a new Railway project and connect this repo.
2. In **Variables**, add `PORT` (Railway auto-provides). No other env vars required.
3. In **Deployments**, ensure the start command uses the `Procfile` (Railway will pick up `web: ...`).
4. Make sure the repo includes cleaned artifacts (`artifacts_clean/**`) in the image. If the build context excludes LFS, run `git lfs pull` before pushing or vendor the files another way.

## Build image (Dockerfile, uv via pip)
- The `Dockerfile` uses `python:3.12-slim`, installs `uv` via pip, runs `uv sync --locked --no-dev`, and serves the marimo app.
- Command uses port 6780 and disables skew protection (`--no-skew-protection`) for iframe/client cache friendliness. Cache is set to `/tmp/.uv-cache` (`UV_CACHE_DIR`, `UV_LINK_MODE=copy`) for writable installs on Railway.
- Railway: choose **Deploy from repo → Dockerfile**; no extra env needed unless your platform forces `$PORT`.
- No GPU dependencies required.

## Local test (matches Railway)
```
uv run marimo run notebooks/probe_analysis.py \
  --host 0.0.0.0 --port 7860 --no-token --allow-origins="*"
```
Visit http://localhost:7860.

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
