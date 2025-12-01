# Semantic Entropy Probes – Handoff Plan (analysis notebook, LFS, deps)
Date: 2025-12-04
Working dir: `/.gunayexp/semantic-entropy-probe-comparison`

## What was done in this session
- Dependency split: `pyproject.toml` now has a light default (marimo, altair, sklearn, umap, pyarrow) and a `pipeline` extra for heavy packages (torch/transformers/datasets/accelerate/sentence-transformers/nnsight/plotly/scipy). Added setuptools package discovery to include only `sep_marimo*`. `uv.lock` regenerated accordingly (with network allowed).
- New compact analysis dataset builder: `scripts/07_build_analysis_dataset.py` reads runs/hidden NPZ/probes and writes `artifacts/analysis/analysis.parquet` (per-run rows, probe margins/probs, entropies, think lengths, UMAP coords, representative flag). Ran it once; `analysis.parquet` exists (math only, 1k rows; ood_test not present).
- New marimo notebook: `notebooks/probe_analysis.py` (Altair). Features: filters (dataset, correctness, per-question representative toggle, max points), margin-vs-entropy scatter with threshold line, fixed UMAP scatter, selection capped to 10 runs with detail panel (question/think/answer/gold/probe stats), metrics display from `probe_eval.json`.
- README updated for dependency split and notebook/analysis build steps.
- LFS setup: installed `git-lfs` system-wide, added `.gitattributes` to track `data/**` and `artifacts/**`, removed `data/` and `artifacts/` from `.gitignore`, added cache ignores (`.hf-cache/`, `.uv-cache/`). `git lfs install --local` executed.
- Environment synced: `UV_CACHE_DIR=.uv-cache uv lock` (network allowed) and `uv sync --locked` succeeded; `.venv` is up to date.
- Generated artifacts for handoff: `data/` (math raw/runs/semantic entropy), `artifacts/` (hidden states, probe datasets, models, probe_eval.json, analysis parquet).

## Current state of outputs (all LFS-tracked paths)
- `data/`: math_raw.jsonl, math_runs.jsonl, math_semantic_entropy.jsonl (~576 MB total).
- `artifacts/`: hidden states NPZ, probe_datasets/*.npz, models/*.pkl + se_threshold.json, probe_eval.json, analysis/analysis.parquet (~47 MB total, plus new parquet).
- No ood data present; ood_test.npz absent; analysis builder skipped ood.

## How to pull and run on another machine
1) Ensure git-lfs is installed on the target machine, then:
   - `git clone <repo>`
   - `cd semantic-entropy-probe-comparison`
   - `git lfs install --local` (or `git lfs pull` if needed)
2) Create/activate venv and sync deps:
   - `uv venv .venv && source .venv/bin/activate`
   - Analysis-only: `uv sync --locked`
   - Full pipeline: `uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1` then `uv sync --locked --extra pipeline`
3) Analysis notebook:
   - Build dataset if needed: `python scripts/07_build_analysis_dataset.py` (already built once; output at `artifacts/analysis/analysis.parquet`)
   - Run: `uv run marimo run notebooks/probe_analysis.py --host 0.0.0.0 --port 7860` (fits iframe on Railway)
4) Pipeline scripts (if re-running): use README commands (scripts 01–06); ensure GPU/vLLM/torch as needed.

## Next steps (handoff)
- If you need OOD coverage: add `data/ood_raw.jsonl`, run scripts 01–06 for OOD, then rerun `07_build_analysis_dataset.py` to include ood_test rows; update `probe_eval.json` accordingly.
- If you plan a larger Qwen/vLLM run: adjust scripts/01 arguments (model, runs, max tokens), regenerate data/artifacts, retrain probes (scripts 05/06), rebuild analysis parquet.
- If editing LFS patterns: update `.gitattributes` before adding new large files; keep caches (.hf-cache, .uv-cache) out of Git.
- Keep using `uv sync --locked` on new hosts to stay aligned with `uv.lock`.

## Files changed/added (key)
- `.gitattributes` (LFS tracking for data/artifacts)
- `.gitignore` (allow data/artifacts; ignore caches)
- `pyproject.toml`, `uv.lock` (deps split, package discovery)
- `README.md` (install/run updates)
- `scripts/07_build_analysis_dataset.py` (new)
- `notebooks/probe_analysis.py` (new)
- `artifacts/analysis/analysis.parquet` (generated)

## Known warnings
- Running `07_build_analysis_dataset.py` on this host emitted joblib/numba shared-memory warnings (permission); it still completed. UMAP fit uses up to 5k samples; adjust flags if larger runs.
