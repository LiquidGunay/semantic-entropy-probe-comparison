# Semantic Entropy Probes (Qwen + uncertainty signals)

This repo runs and inspects uncertainty signals on math/OOD questions:
- Multi-sample generations from Qwen models (vLLM preferred; HF fallback).
- Semantic entropy labels (NLI or cosine).
- Hidden-state tracing at </think> and three probes (accuracy, SE, entropy baseline).
- Evaluation (AUROC/AUPRC, p-values) and an interactive marimo notebook for exploration.
- Data and artifacts are LFS-tracked (`data/`, `artifacts/`) so you can pull a full run via `git lfs pull` without regenerating.

## Install

Quickstart (from repo root):

```bash
cd sep-marimo
uv venv .venv
source .venv/bin/activate
# Analysis-only install (small): marimo + altair + sklearn + umap
uv sync

# Pipeline install (adds torch/transformers/etc.)
# GPU torch build (recommended) – adjust CUDA version if needed.
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
uv sync --extra pipeline

# Editable install is optional if you prefer:
# uv pip install -e . --extra pipeline

# Launch marimo editor (with MCP server optional)
uv run marimo edit probe_training_auto.py --mcp --no-token
```

Notes
- Defaults use GSM8K mini, K=10–20 samples, temp=0.7, top_p=0.9.
- Semantic equivalence: `deberta-v3-small-mnli` (CPU) by default; cosine fallback available.
- Global Hugging Face cache is used (no local data directory).
- GPU memory budget ~4–6GB → keep batch sizes small; NLI on CPU to save VRAM.

## What’s inside
- `scripts/01_generate_with_vllm.py`: sample K runs per question (vLLM or HF fallback).
- `scripts/02_compute_semantic_entropy.py`: compute semantic entropy per math question.
- `scripts/03_trace_with_nnsight.py`: capture hidden states at </think>.
- `scripts/04_build_probe_datasets.py`: join runs + hidden + labels into NPZ splits.
- `scripts/05_train_probes.py`: train accuracy, SE (high semantic entropy), and entropy baseline probes.
- `scripts/06_eval_probes.py`: AUROC/AUPRC + permutation p-values, confidence summary.
- `scripts/07_build_analysis_dataset.py`: compact per-run parquet with probe scores, entropies, UMAP coords for the notebook.
- `notebooks/probe_analysis.py`: Altair + marimo dashboard (filters, margin/entropy scatter, fixed UMAP, capped selection details).
- `artifacts/`: hidden states, probe datasets, models, eval JSON, analysis parquet (LFS).
- `data/`: math_raw/runs/semantic entropy (LFS); add `ood_raw.jsonl` if needed.

## Qwen uncertainty pipeline (MATH-500 style)

For larger runs on Qwen3-4B (or local smoke tests with Qwen3-0.6B):

1) **Prepare data** – put `data/math_raw.jsonl` and `data/ood_raw.jsonl` with fields `{id, question, answer}`.

2) **Generate runs** (vLLM, T=0.6):
```bash
cd sep-marimo
python scripts/01_generate_with_vllm.py --model Qwen/Qwen3-0.6B-Instruct --runs 3 --limit 5
# real run: set --model Qwen/Qwen3-4B-Instruct and drop --limit
```

   *Small GPU / no vLLM?* Switch to the HF fallback: add `--backend hf` (uses transformers generate) and keep `--dtype float16 --max-new-tokens 64` for 6GB cards. Additional knobs: `--gpu-mem-utilization`, `--max-num-seqs`, `--enforce-eager`.

3) **Semantic entropy labels** (math only):
```bash
python scripts/02_compute_semantic_entropy.py
```

4) **Trace hidden states** (</think> position):
```bash
python scripts/03_trace_with_nnsight.py --model Qwen/Qwen3-0.6B-Instruct --backend hf
# use --backend nnsight to trace with nnsight if configured
```

5) **Assemble probe datasets**:
```bash
python scripts/04_build_probe_datasets.py
```

6) **Train probes**:
```bash
python scripts/05_train_probes.py
```

7) **Evaluate**:
```bash
python scripts/06_eval_probes.py
cat artifacts/models/probe_eval.json
```

Outputs land in `data/` (runs, semantic entropy) and `artifacts/` (hidden states, probe datasets, models, eval JSON). Install `vllm>=0.6.2` on a GPU host for step 2.

8) **Build compact analysis dataset + notebook**
```bash
python scripts/07_build_analysis_dataset.py  # reads artifacts, writes artifacts/analysis/analysis.parquet
uv run marimo run notebooks/probe_analysis.py --host 0.0.0.0 --port 7860
```

### Deploy & embed (Railway)
- Deployment command is provided in `Procfile`:\
  `uv run marimo run notebooks/probe_analysis.py --host 0.0.0.0 --port $PORT --no-token --allow-origins="*"`
- The app is read-only (`marimo run`) and CORS-open for iframe embedding. See `Railway.md` for setup steps.
- Example iframe snippet:
```html
<iframe
  src="https://YOUR_APP_URL"
  style="width:100%;height:900px;border:0;"
  allow="clipboard-read; clipboard-write">
</iframe>
```

## LFS and pulling data
- `data/**` and `artifacts/**` are tracked via Git LFS. After cloning, run `git lfs install --local` and `git lfs pull` to fetch the full run outputs (no regeneration needed).
- Caches (`.hf-cache/`, `.uv-cache/`) stay untracked.

### Current smoke test (HF fallback, GTX 1660 Ti)
- Synthetic 10 easy math questions, 10 runs each with `--backend hf --model Qwen/Qwen2-0.5B-Instruct --dtype float16 --max-new-tokens 32`.
- Metrics (`artifacts/models/probe_eval.json`): AUCs — accuracy probe 1.00 (p=0.0), SE probe 0.95 (p=0.0), entropy baseline 0.90 (p=0.0); mean confidence correct vs incorrect ≈ 0.997 vs 0.00023.
- For real MATH-500, prefer vLLM + larger Qwen3 checkpoints on a GPU with compute capability ≥8.0.
