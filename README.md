# Semantic Entropy Probes (marimo + nnsight)

Quickstart (from repo root):

```bash
cd sep-marimo
uv venv .venv
source .venv/bin/activate
# Install GPU torch build (recommended). Adjust CUDA version if needed.
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
uv pip install -e .
# Launch marimo editor (with MCP server optional)
uv run marimo edit probe_training_auto.py --mcp --no-token
```

Notes
- Defaults use GSM8K mini, K=10–20 samples, temp=0.7, top_p=0.9.
- Semantic equivalence: `deberta-v3-small-mnli` (CPU) by default; cosine fallback available.
- Global Hugging Face cache is used (no local data directory).
- GPU memory budget ~4–6GB → keep batch sizes small; NLI on CPU to save VRAM.

## Qwen Uncertainty Pipeline (MATH-500 style)

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

### Current smoke test (HF fallback, GTX 1660 Ti)
- Synthetic 10 easy math questions, 10 runs each with `--backend hf --model Qwen/Qwen2-0.5B-Instruct --dtype float16 --max-new-tokens 32`.
- Metrics (`artifacts/models/probe_eval.json`): AUCs — accuracy probe 1.00 (p=0.0), SE probe 0.95 (p=0.0), entropy baseline 0.90 (p=0.0); mean confidence correct vs incorrect ≈ 0.997 vs 0.00023.
- For real MATH-500, prefer vLLM + larger Qwen3 checkpoints on a GPU with compute capability ≥8.0.
