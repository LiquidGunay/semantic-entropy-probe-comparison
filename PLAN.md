# Semantic Entropy Probes – Implementation Plan (marimo + nnsight)
Date: 2025-11-23
Working dir: `sep-marimo/` (new uv environment)

## Objective
Recreate the Semantic Entropy Probes (SEP) method from *"Semantic Entropy: Reliable Uncertainty from Language Models"* (arXiv:2406.15927) using a small GPT-2 model and nnsight inside a marimo notebook. Deliver a reproducible, readable data-science-style notebook (`probe_training_auto.py`) plus lightweight scripts/config needed to run and test on a 4GB VRAM GPU.

## Constraints & Resources
- GPU: 4 GB VRAM → use `gpt2` (124M) or `gpt2-medium` only if fits; inference batch sizes kept small.
- Time/compute: prioritize small evaluation sets (e.g., 200–500 prompts) and low sampling counts (K≈10–20).
- Environment: new `uv` virtual env inside this directory. No OOP unless necessary; minimal `try/except`.
- Notebook: marimo reactive model → avoid mutating variables across cells; prefer `mo.state`/`mo.storage` for shared artifacts; deterministic cell outputs.

## Method (paper → simplified implementation)
1. **Semantic Entropy labels**
   - For each prompt, sample K completions from GPT-2 with temperature/top-p.
   - Compute semantic equivalence between pairs of completions using a lightweight NLI/STS model (e.g., `MoritzLaurer/deberta-v3-small-mnli`) or sentence-embedding cosine with clustering as a fallback.
   - Form equivalence clusters; derive probability mass over clusters from sample counts (or soft entailment scores) and compute entropy → `SE(p)` label for the prompt.
2. **Probe target**
   - Use nnsight to capture hidden state(s) for the base prompt: final token representation or pooled hidden state before generation.
   - Train a small regression head (linear or 2-layer MLP) on frozen hidden states to predict SE.
3. **Evaluation**
   - Compare probe predictions against true SE: RMSE/MAE, Spearman; correlate with misclassification/error flags (via NLI correctness vs reference answer where available).
   - Baseline: lexical entropy from logprobs to contrast with semantic entropy.
4. **Datasets (lightweight)**
   - GSM8K mini subset (HF `gsm8k`, train split, cap ~200-500 items) as primary; keep option for custom toy list if runtime exceeds budget.

## Deliverables
- `probe_training_auto.py` marimo notebook with:
  - Setup & installs (uv/uvx usable via CLI; inside notebook use `import uv` not required if already installed via shell).
  - Config cell (model names, sampling K, dataset cap, device selection, seeds).
  - Data loading cell (small subset caching to disk).
  - Sampling cell (multi-sample generation → completions cache to avoid reruns).
  - Semantic equivalence + entropy cell (pairwise NLI or embedding-based clustering).
  - Probe dataset creation (hidden state capture with nnsight, label pairing).
  - Training cell (simple PyTorch training loop; early stop; logging).
  - Evaluation/plots cell (scatter, calibration, reliability plots if feasible with matplotlib/plotly).
  - Notes on limitations and how to scale to larger models/experiments.
- `AGENTS.md` defining roles/goals for this project.
- Helper scripts (optional): `scripts/bootstrap.sh`, `requirements.lock` (via uv), maybe `cache/` ignore entries.

## Execution Plan (tasks)
1. Bootstrap project
   - Init uv env (`uv venv`), add deps (marimo, nnsight, torch/transformers, datasets, accelerate, scikit-learn, sentence-transformers or MNLI model, matplotlib/plotly).
   - Write `README.md` stub with run commands; ensure `.gitignore` covers caches and HF data.
2. Develop notebook
   - Prototype core functions in small Python modules for reuse, but keep main flow in notebook cells; ensure marimo-safe state handling.
   - Implement sampling + caching; expose sliders for K/temperature.
   - Implement semantic equivalence via NLI; fallback cosine clustering if NLI too slow.
   - Compute semantic entropy and lexical entropy baselines.
   - Capture hidden states with nnsight hooks; store tensors on disk to avoid recompute.
   - Train probe; visualize results.
3. Validation
   - Run smoke test on 20–50 prompts; ensure GPU memory <4GB (monitor via `torch.cuda.memory_allocated`).
   - Save sample outputs and metrics to `outputs/`.
4. Documentation
   - Inline markdown explanations inside notebook; update `AGENTS.md` with goals and responsibilities; keep PLAN updated if scope shifts.

## Risks / Mitigations
- GPU OOM → use CPU for NLI; keep batch=1 for GPT-2; gradient accumulation off.
- marimo state pitfalls → no in-place mutation of top-level vars; encapsulate state in functions or `mo.store` objects.
- NLI runtime → cap prompts and K; allow switching to embedding cosine to approximate clusters.
- Reproducibility → set seeds; cache generated samples and computed entropies.

## Locked Config (per user)
- Dataset: GSM8K mini subset.
- Semantic equivalence scorer: `deberta-v3-small-mnli` (CPU) default; cosine-embedding fallback available.
- Sampling defaults: K=10–20, temperature=0.7, top_p=0.9.
- HF cache: use global cache (no local data/ subdir unless needed for exports).
- Plots: interactive (Plotly or Altair) acceptable.

## Next Steps
- Set up uv env and scaffolding (README stub, .gitignore, uv dependencies).
- Scaffold marimo notebook with config/setup cells.
