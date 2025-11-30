# Semantic Entropy Probes – Updated Plan (scripts + marimo)
Date: 2025-11-30
Working dir: `sep-marimo/`

## Objective
Run an uncertainty-signal comparison pipeline on math/OOD datasets with Qwen models: generate multi-run samples (vLLM preferred, HF fallback), compute semantic entropy, trace hidden states at </think>, build probe datasets, train logistic probes (accuracy, SE, entropy baseline), and evaluate with AUROC/AUPRC + bootstrap p-values. Marimo notebook remains for interactive exploration.

## Current State
- Scripts 01–06 implemented; HF fallback tested on a 6GB GPU (Qwen2-0.5B) with synthetic math; vLLM path ready for larger GPUs.
- Config centralized in `sep_marimo/config.py`; utilities for prompts, answer parsing, entropy, correctness checks.
- README documents run commands for both backends and summarizes latest smoke metrics.

## Near-Term Tasks
1) Big-GPU run
   - Use `--backend vllm` with Qwen3-4B (or larger) on MATH-500 and an OOD set (e.g., TriviaQA subset).
   - Increase runs per question (e.g., 10) and max_new_tokens (up to 2048).
2) OOD integration
   - Add `data/ood_raw.jsonl`, generate runs, trace, and include in `ood_test.npz` for evaluation.
3) Metrics & analysis
   - Re-evaluate probes on full math test + OOD; capture p-values, confidence summaries; store under `artifacts/models/probe_eval.json`.
4) Notebook touch-up (optional)
   - Reflect script defaults/backends; add a small cell to load probe_eval.json for quick plots.

## Risks / Mitigations
- Small/old GPUs: use `--backend hf`, `--dtype float16`, low `max_new_tokens`; fallbacks already in place.
- vLLM build deps: ensure Python dev headers & CUDA ≥ cc 8.0 for flash attention; otherwise use eager fallback flags.
- Class imbalance on tiny splits: dataset builder now borrows a sample to ensure two classes in val/test; still prefer larger datasets for stable metrics.
