# AGENTS

## Product Goal
Run and compare uncertainty signals (accuracy probe, semantic-entropy probe, entropy baseline) for Qwen models on math/OOD data using the scripted pipeline (vLLM or HF fallback) plus optional marimo notebook exploration.

## Roles
- **You (Owner):** Point the pipeline at target checkpoints/datasets, decide run sizes, and push to larger machines; review probe results.
- **Codex (Implementer):** Maintain scripts/notebook, ensure generation/tracing/probing run end-to-end, keep fallbacks for small GPUs, and document run recipes.

## Success Criteria
- Scripts 01–06 run end-to-end on target hardware (vLLM on big GPU; HF fallback on small GPU) producing runs, hidden states, probe datasets, trained probes, and eval JSON.
- Semantic entropy labels computed per question and used for SE probe thresholding.
- Probes report AUROC/AUPRC with bootstrap p-values; README documents how to reproduce.

## Non-Goals (for now)
- Exact reproduction of full-scale results from the paper.
- Large-model experiments (GPT-2 XL+) or extensive hyperparameter sweeps.
- Production-grade packaging; focus remains on exploratory, readable notebook.
