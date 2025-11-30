# AGENTS

## Product Goal
Build a reproducible marimo notebook that implements Semantic Entropy Probes (SEP) using a small GPT-2 model with nnsight, runnable on a 4GB GPU, including sampling-based semantic entropy labeling, probe training, and evaluation.

## Roles
- **You (Reviewer/Owner):** Provide dataset choice, parameter preferences, and accept trade-offs; run notebook, review outputs, and prioritize future extensions.
- **Codex (Implementer):** Design and code notebook, enforce marimo/reactive constraints, integrate nnsight, ensure lightweight compute, add documentation and plots.

## Success Criteria
- Notebook runs end-to-end on provided hardware within reasonable time (≤30–40 min for small test set).
- Semantic entropy labels computed via NLI or cosine clustering; lexical entropy baseline included.
- Probe trained on captured hidden states; evaluation metrics/plots available.
- Clear instructions for setup and reruns; caches avoid unnecessary recompute.

## Non-Goals (for now)
- Exact reproduction of full-scale results from the paper.
- Large-model experiments (GPT-2 XL+) or extensive hyperparameter sweeps.
- Production-grade packaging; focus remains on exploratory, readable notebook.
