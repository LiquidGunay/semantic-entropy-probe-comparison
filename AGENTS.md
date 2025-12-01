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
- Marimo notebooks render charts and selections correctly (see rules below).

## Non-Goals (for now)
- Exact reproduction of full-scale results from the paper.
- Large-model experiments (GPT-2 XL+) or extensive hyperparameter sweeps.
- Production-grade packaging; focus remains on exploratory, readable notebook.

## Marimo rules to follow
- Always display charts/outputs by returning or exposing the UI element as the last expression in the cell (e.g., `chart = mo.ui.altair_chart(...); mo.vstack([... , chart])`).
- Avoid `return` statements inside branches; marimo cells treat top-level code as reactive scripts. If you need to share values, return once at the end of the cell.
- Keep variable names unique per cell (or prefix with `_`) to satisfy `marimo check` and avoid cross-cell collisions.
- Use `mo.ui.altair_chart` with `chart_selection` (`"interval"` or `"point"`) to capture selections; selected rows are available via `chart.value` (typically a list/Series of the encoded key).
- When showing both title and chart, stack them (`mo.vstack([title_md, chart])`); don’t rely on bare `mo.ui.altair_chart` calls with no expression.
- Provide a safe empty-state path: initialize placeholders, show `mo.alert` when data is missing, and still return the widget (or `None`) so downstream cells don’t break.
- If combining manual selections with chart selections, merge and de-duplicate before slicing details; cap detail rendering to ~10 rows to keep the UI responsive.
