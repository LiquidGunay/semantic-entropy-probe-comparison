import marimo

__generated_with = "0.7.13"
app = marimo.App(width="wide")


@app.cell
def __():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import json

    return alt, json, mo, np, pd, Path


@app.cell
def __(mo):
    mo.md(
        """
        # Probe Analysis (Altair + marimo)

        This notebook visualizes per-run probe outputs, entropies, and UMAP embeddings.
        Use the filters to slice the dataset, then select up to 10 runs for details.
        """
    )


@app.cell
def __(mo, Path):
    default_data = Path("artifacts/analysis/analysis.parquet")
    default_metrics = Path("artifacts/models/probe_eval.json")
    data_path = mo.ui.text(str(default_data), label="Analysis dataset (.parquet)")
    metrics_path = mo.ui.text(str(default_metrics), label="Probe metrics JSON")
    dataset_filter = mo.ui.dropdown(
        options=["all", "math", "ood"], value="all", label="Dataset"
    )
    rep_only = mo.ui.switch(value=False, label="Per-question representative only")
    correctness_filter = mo.ui.dropdown(
        options=["all", "correct", "incorrect"], value="all", label="Correctness"
    )
    max_points = mo.ui.slider(200, 10000, value=5000, step=200, label="Max points to plot")
    seed_box = mo.ui.number(value=42, label="Random seed", min=0, max=10_000, step=1)
    return correctness_filter, data_path, dataset_filter, max_points, metrics_path, rep_only, seed_box


@app.cell
def __(json, mo, np, pd, Path, correctness_filter, data_path, dataset_filter, max_points, rep_only, seed_box):
    path = Path(data_path.value)
    df = None
    load_error = None
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            load_error = str(e)
    else:
        load_error = f"Missing file: {path}"

    if df is None:
        mo.alert(load_error or "Failed to load dataset.")
        filtered = pd.DataFrame()
        selection = mo.ui.multiselect(options=[], value=[], label="Select runs (<=10)")
        return df, filtered, selection

    df["is_correct"] = df["is_correct"].astype(bool)
    df["probe_margin"] = df["probe_margin"].astype(float)
    df["mean_think_entropy"] = df["mean_think_entropy"].astype(float)
    if "semantic_entropy" not in df.columns:
        df["semantic_entropy"] = np.nan
    df["semantic_entropy"] = df["semantic_entropy"].astype(float)

    filtered = df.copy()
    if dataset_filter.value != "all":
        filtered = filtered[filtered["dataset"] == dataset_filter.value]
    if rep_only.value:
        filtered = filtered[filtered.get("is_representative", False)]
    if correctness_filter.value == "correct":
        filtered = filtered[filtered["is_correct"]]
    elif correctness_filter.value == "incorrect":
        filtered = filtered[~filtered["is_correct"]]

    if len(filtered) > max_points.value:
        filtered = filtered.sample(n=max_points.value, random_state=int(seed_box.value))

    filtered = filtered.reset_index(drop=True)
    # Options capped to keep the dropdown small
    options = filtered["run_uid"].tolist()
    selection = mo.ui.multiselect(
        options=options[:5000],
        value=options[: min(3, len(options))],
        label="Selected runs (<=10)",
    )
    return df, filtered, selection


@app.cell
def __(filtered, mo, selection):
    sel_ids = selection.value[:10]
    notice = ""
    if len(selection.value) > 10:
        notice = f"Showing first 10 of {len(selection.value)} selections."
    mo.hstack(
        [
            selection,
            mo.md(f"{notice} {len(filtered)} points in view.")
        ]
    )
    return sel_ids


@app.cell
def __(alt, filtered, mo, pd, sel_ids):
    if filtered.empty:
        mo.alert("No rows to plot.")
        chart_margin = None
    else:
        plot_df = filtered.copy()
        plot_df["selected"] = plot_df["run_uid"].isin(sel_ids)
        plot_df["question_snippet"] = plot_df["question"].str.slice(0, 120)
        plot_df["think_snippet"] = plot_df["think_text"].str.slice(0, 120)
        base = alt.Chart(plot_df).encode(
            x=alt.X("probe_margin:Q", title="Probe margin (distance to boundary)"),
            y=alt.Y("mean_think_entropy:Q", title="Mean think entropy"),
            color=alt.Color("is_correct:N", scale=alt.Scale(domain=[True, False], range=["#1b9e77", "#d95f02"]), title="Correct"),
            tooltip=[
                "run_uid",
                "question_id",
                "dataset",
                alt.Tooltip("probe_prob_correct:Q", format=".3f"),
                alt.Tooltip("mean_think_entropy:Q", format=".3f"),
                alt.Tooltip("semantic_entropy:Q", format=".3f"),
                alt.Tooltip("question_snippet:N", title="Question"),
                alt.Tooltip("think_snippet:N", title="Think"),
                alt.Tooltip("answer_text:N", title="Answer"),
            ],
            size=alt.condition("datum.selected", alt.value(120), alt.value(40)),
            opacity=alt.condition("datum.selected", alt.value(0.95), alt.value(0.6)),
        )
        points = base.mark_circle()
        boundary = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#555", strokeDash=[6, 4]).encode(x="x")
        chart_margin = (points + boundary).properties(height=320, width="container").interactive()
        mo.md("### Margin vs entropy")
        mo.ui.altair_chart(chart_margin)
    return chart_margin


@app.cell
def __(alt, filtered, mo, sel_ids):
    if filtered.empty:
        mo.alert("No UMAP coordinates to plot.")
        chart_umap = None
    else:
        plot_df = filtered.copy()
        plot_df["selected"] = plot_df["run_uid"].isin(sel_ids)
        plot_df["question_snippet"] = plot_df["question"].str.slice(0, 120)
        base = alt.Chart(plot_df).encode(
            x=alt.X("umap_x:Q", title="UMAP-1"),
            y=alt.Y("umap_y:Q", title="UMAP-2"),
            color=alt.Color("is_correct:N", scale=alt.Scale(domain=[True, False], range=["#1b9e77", "#d95f02"]), title="Correct"),
            tooltip=[
                "run_uid",
                "question_id",
                "dataset",
                alt.Tooltip("probe_prob_correct:Q", format=".3f"),
                alt.Tooltip("mean_think_entropy:Q", format=".3f"),
                alt.Tooltip("question_snippet:N", title="Question"),
                alt.Tooltip("answer_text:N", title="Answer"),
            ],
            size=alt.condition("datum.selected", alt.value(120), alt.value(40)),
            opacity=alt.condition("datum.selected", alt.value(0.95), alt.value(0.45)),
        )
        points = base.mark_circle()
        chart_umap = points.properties(height=320, width="container").interactive()
        mo.md("### UMAP (fixed embedding)")
        mo.ui.altair_chart(chart_umap)
    return chart_umap


@app.cell
def __(filtered, mo, sel_ids):
    if filtered.empty:
        mo.alert("No details to show.")
        details = []
    else:
        selected_df = filtered[filtered["run_uid"].isin(sel_ids)].head(10)
        rows = []
        for _, row in selected_df.iterrows():
            rows.append(
                mo.md(
                    f"""**{row.run_uid}** | Dataset: {row.dataset} | QID: {row.question_id} | Correct: {row.is_correct} | Probe prob: {row.probe_prob_correct:.3f} | Margin: {row.probe_margin:.3f}

**Question:** {row.question}

**Think:** {row.think_text}

**Answer (model):** {row.answer_text}

**Gold:** {row.gold_answer}

Entropy (think): {row.mean_think_entropy:.3f} | Semantic entropy: {row.semantic_entropy}
"""
                )
            )
        details = mo.vstack(rows) if rows else mo.md("No selections yet.")
        mo.md("### Selected runs (max 10)")
        details
    return details


@app.cell
def __(json, metrics_path, mo, Path):
    metrics_file = Path(metrics_path.value)
    if not metrics_file.exists():
        mo.alert(f"Metrics file not found: {metrics_file}")
        metrics = {}
    else:
        with metrics_file.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    cards = []
    for split, vals in metrics.items():
        cards.append(
            mo.md(
                f"""**{split}**  
AUROC (acc): {vals.get("auc_accuracy_probe", "n/a")}  
AUROC (SE): {vals.get("auc_se_probe", "n/a")}  
AUROC (entropy): {vals.get("auc_entropy_baseline", "n/a")}  
"""
            )
        )
    mo.md("### Probe metrics")
    mo.hstack(cards) if cards else mo.md("No metrics loaded.")
    return metrics


if __name__ == "__main__":
    app.run()
