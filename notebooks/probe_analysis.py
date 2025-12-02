import marimo

__generated_with = "0.18.1"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    import json
    import os
    from pathlib import Path
    from sklearn.metrics import roc_curve
    return Path, alt, json, mo, os, pd, roc_curve


@app.cell
def _(mo):
    mo.md("""
    # Probe Analysis (Altair + marimo)
    Visualize probe scores, entropies, and UMAP embeddings. Use the controls to filter; charts support interval selection.
    """)
    return


@app.cell
def _(Path, json, mo, os, pd):
    data_path = Path(os.getenv("ANALYSIS_PARQUET", "artifacts_clean/analysis/analysis.parquet"))
    metrics_path = Path(os.getenv("METRICS_JSON", "artifacts_clean/models/probe_eval.json"))
    if data_path.exists():
        df_all = pd.read_parquet(data_path)
        data_notice = mo.md(f"Loaded analysis dataset: `{data_path}`")
    else:
        df_all = pd.DataFrame()
        data_notice = mo.alert(f"Analysis dataset not found: {data_path}. Set ANALYSIS_PARQUET or include the parquet in the image.")

    controls = dict(
        dataset_filter=mo.ui.dropdown(options=["all", "math", "ood"], value="all", label="Dataset"),
        rep_only=mo.ui.switch(value=False, label="Per-question representative only"),
        correctness_filter=mo.ui.dropdown(options=["all", "correct", "incorrect"], value="all", label="Correctness"),
        max_points=mo.ui.slider(200, 10000, value=5000, step=200, label="Max points to plot"),
        seed_box=mo.ui.number(start=0, stop=10_000, step=1, value=42, label="Random seed"),
        metrics_path=metrics_path,
    )
    mo.vstack([
        data_notice,
        mo.hstack(
            [
                controls["dataset_filter"],
                controls["rep_only"],
                controls["correctness_filter"],
                controls["max_points"],
                controls["seed_box"],
            ]
        )
    ])
    return controls, df_all, metrics_path


@app.cell
def _(df_all, mo):
    mo.ui.dataframe(df_all)
    return


@app.cell
def _(controls, df_all, mo):
    df_filt = df_all.copy()
    if controls["dataset_filter"].value != "all":
        df_filt = df_filt[df_filt["dataset"] == controls["dataset_filter"].value]
    if controls["rep_only"].value:
        df_filt = df_filt[df_filt.get("is_representative", False)]
    if controls["correctness_filter"].value == "correct":
        df_filt = df_filt[df_filt["is_correct"]]
    elif controls["correctness_filter"].value == "incorrect":
        df_filt = df_filt[~df_filt["is_correct"]]

    if len(df_filt) > controls["max_points"].value:
        df_filt = df_filt.sample(n=int(controls["max_points"].value), random_state=int(controls["seed_box"].value))

    df_filt = df_filt.reset_index(drop=True)
    selection_widget = mo.ui.multiselect(
        options=df_filt["run_uid"].tolist()[:5000],
        value=df_filt["run_uid"].tolist()[: min(3, len(df_filt))],
        label="Selected runs (<=10)",
    )
    mo.hstack(
        [
            selection_widget,
            mo.md(
                f"{len(df_filt)} rows | correct={df_filt['is_correct'].sum()} / incorrect={len(df_filt)-df_filt['is_correct'].sum()}"
            ),
        ]
    )
    return df_filt, selection_widget


@app.cell
def _(alt, df_filt, mo, pd, selection_widget):
    sel_ids = selection_widget.value[:10]
    if df_filt.empty:
        margin_chart = None
        view_margin = mo.alert("No rows to plot")
    else:
        zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#555", strokeDash=[6, 4]).encode(x="x:Q")
        margin_scatter = (
            alt.Chart(df_filt)
            .mark_circle()
            .encode(
                x=alt.X("probe_margin:Q", title="Probe margin"),
                y=alt.Y("mean_think_entropy:Q", title="Mean think entropy"),
                color=alt.Color("is_correct:N", scale=alt.Scale(domain=[True, False], range=["#1b9e77", "#d95f02"])),
                tooltip=["run_uid", "question_id", "probe_margin", "mean_think_entropy", "is_correct"],
                opacity=alt.condition(alt.FieldOneOfPredicate(field="run_uid", oneOf=sel_ids), alt.value(0.95), alt.value(0.5)),
                size=alt.condition(alt.FieldOneOfPredicate(field="run_uid", oneOf=sel_ids), alt.value(120), alt.value(40)),
            )
            .properties(height=280, width="container")
        )
        margin_chart = mo.ui.altair_chart(margin_scatter + zero_rule, chart_selection="interval")
        view_margin = mo.vstack([mo.md("### Margin vs entropy (probe)"), margin_chart])
    view_margin
    return margin_chart, sel_ids


@app.cell
def _(alt, df_filt, mo, sel_ids):
    umap_widgets = []
    if df_filt.empty:
        view_umap = mo.alert("No UMAP coordinates to plot")
    else:
        specs = [
            ("probe_margin", "Probe margin"),
            ("se_probe_margin", "SE probe margin"),
            ("entropy_baseline_margin", "Entropy baseline margin"),
        ]
        for col, title in specs:
            chart_spec = (
                alt.Chart(df_filt)
                .mark_circle()
                .encode(
                    x=alt.X("umap_x:Q", title="UMAP-1"),
                    y=alt.Y("umap_y:Q", title="UMAP-2"),
                    color=alt.Color(col + ":Q", title=title),
                    tooltip=["run_uid", "question_id", col, "is_correct"],
                    opacity=alt.condition(alt.FieldOneOfPredicate(field="run_uid", oneOf=sel_ids), alt.value(0.95), alt.value(0.5)),
                    size=alt.condition(alt.FieldOneOfPredicate(field="run_uid", oneOf=sel_ids), alt.value(120), alt.value(40)),
                )
                .properties(height=300, width=300)
            )
            widget = mo.ui.altair_chart(chart_spec, chart_selection="interval")
            umap_widgets.append(mo.vstack([mo.md(f"### UMAP colored by {title}"), widget]))
        view_umap = mo.hstack(umap_widgets)
    view_umap
    return (umap_widgets,)


@app.cell
def _(alt, df_filt, mo):
    if df_filt.empty:
        se_scatter_widget = None
        view_se = mo.alert("No data to plot SE vs margin")
    else:
        se_scatter = (
            alt.Chart(df_filt)
            .mark_circle()
            .encode(
                x=alt.X("semantic_entropy:Q", title="Semantic entropy"),
                y=alt.Y("se_probe_margin:Q", title="SE probe margin"),
                color="is_correct:N",
                tooltip=["run_uid", "question_id", "semantic_entropy", "se_probe_margin", "is_correct"],
            )
            .properties(height=280, width="container")
        )
        se_scatter_widget = mo.ui.altair_chart(se_scatter, chart_selection="interval")
        view_se = mo.vstack([mo.md("### SE vs SE-probe margin"), se_scatter_widget])
    view_se
    return (se_scatter_widget,)


@app.cell
def _(alt, df_filt, mo, pd, roc_curve):
    if df_filt.empty:
        view_roc = mo.alert("No rows for AUC plot")
    else:
        labels = df_filt["is_correct"].astype(int)
        curves = []
        for name, scores in {
            "probe": df_filt["probe_prob_correct"],
            "se_probe": df_filt["se_probe_prob_high"],
            "entropy": df_filt["entropy_baseline_prob"],
        }.items():
            fpr, tpr, _ = roc_curve(labels, scores)
            curves.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "probe": name}))
        roc_df = pd.concat(curves, ignore_index=True)
        roc_chart = (
            alt.Chart(roc_df)
            .mark_line()
            .encode(x="fpr:Q", y="tpr:Q", color="probe:N")
            .properties(height=240, width="container")
        )
        view_roc = mo.vstack([mo.md("### ROC curves (current filter)"), mo.ui.altair_chart(roc_chart)])
    view_roc
    return


@app.cell
def _(
    df_filt,
    mo,
    selection_widget,
):
    selected_ids = list(selection_widget.value)[:10]

    if df_filt.empty or not selected_ids:
        view_table = mo.alert("No selections yet.")
    else:
        cols = [
            "run_uid",
            "dataset",
            "problem_type",
            "is_correct",
            "probe_margin",
            "se_probe_margin",
            "entropy_baseline_margin",
            "semantic_entropy",
            "mean_think_entropy",
            "think_token_len",
            "think_char_len",
        ]
        missing = [c for c in cols if c not in df_filt.columns]
        for c in missing:
            df_filt[c] = "" if "type" in c else 0
        table_df = df_filt[df_filt["run_uid"].isin(selected_ids)][cols].copy()
        view_table = mo.ui.table(table_df)
    view_table
    return


@app.cell
def _(controls, json, mo):
    metrics_path_val = controls["metrics_path"]
    if not metrics_path_val.exists():
        view_metrics = mo.alert(f"Metrics file not found: {metrics_path_val}")
    else:
        with metrics_path_val.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        cards = []
        for split, vals in metrics.items():
            cards.append(
                mo.md(
                    f"**{split}** | AUC acc={vals.get('auc_accuracy_probe','n/a')} | AUC se={vals.get('auc_se_probe','n/a')} | AUC ent={vals.get('auc_entropy_baseline','n/a')}"
                )
            )
        view_metrics = mo.vstack([mo.md("### Probe metrics"), mo.hstack(cards)])
    view_metrics
    return


if __name__ == "__main__":
    app.run()
