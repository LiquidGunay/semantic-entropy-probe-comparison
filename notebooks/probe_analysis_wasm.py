import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, alt, json, mo, np, os, pd


@app.cell
def _(np):
    def roc_curve_np(labels, scores):
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores)
        if labels.size == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])
        order = np.argsort(-scores)
        labels = labels[order]
        tps = np.cumsum(labels)
        fps = np.cumsum(1 - labels)
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
        return np.concatenate([[0.0], fpr, [1.0]]), np.concatenate([[0.0], tpr, [1.0]])

    return (roc_curve_np,)


@app.cell
def _(mo):
    mo.md(
        """
    # Probe Analysis (WASM)
    Lightweight browser-first view using CSV assets under `public/`.

    Preferred files:
    - `public/analysis_chart.csv`
    - `public/analysis_detail.csv`
    - `public/probe_eval.json`
    """
    )
    return


@app.cell
def _(Path, json, mo, os, pd):
    try:
        base = mo.notebook_location()
    except Exception:  # noqa: BLE001
        base = None
    if base is None:
        base = Path(".").resolve()

    base_str = str(base)
    is_wasm = base_str.startswith("http://") or base_str.startswith("https://")

    chart_source = os.getenv("ANALYSIS_CHART_CSV", str(base / "public" / "analysis_chart.csv"))
    fallback_source = os.getenv("ANALYSIS_CSV", str(base / "public" / "analysis.csv"))
    detail_source = os.getenv("ANALYSIS_DETAIL_CSV", str(base / "public" / "analysis_detail.csv"))
    metrics_source = os.getenv("METRICS_JSON", str(base / "public" / "probe_eval.json"))

    def _read_csv(source: str) -> pd.DataFrame:
        if is_wasm:
            from pyodide.http import open_url

            with open_url(source) as f:
                return pd.read_csv(f)
        return pd.read_csv(source)

    try:
        chart_df = _read_csv(chart_source)
        chart_notice = mo.md(f"Loaded chart CSV: `{chart_source}`")
    except Exception as exc:  # noqa: BLE001
        try:
            chart_df = _read_csv(fallback_source)
            chart_notice = mo.callout(
                f"Chart CSV missing. Using fallback dataset `{fallback_source}`. Error: {exc}",
                kind="warn",
            )
        except Exception as fallback_exc:  # noqa: BLE001
            chart_df = pd.DataFrame()
            chart_notice = mo.alert(
                f"Failed loading chart data:\n- {chart_source}\n- {fallback_source}\n{fallback_exc}"
            )

    try:
        detail_df = _read_csv(detail_source)
        detail_notice = mo.md(f"Loaded detail CSV: `{detail_source}`")
    except Exception:
        detail_df = pd.DataFrame()
        detail_notice = mo.callout("Detail CSV missing; showing score-only details.", kind="warn")

    metrics = None
    try:
        if is_wasm:
            from pyodide.http import open_url

            with open_url(metrics_source) as f:
                metrics = json.load(f)
        else:
            with Path(metrics_source).open("r", encoding="utf-8") as f:
                metrics = json.load(f)
    except Exception:
        metrics = None

    # dtype coercion for plotting
    if not chart_df.empty:
        bool_cols = ["is_correct", "is_representative"]
        num_cols = [
            "probe_margin",
            "mean_think_entropy",
            "se_probe_margin",
            "entropy_baseline_margin",
            "probe_prob_correct",
            "se_probe_prob_high",
            "entropy_baseline_prob",
            "semantic_entropy",
            "umap_x",
            "umap_y",
            "think_token_len",
            "think_char_len",
        ]
        for col in num_cols:
            if col in chart_df.columns:
                chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")
        for col in bool_cols:
            if col in chart_df.columns:
                chart_df[col] = chart_df[col].astype(str).str.lower().isin({"true", "1", "yes"})

    controls = {
        "dataset_filter": mo.ui.dropdown(options=["all", "math", "ood"], value="all", label="Dataset"),
        "rep_only": mo.ui.switch(value=False, label="Representatives only"),
        "correctness_filter": mo.ui.dropdown(
            options=["all", "correct", "incorrect"], value="all", label="Correctness"
        ),
        "max_points": mo.ui.slider(200, 10000, value=4000, step=200, label="Max plotted points"),
        "seed_box": mo.ui.number(start=0, stop=10000, step=1, value=42, label="Sampling seed"),
    }

    mo.vstack(
        [
            chart_notice,
            detail_notice,
            mo.callout("WASM mode uses dataframe reduction; VegaFusion transformer is not enabled here.", kind="warn"),
            mo.hstack(
                [
                    controls["dataset_filter"],
                    controls["rep_only"],
                    controls["correctness_filter"],
                    controls["max_points"],
                    controls["seed_box"],
                ]
            ),
        ]
    )
    return chart_df, controls, detail_df, metrics


@app.cell
def _(chart_df, controls, mo):
    df_filt = chart_df.copy()
    if controls["dataset_filter"].value != "all":
        df_filt = df_filt[df_filt["dataset"] == controls["dataset_filter"].value]
    if controls["rep_only"].value and "is_representative" in df_filt.columns:
        df_filt = df_filt[df_filt["is_representative"]]
    if controls["correctness_filter"].value == "correct":
        df_filt = df_filt[df_filt["is_correct"]]
    elif controls["correctness_filter"].value == "incorrect":
        df_filt = df_filt[~df_filt["is_correct"]]

    if len(df_filt) > controls["max_points"].value:
        df_filt = df_filt.sample(
            n=int(controls["max_points"].value),
            random_state=int(controls["seed_box"].value),
        )

    df_filt = df_filt.reset_index(drop=True)
    selection_widget = mo.ui.multiselect(
        options=df_filt.get("run_uid", []).tolist()[:5000],
        value=df_filt.get("run_uid", []).tolist()[: min(3, len(df_filt))],
        label="Selected runs (<=10)",
    )
    mo.hstack([selection_widget, mo.md(f"{len(df_filt)} rows")])
    return df_filt, selection_widget


@app.cell
def _(alt, df_filt, mo, selection_widget):
    selected_ids = list(selection_widget.value)[:10]
    if df_filt.empty:
        view = mo.alert("No rows to plot")
    else:
        plot_df = df_filt[["run_uid", "question_id", "probe_margin", "mean_think_entropy", "is_correct"]].dropna(
            subset=["probe_margin", "mean_think_entropy"]
        )
        chart = (
            alt.Chart(plot_df)
            .mark_circle(size=54, opacity=0.72)
            .encode(
                x=alt.X("probe_margin:Q", title="Probe margin"),
                y=alt.Y("mean_think_entropy:Q", title="Mean think entropy"),
                color=alt.Color("is_correct:N", title="Correct"),
                tooltip=["run_uid", "question_id", "probe_margin", "mean_think_entropy", "is_correct"],
                opacity=alt.condition(
                    alt.FieldOneOfPredicate(field="run_uid", oneOf=selected_ids),
                    alt.value(0.95),
                    alt.value(0.4),
                ),
            )
            .properties(height=280, width="container")
        )
        view = mo.vstack([mo.md("### Margin vs entropy"), mo.ui.altair_chart(chart, chart_selection="interval")])
    view
    return


@app.cell
def _(alt, df_filt, mo):
    if df_filt.empty or not {"umap_x", "umap_y"}.issubset(df_filt.columns):
        view = mo.alert("No UMAP data available")
    else:
        chart = (
            alt.Chart(df_filt[["run_uid", "question_id", "umap_x", "umap_y", "probe_margin", "is_correct"]])
            .mark_circle(size=45, opacity=0.78)
            .encode(
                x=alt.X("umap_x:Q", title="UMAP-1"),
                y=alt.Y("umap_y:Q", title="UMAP-2"),
                color=alt.Color("probe_margin:Q", title="Probe margin"),
                tooltip=["run_uid", "question_id", "probe_margin", "is_correct"],
            )
            .properties(height=300, width="container")
        )
        view = mo.vstack([mo.md("### UMAP"), mo.ui.altair_chart(chart)])
    view
    return


@app.cell
def _(alt, df_filt, mo, np, pd, roc_curve_np):
    if df_filt.empty:
        view = mo.alert("No rows for ROC plot")
    else:
        labels = df_filt["is_correct"].astype(int).to_numpy()
        curves = []
        for name, series_name in [
            ("probe", "probe_prob_correct"),
            ("se_probe", "se_probe_prob_high"),
            ("entropy", "entropy_baseline_prob"),
        ]:
            if series_name not in df_filt.columns:
                continue
            scores = df_filt[series_name].fillna(0).to_numpy()
            fpr, tpr = roc_curve_np(labels, scores)
            curves.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "probe": name}))
        if not curves:
            view = mo.alert("No probability columns for ROC plot")
        else:
            roc_df = pd.concat(curves, ignore_index=True)
            chart = (
                alt.Chart(roc_df)
                .mark_line()
                .encode(x="fpr:Q", y="tpr:Q", color="probe:N")
                .properties(height=240, width="container")
            )
            view = mo.vstack([mo.md("### ROC curves"), mo.ui.altair_chart(chart)])
    view
    return


@app.cell
def _(detail_df, df_filt, mo, selection_widget):
    selected_ids = list(selection_widget.value)[:10]
    if not selected_ids:
        view = mo.alert("No selected runs yet")
    else:
        score_cols = [
            "run_uid",
            "dataset",
            "problem_type",
            "is_correct",
            "probe_margin",
            "se_probe_margin",
            "entropy_baseline_margin",
            "semantic_entropy",
            "mean_think_entropy",
        ]
        score_df = df_filt[[c for c in score_cols if c in df_filt.columns]].copy()
        score_df = score_df[score_df["run_uid"].isin(selected_ids)]

        if detail_df.empty:
            view = mo.ui.table(score_df)
        else:
            detail_cols = ["run_uid", "question", "gold_answer", "answer_text", "think_text"]
            ddf = detail_df[[c for c in detail_cols if c in detail_df.columns]].copy()
            ddf = ddf[ddf["run_uid"].isin(selected_ids)]
            merged = score_df.merge(ddf, on="run_uid", how="left")
            view = mo.ui.table(merged)
    view
    return


@app.cell
def _(metrics, mo):
    if not metrics:
        view = mo.callout("Metrics JSON missing or unreadable.", kind="warn")
    else:
        cards = []
        for split, vals in metrics.items():
            cards.append(
                mo.md(
                    f"**{split}** | AUC acc={vals.get('auc_accuracy_probe', 'n/a')} | "
                    f"AUC se={vals.get('auc_se_probe', 'n/a')} | "
                    f"AUC ent={vals.get('auc_entropy_baseline', 'n/a')}"
                )
            )
        view = mo.vstack([mo.md("### Probe metrics"), mo.hstack(cards)])
    view
    return


if __name__ == "__main__":
    app.run()
