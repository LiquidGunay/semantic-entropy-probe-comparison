import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd
    from sklearn.metrics import roc_curve

    return Path, alt, json, mo, os, pd, roc_curve


@app.cell
def _(mo, os):
    # Keep marimo virtual files out of /dev/shm on constrained hosts.
    if os.getenv("MARIMO_NO_SHM", "0") == "1":
        import marimo._runtime.virtual_file as vf

        def _create_data_url(self, context):  # noqa: ARG001
            import marimo._runtime.virtual_file as _vf

            filename = _vf.random_filename(self.ext)
            self._virtual_file = _vf.VirtualFile(
                filename=filename,
                buffer=self.buffer,
                as_data_url=True,
            )

        if getattr(vf.VirtualFileLifecycleItem.create, "__marimo_no_shm_patch__", False) is False:
            _create_data_url.__marimo_no_shm_patch__ = True  # type: ignore[attr-defined]
            vf.VirtualFileLifecycleItem.create = _create_data_url  # type: ignore[method-assign]

        shm_notice = mo.md("`MARIMO_NO_SHM=1`: virtual files inlined as data URLs.")
    else:
        shm_notice = None
    shm_notice
    return


@app.cell
def _(alt, mo, os):
    status = "VegaFusion transformer disabled (set USE_VEGAFUSION=1 to test)."
    if os.getenv("USE_VEGAFUSION", "0") == "1":
        try:
            alt.data_transformers.enable("vegafusion")
            status = "VegaFusion transformer enabled for Altair."
        except Exception as exc:  # noqa: BLE001
            status = f"VegaFusion unavailable: {exc}"
    mo.callout(status, kind="warn")
    return


@app.cell
def _(mo):
    mo.md(
        """
    # Probe Analysis (Altair + marimo)
    Fast exploration view for probe margins, entropy signals, and UMAP structure.

    This notebook expects a **split dataset**:
    - chart dataset: light-weight numeric columns for plotting
    - detail dataset: heavy text columns for selected-run inspection
    """
    )
    return


@app.cell
def _(Path, json, mo, os, pd):
    chart_path = Path(os.getenv("ANALYSIS_CHART_PARQUET", "artifacts_clean/analysis/analysis_chart.parquet"))
    fallback_path = Path(os.getenv("ANALYSIS_PARQUET", "artifacts_clean/analysis/analysis.parquet"))
    detail_path = Path(os.getenv("ANALYSIS_DETAIL_PARQUET", "artifacts_clean/analysis/analysis_detail.parquet"))
    metrics_path = Path(os.getenv("METRICS_JSON", "artifacts_clean/models/probe_eval.json"))

    if chart_path.exists():
        chart_df = pd.read_parquet(chart_path)
        chart_notice = mo.md(f"Loaded chart dataset: `{chart_path}`")
    elif fallback_path.exists():
        chart_df = pd.read_parquet(fallback_path)
        chart_notice = mo.callout(
            f"Chart parquet not found. Using fallback full dataset: `{fallback_path}`",
            kind="warn",
        )
    else:
        chart_df = pd.DataFrame()
        chart_notice = mo.alert(f"No chart dataset found. Checked: `{chart_path}` and `{fallback_path}`")

    if detail_path.exists():
        detail_df = pd.read_parquet(detail_path)
        detail_notice = mo.md(f"Loaded detail dataset: `{detail_path}`")
    else:
        detail_df = pd.DataFrame()
        detail_notice = mo.callout(
            "Detail dataset not found; selected-run inspection will be limited.",
            kind="warn",
        )

    metrics = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    controls = {
        "dataset_filter": mo.ui.dropdown(options=["all", "math", "ood"], value="all", label="Dataset"),
        "rep_only": mo.ui.switch(value=False, label="Representatives only"),
        "correctness_filter": mo.ui.dropdown(
            options=["all", "correct", "incorrect"], value="all", label="Correctness"
        ),
        "max_points": mo.ui.slider(200, 10000, value=5000, step=200, label="Max plotted points"),
        "seed_box": mo.ui.number(start=0, stop=10000, step=1, value=42, label="Sampling seed"),
    }

    mo.vstack(
        [
            chart_notice,
            detail_notice,
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
    if controls["rep_only"].value:
        if "is_representative" in df_filt.columns:
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

    mo.hstack(
        [
            selection_widget,
            mo.md(
                f"{len(df_filt)} rows | correct={int(df_filt.get('is_correct', []).sum()) if not df_filt.empty else 0}"
            ),
        ]
    )
    return df_filt, selection_widget


@app.cell
def _(alt, df_filt, mo, selection_widget):
    margin_sel_ids = list(selection_widget.value)[:10]
    if df_filt.empty:
        margin_chart = None
        margin_view = mo.alert("No rows to plot")
    else:
        cols = ["run_uid", "question_id", "probe_margin", "mean_think_entropy", "is_correct"]
        margin_plot_df = df_filt[[c for c in cols if c in df_filt.columns]].dropna(
            subset=[c for c in ["probe_margin", "mean_think_entropy"] if c in df_filt.columns]
        )
        margin_chart_spec = (
            alt.Chart(margin_plot_df)
            .mark_circle(size=56, opacity=0.72)
            .encode(
                x=alt.X("probe_margin:Q", title="Probe margin"),
                y=alt.Y("mean_think_entropy:Q", title="Mean think entropy"),
                color=alt.Color("is_correct:N", title="Correct"),
                tooltip=["run_uid", "question_id", "probe_margin", "mean_think_entropy", "is_correct"],
                opacity=alt.condition(
                    alt.FieldOneOfPredicate(field="run_uid", oneOf=margin_sel_ids),
                    alt.value(0.95),
                    alt.value(0.4),
                ),
            )
            .properties(height=290, width="container")
        )
        margin_chart = mo.ui.altair_chart(margin_chart_spec, chart_selection="interval")
        margin_view = mo.vstack([mo.md("### Margin vs entropy"), margin_chart])
    margin_view
    return margin_chart


@app.cell
def _(alt, df_filt, mo, selection_widget):
    umap_sel_ids = list(selection_widget.value)[:10]
    if df_filt.empty or not {"umap_x", "umap_y"}.issubset(df_filt.columns):
        umap_view = mo.alert("No UMAP data available")
    else:
        umap_blocks = []
        for col, title in [
            ("probe_margin", "Probe margin"),
            ("se_probe_margin", "SE probe margin"),
            ("entropy_baseline_margin", "Entropy baseline margin"),
        ]:
            if col not in df_filt.columns:
                continue
            umap_plot_df = df_filt[["run_uid", "question_id", "umap_x", "umap_y", "is_correct", col]].copy()
            umap_chart = (
                alt.Chart(umap_plot_df)
                .mark_circle(size=45, opacity=0.78)
                .encode(
                    x=alt.X("umap_x:Q", title="UMAP-1"),
                    y=alt.Y("umap_y:Q", title="UMAP-2"),
                    color=alt.Color(f"{col}:Q", title=title),
                    tooltip=["run_uid", "question_id", col, "is_correct"],
                    opacity=alt.condition(
                        alt.FieldOneOfPredicate(field="run_uid", oneOf=umap_sel_ids),
                        alt.value(0.95),
                        alt.value(0.35),
                    ),
                )
                .properties(height=300, width=300)
            )
            umap_blocks.append(
                mo.vstack([mo.md(f"### {title}"), mo.ui.altair_chart(umap_chart, chart_selection="interval")])
            )
        umap_view = mo.hstack(umap_blocks) if umap_blocks else mo.alert("No UMAP margin columns available")
    umap_view
    return


@app.cell
def _(alt, df_filt, mo):
    if df_filt.empty or not {"semantic_entropy", "se_probe_margin"}.issubset(df_filt.columns):
        se_view = mo.alert("No SE data to plot")
    else:
        se_plot_df = df_filt[["run_uid", "question_id", "semantic_entropy", "se_probe_margin", "is_correct"]].dropna()
        se_chart = (
            alt.Chart(se_plot_df)
            .mark_circle(size=48, opacity=0.74)
            .encode(
                x=alt.X("semantic_entropy:Q", title="Semantic entropy"),
                y=alt.Y("se_probe_margin:Q", title="SE probe margin"),
                color=alt.Color("is_correct:N", title="Correct"),
                tooltip=["run_uid", "question_id", "semantic_entropy", "se_probe_margin", "is_correct"],
            )
            .properties(height=290, width="container")
        )
        se_view = mo.vstack([mo.md("### Semantic entropy vs SE-probe margin"), mo.ui.altair_chart(se_chart)])
    se_view
    return


@app.cell
def _(alt, df_filt, mo, pd, roc_curve):
    if df_filt.empty:
        roc_view = mo.alert("No rows for ROC plot")
    else:
        labels = df_filt["is_correct"].astype(int)
        curves = []
        for name, series_name in [
            ("probe", "probe_prob_correct"),
            ("se_probe", "se_probe_prob_high"),
            ("entropy", "entropy_baseline_prob"),
        ]:
            if series_name not in df_filt.columns:
                continue
            fpr, tpr, _ = roc_curve(labels, df_filt[series_name])
            curves.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "probe": name}))
        if not curves:
            roc_view = mo.alert("No probability columns for ROC plot")
        else:
            roc_df = pd.concat(curves, ignore_index=True)
            roc_chart = (
                alt.Chart(roc_df)
                .mark_line()
                .encode(x="fpr:Q", y="tpr:Q", color="probe:N")
                .properties(height=250, width="container")
            )
            roc_view = mo.vstack([mo.md("### ROC curves"), mo.ui.altair_chart(roc_chart)])
    roc_view
    return


@app.cell
def _(detail_df, df_filt, mo, selection_widget):
    selected_ids = list(selection_widget.value)[:10]
    if not selected_ids:
        detail_view = mo.alert("No selected runs yet")
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
            "think_token_len",
            "think_char_len",
        ]
        score_df = df_filt[[c for c in score_cols if c in df_filt.columns]].copy()
        score_df = score_df[score_df["run_uid"].isin(selected_ids)]

        if detail_df.empty:
            detail_view = mo.vstack(
                [mo.callout("Detail dataset missing; showing score-only table.", kind="warn"), mo.ui.table(score_df)]
            )
        else:
            detail_cols = [
                "run_uid",
                "question",
                "gold_answer",
                "answer_text",
                "think_text",
                "output_text",
            ]
            ddf = detail_df[[c for c in detail_cols if c in detail_df.columns]].copy()
            ddf = ddf[ddf["run_uid"].isin(selected_ids)]
            merged = score_df.merge(ddf, on="run_uid", how="left")
            detail_view = mo.ui.table(merged)
    detail_view
    return


@app.cell
def _(metrics, mo):
    if not metrics:
        metrics_view = mo.callout("Metrics JSON missing or unreadable.", kind="warn")
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
        metrics_view = mo.vstack([mo.md("### Probe metrics"), mo.hstack(cards)])
    metrics_view
    return


if __name__ == "__main__":
    app.run()
