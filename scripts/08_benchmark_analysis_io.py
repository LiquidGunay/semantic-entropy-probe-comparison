#!/usr/bin/env python3
"""Benchmark analysis dataset load and chart-spec generation latency."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("artifacts_clean/analysis/analysis_chart.parquet"),
        help="Chart dataset parquet",
    )
    parser.add_argument(
        "--detail",
        type=Path,
        default=Path("artifacts_clean/analysis/analysis_detail.parquet"),
        help="Detail dataset parquet",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Point cap to benchmark plotting path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report",
    )
    return parser.parse_args()


def timed_read_parquet(path: Path) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    frame = pd.read_parquet(path)
    end = time.perf_counter()
    return frame, end - start


def main() -> int:
    args = parse_args()
    report: dict[str, float | int] = {}

    if not args.chart.exists():
        print(f"Chart parquet not found: {args.chart}")
        return 1

    tracemalloc.start()

    chart_df, chart_load_s = timed_read_parquet(args.chart)
    report["chart_rows"] = int(len(chart_df))
    report["chart_load_seconds"] = round(chart_load_s, 6)
    report["chart_memory_bytes"] = int(chart_df.memory_usage(deep=True).sum())

    if args.detail.exists():
        detail_df, detail_load_s = timed_read_parquet(args.detail)
        report["detail_rows"] = int(len(detail_df))
        report["detail_load_seconds"] = round(detail_load_s, 6)
        report["detail_memory_bytes"] = int(detail_df.memory_usage(deep=True).sum())
    else:
        detail_df = pd.DataFrame()
        report["detail_rows"] = 0
        report["detail_load_seconds"] = 0.0
        report["detail_memory_bytes"] = 0

    filter_start = time.perf_counter()
    filt = chart_df
    if "dataset" in filt.columns:
        filt = filt[filt["dataset"] == "math"] if (filt["dataset"] == "math").any() else filt
    if len(filt) > args.max_points:
        filt = filt.sample(n=args.max_points, random_state=args.seed)
    filt = filt.reset_index(drop=True)
    filter_end = time.perf_counter()
    report["filter_seconds"] = round(filter_end - filter_start, 6)
    report["filtered_rows"] = int(len(filt))

    spec_seconds = 0.0
    try:
        import altair as alt

        plot_cols = [c for c in ["probe_margin", "mean_think_entropy", "is_correct"] if c in filt.columns]
        if {"probe_margin", "mean_think_entropy"}.issubset(plot_cols):
            spec_start = time.perf_counter()
            chart = (
                alt.Chart(filt[["probe_margin", "mean_think_entropy", "is_correct"]])
                .mark_circle(size=56, opacity=0.72)
                .encode(
                    x=alt.X("probe_margin:Q"),
                    y=alt.Y("mean_think_entropy:Q"),
                    color=alt.Color("is_correct:N"),
                )
            )
            _ = chart.to_dict()
            spec_seconds = time.perf_counter() - spec_start
    except Exception:
        spec_seconds = 0.0
    report["chart_spec_seconds"] = round(spec_seconds, 6)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report["tracemalloc_current_bytes"] = int(current)
    report["tracemalloc_peak_bytes"] = int(peak)

    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote benchmark report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
