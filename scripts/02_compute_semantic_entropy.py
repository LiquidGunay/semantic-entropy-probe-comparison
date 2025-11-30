#!/usr/bin/env python3
"""Compute question-level semantic entropy from multiple sampled answers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from sep_marimo.config import DEFAULT_CONFIG
from sep_marimo import utils as U


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-path", type=Path, default=None, help="Path to math_runs.jsonl")
    p.add_argument("--out", type=Path, default=None, help="Output JSONL path (semantic entropy)")
    return p.parse_args()


def compute_entropy(probabilities: List[float]) -> float:
    ps = np.asarray(probabilities, dtype=np.float32)
    ps = ps[ps > 0]
    if ps.size == 0:
        return float("nan")
    return float(-(ps * np.log(ps)).sum())


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    runs_path = args.runs_path or cfg.math_runs_path
    out_path = args.out or cfg.math_semantic_entropy_path

    runs = U.load_jsonl(runs_path)
    if not runs:
        print(f"No runs found at {runs_path}")
        return 0

    grouped = U.collate_runs_by_question(runs)
    results: List[Dict] = []
    for qid, recs in grouped.items():
        answers = []
        clusters: Dict[str, List[int]] = {}
        for rec in recs:
            run_id = int(rec.get("run_id", 0))
            ans = rec.get("answer_text", "")
            norm = U.normalize_answer_for_cluster(ans)
            if norm not in clusters:
                clusters[norm] = []
            clusters[norm].append(run_id)
            answers.append({"run_id": run_id, "answer_text": ans})
        total = sum(len(v) for v in clusters.values())
        probs = [len(v) / total for v in clusters.values() if total > 0]
        sentropy = compute_entropy(probs)
        record = {
            "question_id": qid,
            "answers": answers,
            "clusters": [
                {"cluster_id": idx, "member_run_ids": members}
                for idx, members in enumerate(clusters.values())
            ],
            "cluster_probs": probs,
            "semantic_entropy": sentropy,
        }
        results.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote semantic entropy for {len(results)} questions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
