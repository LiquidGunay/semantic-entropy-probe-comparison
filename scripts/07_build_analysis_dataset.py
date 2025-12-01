#!/usr/bin/env python3
"""Build a compact per-run analysis dataset with probe scores and UMAP coords."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import umap

from sep_marimo.config import DEFAULT_CONFIG
from sep_marimo import utils as U


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None, help="Data directory (raw, runs, semantic entropy)")
    p.add_argument("--artifacts-dir", type=Path, default=None, help="Artifacts directory (hidden states, probe datasets)")
    p.add_argument("--models-dir", type=Path, default=None, help="Directory containing trained probes")
    p.add_argument("--out", type=Path, default=None, help="Output parquet path (default: artifacts/analysis/analysis.parquet)")
    p.add_argument("--math-runs", type=Path, default=None, help="Optional override for math runs JSONL (cleaned)")
    p.add_argument("--math-sem", type=Path, default=None, help="Optional override for math semantic entropy JSONL (cleaned)")
    p.add_argument("--ood-runs", type=Path, default=None, help="Optional override for ood runs JSONL")
    p.add_argument("--ood-sem", type=Path, default=None, help="Optional override for ood semantic entropy JSONL")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows to include (random sample if set)")
    p.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    p.add_argument("--umap-sample", type=int, default=5000, help="Max rows to fit UMAP on (transforms all rows)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _load_questions(path: Path) -> Dict[str, Dict]:
    qmap: Dict[str, Dict] = {}
    for rec in U.load_jsonl(path):
        qmap[str(rec.get("id"))] = rec
    return qmap


def _load_runs(path: Path) -> Dict[Tuple[str, int], Dict]:
    rmap: Dict[Tuple[str, int], Dict] = {}
    for rec in U.load_jsonl(path):
        qid = str(rec.get("question_id"))
        rid = int(rec.get("run_id", 0))
        rmap[(qid, rid)] = rec
    return rmap


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X": data["X_hidden"].astype(np.float32),
        "y": data["y_correct"].astype(np.int64),
        "ent": data["mean_think_entropy"].astype(np.float32),
        "se": data["semantic_entropy"].astype(np.float32) if "semantic_entropy" in data else None,
        "question_ids": np.asarray([str(x) for x in data["question_ids"]], dtype=object),
        "run_ids": data["run_ids"].astype(np.int32),
    }


def _projection(pipe, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (normalized projection, normalized margin) for a sklearn Pipeline."""
    scaler = None
    clf = None
    if hasattr(pipe, "named_steps"):
        scaler = pipe.named_steps.get("standardscaler")
        clf = pipe.named_steps.get("logisticregression")
    if scaler is None and hasattr(pipe, "steps"):
        # fallback to first/last
        scaler = pipe.steps[0][1]
        clf = pipe.steps[-1][1]
    if scaler is None or clf is None:
        return np.zeros(X.shape[0], dtype=np.float32), np.zeros(X.shape[0], dtype=np.float32)
    X_scaled = scaler.transform(X)
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    norm = float(np.linalg.norm(coef) + 1e-12)
    projection = (X_scaled @ coef) / norm
    margin = (X_scaled @ coef + intercept) / norm
    return projection.astype(np.float32), margin.astype(np.float32)


def _think_token_length(rec: Dict) -> int:
    idx = rec.get("think_token_indices")
    if isinstance(idx, (list, tuple)) and len(idx) == 2:
        start, end = idx
        try:
            return max(0, int(end) - int(start) + 1)
        except Exception:
            return 0
    return 0


def _maybe_nan(val) -> float:
    try:
        if val is None:
            return math.nan
        return float(val)
    except Exception:
        return math.nan


def _fit_umap(X: np.ndarray, sample: int, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    idx = np.arange(X.shape[0])
    if sample and X.shape[0] > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=sample, replace=False)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
        metric="euclidean",
    )
    reducer.fit(X[idx])
    coords = reducer.transform(X)
    return coords.astype(np.float32)


def _process_split(
    name: str,
    dataset_label: str,
    npz_data: Dict[str, np.ndarray],
    runs: Dict[Tuple[str, int], Dict],
    qmap: Dict[str, Dict],
    acc_probe,
    se_probe,
    ent_probe,
    tau_answer: float,
    ) -> Tuple[List[Dict], np.ndarray]:
    X = npz_data["X"]
    y = npz_data["y"]
    ent = npz_data["ent"]
    se_arr = npz_data.get("se")
    qids = npz_data["question_ids"]
    rids = npz_data["run_ids"]

    if X.size == 0:
        return [], np.zeros((0, X.shape[1]), dtype=np.float32)

    prob_acc = acc_probe.predict_proba(X)[:, 1]
    margin_acc = acc_probe.decision_function(X)
    proj_acc, norm_margin_acc = _projection(acc_probe, X)

    prob_se = se_probe.predict_proba(X)[:, 1]
    margin_se = se_probe.decision_function(X)
    proj_se, norm_margin_se = _projection(se_probe, X)

    ent_feat = ent.reshape(-1, 1)
    prob_ent = ent_probe.predict_proba(ent_feat)[:, 1]
    margin_ent = ent_probe.decision_function(ent_feat)

    rows: List[Dict] = []
    kept_feats: List[np.ndarray] = []
    for i in range(X.shape[0]):
        qid = str(qids[i])
        rid = int(rids[i])
        rec = runs.get((qid, rid))
        if rec is None:
            continue
        kept_feats.append(X[i])
        qinfo = qmap.get(qid, {})
        se_val = _maybe_nan(se_arr[i]) if se_arr is not None else _maybe_nan(rec.get("semantic_entropy"))
        row = {
            "run_uid": f"{dataset_label}-{qid}-{rid}",
            "dataset": dataset_label,
            "split": name,
            "question_id": qid,
            "run_id": rid,
            "is_correct": bool(y[i]),
            "probe_prob_correct": float(prob_acc[i]),
            "probe_margin": float(margin_acc[i]),
            "probe_norm_margin": float(norm_margin_acc[i]),
            "probe_projection": float(proj_acc[i]),
            "se_probe_prob_high": float(prob_se[i]),
            "se_probe_margin": float(margin_se[i]),
            "se_probe_norm_margin": float(norm_margin_se[i]),
            "entropy_baseline_prob": float(prob_ent[i]),
            "entropy_baseline_margin": float(margin_ent[i]),
            "mean_think_entropy": float(ent[i]),
            "semantic_entropy": se_val,
            "tau_answer": float(tau_answer),
            "question": qinfo.get("question", ""),
            "gold_answer": qinfo.get("answer", ""),
            "think_text": rec.get("think_text", ""),
            "answer_text": rec.get("answer_text", ""),
            "output_text": rec.get("output_text", ""),
            "think_token_len": _think_token_length(rec),
            "think_char_len": len(rec.get("think_text", "") or ""),
        }
        rows.append(row)
    feats = np.stack(kept_feats, axis=0) if kept_feats else np.zeros((0, X.shape[1]), dtype=np.float32)
    return rows, feats


def _mark_representatives(rows: List[Dict]) -> None:
    by_question: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in enumerate(rows):
        key = (row["dataset"], row["question_id"])
        by_question.setdefault(key, []).append(idx)
    for key, idxs in by_question.items():
        best = max(idxs, key=lambda j: rows[j].get("probe_prob_correct", 0.0))
        for j in idxs:
            rows[j]["is_representative"] = j == best


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    data_dir = args.data_dir or cfg.data_dir
    artifacts_dir = args.artifacts_dir or cfg.artifacts_dir
    models_dir = args.models_dir or cfg.models_dir
    out_path = args.out or (artifacts_dir / "analysis" / "analysis.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load probes and threshold
    acc_probe = joblib.load(models_dir / "accuracy_probe.pkl")
    se_probe = joblib.load(models_dir / "se_probe.pkl")
    ent_probe = joblib.load(models_dir / "entropy_baseline.pkl")
    tau = 0.0
    try:
        import json

        with (models_dir / "se_threshold.json").open("r", encoding="utf-8") as f:
            tau_json = json.load(f)
        tau = float(tau_json.get("tau_answer", 0.0))
    except Exception:
        tau = 0.0

    math_qmap = _load_questions(data_dir / "math_raw.jsonl")
    ood_qmap = _load_questions(data_dir / "ood_raw.jsonl") if (data_dir / "ood_raw.jsonl").exists() else {}

    math_runs_path = args.math_runs or data_dir / "math_runs.jsonl"
    ood_runs_path = args.ood_runs or data_dir / "ood_runs.jsonl"
    math_runs = _load_runs(math_runs_path)
    ood_runs = _load_runs(ood_runs_path) if ood_runs_path.exists() else {}

    rows: List[Dict] = []
    feats: List[np.ndarray] = []

    for split_name in ["math_train", "math_val", "math_test"]:
        path = artifacts_dir / "probe_datasets" / f"{split_name}.npz"
        if not path.exists():
            print(f"{split_name}.npz not found; skipping {split_name}")
            continue
        data = _load_npz(path)
        tag = split_name
        math_rows, math_feats = _process_split(
            tag, "math", data, math_runs, math_qmap, acc_probe, se_probe, ent_probe, tau
        )
        rows.extend(math_rows)
        feats.append(math_feats)
        print(f"Loaded {len(math_rows)} {tag} rows")

    ood_npz = artifacts_dir / "probe_datasets" / "ood_test.npz"
    if ood_npz.exists():
        ood_data = _load_npz(ood_npz)
        ood_rows, ood_feats = _process_split(
            "ood_test", "ood", ood_data, ood_runs, ood_qmap, acc_probe, se_probe, ent_probe, tau
        )
        rows.extend(ood_rows)
        feats.append(ood_feats)
        print(f"Loaded {len(ood_rows)} ood_test rows")
    else:
        print("ood_test.npz not found; skipping ood")

    if not rows:
        print("No rows to write")
        return 0

    _mark_representatives(rows)

    X_all = np.vstack(feats)
    coords = _fit_umap(
        X_all,
        sample=args.umap_sample,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )
    for row, (x, y) in zip(rows, coords):
        row["umap_x"] = float(x)
        row["umap_y"] = float(y)

    if args.max_rows is not None and len(rows) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(rows), size=args.max_rows, replace=False)
        rows = [rows[i] for i in idx]

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
