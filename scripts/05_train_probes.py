#!/usr/bin/env python3
"""Train accuracy, semantic-entropy, and entropy-baseline probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sep_marimo.config import DEFAULT_CONFIG


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None, help="Probe dataset directory")
    p.add_argument("--models-dir", type=Path, default=None, help="Where to save models")
    return p.parse_args()


def _load_split(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return (
        data["X_hidden"].astype(np.float32),
        data["y_correct"].astype(np.int64),
        data["mean_think_entropy"].astype(np.float32),
        data["semantic_entropy"].astype(np.float32),
    )


def _train_logreg(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    pipe.fit(X, y)
    return pipe


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    data_dir = args.data_dir or cfg.probe_dataset_dir
    models_dir = args.models_dir or cfg.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "math_train.npz"
    val_path = data_dir / "math_val.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Train/val probe datasets not found. Run 04_build_probe_datasets.py first.")

    X_train, y_train, ent_train, se_train = _load_split(train_path)
    X_val, y_val, ent_val, se_val = _load_split(val_path)

    # Semantic entropy threshold (median over train, ignoring NaNs)
    se_clean = se_train[~np.isnan(se_train)]
    if se_clean.size == 0:
        raise RuntimeError("No semantic entropy values in train set")
    tau = float(np.median(se_clean))
    y_high_se_train = (se_train > tau).astype(int)
    y_high_se_val = (se_val > tau).astype(int)

    acc_probe = _train_logreg(X_train, y_train)
    se_probe = _train_logreg(X_train, y_high_se_train)
    ent_probe = _train_logreg(ent_train.reshape(-1, 1), y_train)

def _metrics(model, X, y_true):
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "auprc": float("nan")}
    probs = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
    }

    print("[val] accuracy probe:", _metrics(acc_probe, X_val, y_val))
    print("[val] se probe (label=high SE):", _metrics(se_probe, X_val, y_high_se_val))
    print("[val] entropy baseline:", _metrics(ent_probe, ent_val.reshape(-1, 1), y_val))

    joblib.dump(acc_probe, models_dir / "accuracy_probe.pkl")
    joblib.dump(se_probe, models_dir / "se_probe.pkl")
    joblib.dump(ent_probe, models_dir / "entropy_baseline.pkl")
    with (models_dir / "se_threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"tau_answer": tau}, f, indent=2)
    print(f"Saved probes to {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
