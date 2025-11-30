#!/usr/bin/env python3
"""Evaluate probes on math test and OOD sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample

from sep_marimo.config import DEFAULT_CONFIG


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="Where to write eval JSON")
    return p.parse_args()


def _load_split(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "X": data["X_hidden"].astype(np.float32),
        "y": data["y_correct"].astype(np.int64),
        "ent": data["mean_think_entropy"].astype(np.float32),
    }


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "auprc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def _bootstrap_p(y_true: np.ndarray, scores: np.ndarray, n: int = 500, seed: int = 42) -> float:
    """One-sided p-value that AUROC > 0.5 via bootstrap."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yb = y_true[idx]
        sb = scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return float("nan")
    aucs = np.asarray(aucs)
    return float((aucs <= 0.5).mean())


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    data_dir = args.data_dir or cfg.probe_dataset_dir
    models_dir = args.models_dir or cfg.models_dir
    out_path = args.out or (models_dir / "probe_eval.json")

    acc_probe = joblib.load(models_dir / "accuracy_probe.pkl")
    se_probe = joblib.load(models_dir / "se_probe.pkl")
    ent_probe = joblib.load(models_dir / "entropy_baseline.pkl")
    with (models_dir / "se_threshold.json").open("r", encoding="utf-8") as f:
        tau_data = json.load(f)
    tau = float(tau_data.get("tau_answer"))

    results: Dict[str, Dict[str, float]] = {}

    math_path = data_dir / "math_test.npz"
    def eval_split(name: str, split_data: Dict[str, np.ndarray]):
        y = split_data["y"]
        p_acc = acc_probe.predict_proba(split_data["X"])[:, 1]
        p_high_se = se_probe.predict_proba(split_data["X"])[:, 1]
        conf_se = 1.0 - p_high_se
        p_ent = ent_probe.predict_proba(split_data["ent"].reshape(-1, 1))[:, 1]

        metrics = {
            "auc_accuracy_probe": _metrics(y, p_acc)["roc_auc"],
            "auprc_accuracy_probe": _metrics(y, p_acc)["auprc"],
            "p_value_auc_accuracy": _bootstrap_p(y, p_acc),
            "auc_se_probe": _metrics(y, conf_se)["roc_auc"],
            "auprc_se_probe": _metrics(y, conf_se)["auprc"],
            "p_value_auc_se": _bootstrap_p(y, conf_se),
            "auc_entropy_baseline": _metrics(y, p_ent)["roc_auc"],
            "auprc_entropy_baseline": _metrics(y, p_ent)["auprc"],
            "p_value_auc_entropy": _bootstrap_p(y, p_ent),
            "mean_conf_acc_correct": float(p_acc[y == 1].mean()) if (y == 1).any() else float("nan"),
            "mean_conf_acc_incorrect": float(p_acc[y == 0].mean()) if (y == 0).any() else float("nan"),
            "tau_answer": tau,
        }
        results[name] = metrics
        print(f"[{name}]", metrics)

    if math_path.exists():
        eval_split("math_test", _load_split(math_path))
    else:
        print("math_test.npz not found; skipping")

    ood_path = data_dir / "ood_test.npz"
    if ood_path.exists():
        o = _load_split(ood_path)
        p_acc = acc_probe.predict_proba(o["X"])[:, 1]
        p_high_se = se_probe.predict_proba(o["X"])[:, 1]
        conf_se = 1.0 - p_high_se
        p_ent = ent_probe.predict_proba(o["ent"].reshape(-1, 1))[:, 1]
        results["ood_test"] = {
            "auc_accuracy_probe": _metrics(o["y"], p_acc)["roc_auc"],
            "auprc_accuracy_probe": _metrics(o["y"], p_acc)["auprc"],
            "auc_se_probe": _metrics(o["y"], conf_se)["roc_auc"],
            "auprc_se_probe": _metrics(o["y"], conf_se)["auprc"],
            "auc_entropy_baseline": _metrics(o["y"], p_ent)["roc_auc"],
            "auprc_entropy_baseline": _metrics(o["y"], p_ent)["auprc"],
            "tau_answer": tau,
        }
        print("[ood_test]", results["ood_test"])
    else:
        print("ood_test.npz not found; skipping")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
