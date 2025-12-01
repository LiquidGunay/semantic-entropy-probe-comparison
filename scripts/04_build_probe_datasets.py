#!/usr/bin/env python3
"""Join runs, semantic entropy, and hidden states into probe-ready NPZ files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sep_marimo.config import DEFAULT_CONFIG
from sep_marimo import utils as U


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--math-runs", type=Path, default=None)
    p.add_argument("--ood-runs", type=Path, default=None)
    p.add_argument("--math-hidden", type=Path, default=None)
    p.add_argument("--ood-hidden", type=Path, default=None)
    p.add_argument("--semantic", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _load_hidden(npz_path: Path) -> Dict[Tuple[str, int], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    qids = data["question_ids"]
    run_ids = data["run_ids"]
    h = data["hidden_states"]
    mapping: Dict[Tuple[str, int], np.ndarray] = {}
    for qid, rid, vec in zip(qids, run_ids, h):
        mapping[(str(qid), int(rid))] = np.asarray(vec, dtype=np.float32)
    return mapping


def _build_split_map(question_ids: List[str], split: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    uniq = list(sorted(set(question_ids)))
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    split_map: Dict[str, str] = {}
    for i, qid in enumerate(uniq):
        if i < n_train:
            split_map[qid] = "train"
        elif i < n_train + n_val:
            split_map[qid] = "val"
        else:
            split_map[qid] = "test"
    return split_map


def _append_example(store: Dict[str, List], split: str, qid: str, rid: int, hidden: np.ndarray, rec: Dict, se_map: Dict[str, float]):
    key = f"{split}"
    store.setdefault(f"X_{split}", []).append(hidden)
    store.setdefault(f"y_{split}", []).append(int(rec.get("is_correct", 0)))
    store.setdefault(f"entropy_{split}", []).append(float(rec.get("mean_think_entropy", float("nan"))))
    store.setdefault(f"qid_{split}", []).append(qid)
    store.setdefault(f"rid_{split}", []).append(rid)
    store.setdefault(f"se_{split}", []).append(float(se_map.get(qid, float("nan"))))


def _save_split(out_dir: Path, name: str, X: List[np.ndarray], y: List[int], ent: List[float], se: List[float], qids: List[str], rids: List[int]):
    if not X:
        return
    arr_X = np.stack([np.asarray(x, dtype=np.float32) for x in X], axis=0)
    out_path = out_dir / f"{name}.npz"
    np.savez_compressed(
        out_path,
        X_hidden=arr_X,
        y_correct=np.asarray(y, dtype=np.int8),
        mean_think_entropy=np.asarray(ent, dtype=np.float32),
        semantic_entropy=np.asarray(se, dtype=np.float32),
        question_ids=np.asarray(qids, dtype=object),
        run_ids=np.asarray(rids, dtype=np.int32),
    )
    print(f"Saved {arr_X.shape[0]} rows to {out_path}")


def _ensure_two_classes(target_y: List[int], source: Dict[str, List], source_split: str = "train"):
    uniq = set(target_y)
    if len(uniq) >= 2:
        return
    # Attempt to borrow opposite class from the source split (usually train)
    desired = 1 if 0 in uniq else 0
    source_y = source.get(f"y_{source_split}", [])
    for idx, yval in enumerate(source_y):
        if yval == desired:
            source_X = source.get(f"X_{source_split}", [])[idx]
            source_ent = source.get(f"entropy_{source_split}", [])[idx]
            source_se = source.get(f"se_{source_split}", [])[idx]
            source_qid = source.get(f"qid_{source_split}", [])[idx]
            source_rid = source.get(f"rid_{source_split}", [])[idx]
            target_y.append(yval)
            source.get(f"X_{source_split}")  # keep mypy happy
            return {
                "X": source_X,
                "ent": source_ent,
                "se": source_se,
                "qid": source_qid,
                "rid": source_rid,
            }
    return None


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    math_runs_path = args.math_runs or cfg.math_runs_path
    ood_runs_path = args.ood_runs or cfg.ood_runs_path
    math_hidden_path = args.math_hidden or cfg.math_hidden_path
    ood_hidden_path = args.ood_hidden or cfg.ood_hidden_path
    semantic_path = args.semantic or cfg.math_semantic_entropy_path
    out_dir = args.out_dir or cfg.probe_dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    math_runs = U.load_jsonl(math_runs_path)
    if not math_runs:
        print(f"No math runs at {math_runs_path}")
        return 0
    if not math_hidden_path.exists():
        raise FileNotFoundError(f"Hidden state file missing: {math_hidden_path}")
    hidden_map = _load_hidden(math_hidden_path)

    se_map: Dict[str, float] = {}
    for rec in U.load_jsonl(semantic_path):
        se_map[str(rec.get("question_id"))] = float(rec.get("semantic_entropy", float("nan")))

    split_map = _build_split_map([str(r["question_id"]) for r in math_runs], cfg.train_val_test_split, cfg.random_seed)

    store: Dict[str, List] = {}
    for rec in math_runs:
        qid = str(rec.get("question_id"))
        rid = int(rec.get("run_id", 0))
        hidden = hidden_map.get((qid, rid))
        if hidden is None:
            continue
        split = split_map.get(qid, "train")
        _append_example(store, split, qid, rid, hidden, rec, se_map)

    # Balance classes per split if possible by borrowing from train
    for split in ["val", "test"]:
        y_list = store.get(f"y_{split}", [])
        if not y_list:
            continue
        borrow = _ensure_two_classes(y_list, store, "train")
        if borrow:
            store.setdefault(f"X_{split}", []).append(borrow["X"])
            store.setdefault(f"entropy_{split}", []).append(borrow["ent"])
            store.setdefault(f"se_{split}", []).append(borrow["se"])
            store.setdefault(f"qid_{split}", []).append(borrow["qid"])
            store.setdefault(f"rid_{split}", []).append(borrow["rid"])

    _save_split(out_dir, "math_train", store.get("X_train", []), store.get("y_train", []), store.get("entropy_train", []), store.get("se_train", []), store.get("qid_train", []), store.get("rid_train", []))
    _save_split(out_dir, "math_val", store.get("X_val", []), store.get("y_val", []), store.get("entropy_val", []), store.get("se_val", []), store.get("qid_val", []), store.get("rid_val", []))
    _save_split(out_dir, "math_test", store.get("X_test", []), store.get("y_test", []), store.get("entropy_test", []), store.get("se_test", []), store.get("qid_test", []), store.get("rid_test", []))

    # OOD test set (no semantic entropy labels)
    if ood_runs_path.exists() and ood_hidden_path.exists():
        ood_runs = U.load_jsonl(ood_runs_path)
        ood_hidden = _load_hidden(ood_hidden_path)
        X, y, ent, qids, rids = [], [], [], [], []
        for rec in ood_runs:
            qid = str(rec.get("question_id"))
            rid = int(rec.get("run_id", 0))
            h = ood_hidden.get((qid, rid))
            if h is None:
                continue
            X.append(h)
            y.append(int(rec.get("is_correct", 0)))
            ent.append(float(rec.get("mean_think_entropy", float("nan"))))
            qids.append(qid)
            rids.append(rid)
        if X:
            out_path = out_dir / "ood_test.npz"
            np.savez_compressed(
                out_path,
                X_hidden=np.stack(X, axis=0),
                y_correct=np.asarray(y, dtype=np.int8),
                mean_think_entropy=np.asarray(ent, dtype=np.float32),
                question_ids=np.asarray(qids, dtype=object),
                run_ids=np.asarray(rids, dtype=np.int32),
            )
            print(f"Saved {len(X)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
