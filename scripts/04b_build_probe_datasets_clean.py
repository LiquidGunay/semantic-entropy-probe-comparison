#!/usr/bin/env python3
"""Build cleaned probe datasets from cleaned runs, hidden states, and semantic entropy.

Inputs (defaults):
- data/math_runs_clean.jsonl (from 02b)
- data/math_semantic_entropy_clean.jsonl (from 02b)
- artifacts/math_hidden_states.npz (existing hidden states)

Outputs:
- artifacts_clean/probe_datasets/math_{train,val,test}.npz with keys X_hidden, y_correct,
  mean_think_entropy, semantic_entropy, question_ids, run_ids
"""

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
    p.add_argument("--runs", type=Path, default=None, help="Cleaned runs JSONL")
    p.add_argument("--semantic", type=Path, default=None, help="Cleaned semantic entropy JSONL")
    p.add_argument("--hidden", type=Path, default=None, help="Hidden states NPZ")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts_clean/probe_datasets"))
    return p.parse_args()


def _load_hidden(npz_path: Path) -> Dict[Tuple[str, int], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    mapping: Dict[Tuple[str, int], np.ndarray] = {}
    for qid, rid, vec in zip(data["question_ids"], data["run_ids"], data["hidden_states"]):
        mapping[(str(qid), int(rid))] = np.asarray(vec, dtype=np.float32)
    return mapping


def _semantic_map(path: Path) -> Dict[str, float]:
    sem = {}
    for rec in U.load_jsonl(path):
        sem[str(rec["question_id"])] = float(rec.get("semantic_entropy", float("nan")))
    return sem


def _split_questions(question_ids: List[str], split: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    qids = list(question_ids)
    rng.shuffle(qids)
    n = len(qids)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    split_map = {}
    for i, qid in enumerate(qids):
        if i < n_train:
            split_map[qid] = "train"
        elif i < n_train + n_val:
            split_map[qid] = "val"
        else:
            split_map[qid] = "test"
    return split_map


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    runs_path = args.runs or Path("data/math_runs_clean.jsonl")
    sem_path = args.semantic or Path("data/math_semantic_entropy_clean.jsonl")
    hidden_path = args.hidden or Path("artifacts/math_hidden_states.npz")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = U.load_jsonl(runs_path)
    hidden_map = _load_hidden(hidden_path)
    sem_map = _semantic_map(sem_path)

    # gather by question
    by_q = U.collate_runs_by_question(runs)
    split_map = _split_questions(list(by_q.keys()), cfg.train_val_test_split, cfg.random_seed)

    buckets = {"train": [], "val": [], "test": []}
    for qid, recs in by_q.items():
        split = split_map.get(qid, "train")
        for rec in recs:
            key = (str(rec.get("question_id")), int(rec.get("run_id", 0)))
            if key not in hidden_map:
                continue
            vec = hidden_map[key]
            buckets[split].append(
                {
                    "X": vec,
                    "y": int(bool(rec.get("is_correct", False))),
                    "ent": float(rec.get("mean_think_entropy", float("nan"))),
                    "sem": float(sem_map.get(str(rec.get("question_id")), float("nan"))),
                    "qid": key[0],
                    "rid": key[1],
                }
            )

    for split, rows in buckets.items():
        if not rows:
            print(f"[warn] no rows for split {split}")
            continue
        X = np.stack([r["X"] for r in rows]).astype(np.float32)
        y = np.asarray([r["y"] for r in rows], dtype=np.int64)
        ent = np.asarray([r["ent"] for r in rows], dtype=np.float32)
        sem = np.asarray([r["sem"] for r in rows], dtype=np.float32)
        qids = np.asarray([r["qid"] for r in rows], dtype=object)
        rids = np.asarray([r["rid"] for r in rows], dtype=np.int32)
        out_path = out_dir / f"math_{split}.npz"
        np.savez_compressed(
            out_path,
            X_hidden=X,
            y_correct=y,
            mean_think_entropy=ent,
            semantic_entropy=sem,
            question_ids=qids,
            run_ids=rids,
        )
        print(f"[{split}] wrote {out_path} with {len(rows)} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
