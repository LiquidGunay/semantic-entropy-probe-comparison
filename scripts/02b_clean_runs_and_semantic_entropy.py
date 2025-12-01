#!/usr/bin/env python3
"""Clean runs (drop truncated / think-missing), recompute answers, and semantic entropy.

Outputs:
- data/math_runs_clean.jsonl
- data/math_semantic_entropy_clean.jsonl

Notes:
- Does not modify original step-1 data. Safe to rerun.
- Drops runs without closing </think> or with think_token_len >= 8192 (likely truncation).
- Drops questions that end up with fewer than 4 valid runs.
- final_answer uses boxed content within the full answer; full_answer is everything after </think>.
- Semantic entropy is computed over full_answer strings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sep_marimo.config import DEFAULT_CONFIG
from sep_marimo import utils as U

MIN_RUNS_PER_QUESTION = 4
TRUNC_THINK_TOKENS = 8192


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-path", type=Path, default=None, help="Input runs JSONL (math_runs.jsonl)")
    p.add_argument("--raw-path", type=Path, default=None, help="math_raw.jsonl with gold answers")
    p.add_argument("--out-runs", type=Path, default=None, help="Output cleaned runs JSONL")
    p.add_argument("--out-sentropy", type=Path, default=None, help="Output semantic entropy JSONL")
    return p.parse_args()


def load_gold(raw_path: Path) -> Dict[str, Dict]:
    gold = {}
    if raw_path.exists():
        for rec in U.load_jsonl(raw_path):
            gold[str(rec.get("id"))] = rec
    return gold


def run_is_valid(rec: Dict) -> bool:
    text = rec.get("output_text", "")
    think_len = rec.get("think_token_len", 0)
    has_end = "</think>" in text
    if not has_end:
        return False
    if think_len and think_len >= TRUNC_THINK_TOKENS:
        return False
    return True


def rewrite_run(rec: Dict, gold_map: Dict[str, Dict]) -> Dict:
    out = dict(rec)
    full_ans = U.extract_full_answer(rec.get("output_text", ""))
    final_ans = U.extract_final_answer(full_ans)
    out["full_answer_text"] = full_ans
    out["final_answer_text"] = final_ans
    qid = str(rec.get("question_id"))
    gold = gold_map.get(qid, {})
    gold_answer = gold.get("answer", "")
    dataset = rec.get("dataset", "math")
    out["is_correct"] = U.compute_is_correct(final_ans, gold_answer, dataset)
    return out


def compute_semantic_entropy(clean_runs: List[Dict]) -> List[Dict]:
    grouped = U.collate_runs_by_question(clean_runs)
    results: List[Dict] = []
    for qid, recs in grouped.items():
        answers = []
        clusters = {}
        for rec in recs:
            rid = int(rec.get("run_id", 0))
            full_ans = rec.get("full_answer_text", "")
            norm = U.normalize_answer_for_cluster(full_ans)
            clusters.setdefault(norm, []).append(rid)
            answers.append({"run_id": rid, "full_answer_text": full_ans})
        total = sum(len(v) for v in clusters.values())
        probs = [len(v) / total for v in clusters.values() if total > 0]
        sentropy = float("nan") if not probs else float(-(U.np.log(probs) * probs).sum())
        results.append(
            {
                "question_id": qid,
                "answers": answers,
                "clusters": [
                    {"cluster_id": idx, "member_run_ids": members}
                    for idx, members in enumerate(clusters.values())
                ],
                "cluster_probs": probs,
                "semantic_entropy": sentropy,
            }
        )
    return results


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    runs_path = args.runs_path or cfg.math_runs_path
    raw_path = args.raw_path or cfg.math_raw_path
    out_runs = args.out_runs or (cfg.data_dir / "math_runs_clean.jsonl")
    out_sentropy = args.out_sentropy or (cfg.data_dir / "math_semantic_entropy_clean.jsonl")

    runs = U.load_jsonl(runs_path)
    gold_map = load_gold(raw_path)

    valid = [r for r in runs if run_is_valid(r)]
    by_q = U.collate_runs_by_question(valid)
    kept: List[Dict] = []
    dropped_qids = 0
    for qid, recs in by_q.items():
        if len(recs) < MIN_RUNS_PER_QUESTION:
            dropped_qids += 1
            continue
        for rec in recs:
            kept.append(rewrite_run(rec, gold_map))

    out_runs.parent.mkdir(parents=True, exist_ok=True)
    U.write_jsonl(out_runs, kept)
    print(f"Kept {len(kept)} runs across {len(U.collate_runs_by_question(kept))} questions (dropped {dropped_qids} qids)")

    sent = compute_semantic_entropy(kept)
    out_sentropy.parent.mkdir(parents=True, exist_ok=True)
    with out_sentropy.open("w", encoding="utf-8") as f:
        for rec in sent:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote semantic entropy for {len(sent)} questions to {out_sentropy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
