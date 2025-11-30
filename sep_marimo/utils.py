from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_think_token_ids(tokenizer, think_start: str, think_end: str) -> Tuple[List[int], List[int]]:
    start_ids = tokenizer.encode(think_start, add_special_tokens=False)
    end_ids = tokenizer.encode(think_end, add_special_tokens=False)
    return start_ids, end_ids


def find_subsequence(seq: Sequence[int], subseq: Sequence[int], start: int = 0) -> int:
    if not subseq:
        return -1
    for i in range(start, len(seq) - len(subseq) + 1):
        if list(seq[i : i + len(subseq)]) == list(subseq):
            return i
    return -1


def token_entropy_from_logprobs(logprob_entry: Any) -> Optional[float]:
    if logprob_entry is None:
        return None
    # vLLM returns dict[token_id -> LogProb] where value has .logprob
    if isinstance(logprob_entry, Mapping):
        vals = [getattr(v, "logprob", v) for v in logprob_entry.values()]
    elif isinstance(logprob_entry, Sequence):
        vals = [getattr(v, "logprob", v) for v in logprob_entry]
    else:
        return None
    if not vals:
        return None
    vals_arr = np.asarray(vals, dtype=np.float32)
    vals_arr = vals_arr - vals_arr.max()  # stability
    probs = np.exp(vals_arr)
    probs = probs / (probs.sum() + 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return entropy


def mean_entropy_in_span(entropies: Sequence[Optional[float]], start: int, end: int) -> float:
    vals = [e for e in entropies[start : end + 1] if e is not None and not math.isnan(e)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def extract_last_boxed(text: str) -> str:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return ""
    start_brace_idx = matches[-1].end() - 1
    start = start_brace_idx + 1
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].replace(" ", "").rstrip("\\%")
        i += 1
    return text[start:].strip()


def extract_think_and_answer(full_text: str, think_start: str, think_end: str) -> Tuple[str, str]:
    start_idx = full_text.find(think_start)
    end_idx = full_text.find(think_end, start_idx + len(think_start)) if start_idx != -1 else -1
    think_text = ""
    answer_text = full_text
    if start_idx != -1 and end_idx != -1:
        think_text = full_text[start_idx + len(think_start) : end_idx].strip()
        answer_text = full_text[end_idx + len(think_end) :].strip()
    elif end_idx != -1:
        answer_text = full_text[end_idx + len(think_end) :].strip()
    return think_text, answer_text


def extract_answer_text(full_text: str) -> str:
    boxed = extract_last_boxed(full_text)
    if boxed:
        return boxed
    after = full_text
    if "</think>" in full_text:
        after = full_text.split("</think>", 1)[1]
    match = re.search(r"(?i)answer\s*[:：]\s*(.+)", after)
    if match:
        return match.group(1).strip()
    lines = [ln.strip() for ln in after.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return after.strip()


def _maybe_import_math_verify():
    try:
        from math_verify import parse, verify  # type: ignore
    except Exception:
        return None
    return parse, verify


def _normalize_str(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def _normalize_numeric(text: str) -> Optional[float]:
    try:
        cleaned = text.replace(",", "").replace("\\", "").strip()
        return float(cleaned)
    except Exception:
        return None


def answers_equivalent(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    mv = _maybe_import_math_verify()
    if mv is not None:
        parse, verify = mv
        try:
            lhs = parse(r"\\boxed{" + gold + "}")
            rhs = parse(r"\\boxed{" + pred + "}")
            return bool(verify(lhs, rhs))
        except Exception:
            pass
    pred_num = _normalize_numeric(pred)
    gold_num = _normalize_numeric(gold)
    if pred_num is not None and gold_num is not None:
        if math.isclose(pred_num, gold_num, rel_tol=1e-4, abs_tol=1e-4):
            return True
    return _normalize_str(pred) == _normalize_str(gold)


def open_qa_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    strip = lambda x: re.sub(r"[^a-z0-9]+", "", x.lower())
    return strip(pred) == strip(gold)


def normalize_answer_for_cluster(answer: str) -> str:
    if not answer:
        return ""
    boxed = extract_last_boxed(answer)
    if boxed:
        answer = boxed
    answer = answer.strip().lower()
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"[^0-9a-zA-Z.+\-/ ]", "", answer)
    return answer.strip()


def compute_is_correct(answer_text: str, gold: str, dataset: str) -> bool:
    if dataset == "math":
        return answers_equivalent(answer_text, gold)
    return open_qa_match(answer_text, gold)


def collate_runs_by_question(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["question_id"])].append(rec)
    return grouped


def build_prompt(question: str, think_start: str, think_end: str) -> str:
    return (
        "You are a helpful math reasoning assistant.\n"
        "Solve the following problem step-by-step. Wrap your reasoning inside the"
        f" {think_start} ... {think_end} tags and give a short final answer after {think_end}.\n\n"
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        f"{think_start}\n"
    )
