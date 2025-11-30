#!/usr/bin/env python3
"""Replay tokenized runs to extract hidden states at </think> tokens."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sep_marimo.config import DEFAULT_CONFIG
from sep_marimo import utils as U


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_CONFIG.model_name, help="HF model name")
    p.add_argument("--math-runs", type=Path, default=None, help="math_runs.jsonl path")
    p.add_argument("--ood-runs", type=Path, default=None, help="ood_runs.jsonl path")
    p.add_argument("--out-math", type=Path, default=None, help="Output npz for math")
    p.add_argument("--out-ood", type=Path, default=None, help="Output npz for ood")
    p.add_argument("--device", default=None, help="torch device override (e.g., cuda:0)")
    p.add_argument("--dtype", default="bfloat16", help="torch dtype for model")
    p.add_argument(
        "--backend",
        choices=["hf", "nnsight"],
        default="hf",
        help="Use plain HF forward pass or nnsight trace",
    )
    return p.parse_args()


def _as_torch_dtype(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _load_model(model_name: str, device: str | None, dtype_name: str):
    dtype = _as_torch_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device is None else None,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def _trace_hf(model, tokens: List[int], idx: int, device: torch.device):
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, idx, :].detach().cpu().float().numpy()
    return hidden


def _trace_nnsight(llm, tokens: List[int], idx: int):  # pragma: no cover - optional path
    import torch as _torch

    toks = _torch.tensor(tokens, device=llm.device).unsqueeze(0)
    with llm.trace(toks) as tracer:
        hidden = llm.model.layers[-1].output.save()
    return hidden[0, idx, :].detach().cpu().float().numpy()


def _process(runs_path: Path, out_path: Path, model, backend: str, device) -> None:
    runs = U.load_jsonl(runs_path)
    if not runs:
        return
    question_ids: List[str] = []
    run_ids: List[int] = []
    hidden_states: List[np.ndarray] = []
    for rec in runs:
        tokens: List[int] = rec.get("tokens", [])
        think_span: List[int] = rec.get("think_token_indices", [-1, -1])
        if not tokens or think_span[1] >= len(tokens) or think_span[1] < 0:
            continue
        if backend == "hf":
            h = _trace_hf(model, tokens, think_span[1], device)
        else:
            h = _trace_nnsight(model, tokens, think_span[1])
        question_ids.append(str(rec.get("question_id")))
        run_ids.append(int(rec.get("run_id", len(run_ids))))
        hidden_states.append(h)
    if not hidden_states:
        print(f"[warn] no hidden states extracted from {runs_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        question_ids=np.asarray(question_ids, dtype=object),
        run_ids=np.asarray(run_ids, dtype=np.int32),
        hidden_states=np.stack(hidden_states, axis=0),
    )
    print(f"Saved {len(hidden_states)} hidden vectors to {out_path}")


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    math_runs = args.math_runs or cfg.math_runs_path
    ood_runs = args.ood_runs or cfg.ood_runs_path
    out_math = args.out_math or cfg.math_hidden_path
    out_ood = args.out_ood or cfg.ood_hidden_path

    if args.backend == "nnsight":  # pragma: no cover - requires nnsight runtime
        from nnsight import LanguageModel

        llm = LanguageModel(args.model, trust_remote_code=True)
        device = llm.device
        model_obj = llm
        tokenizer = llm.tokenizer
    else:
        model_obj, tokenizer = _load_model(args.model, args.device, args.dtype)
        device = next(model_obj.parameters()).device

    if math_runs.exists():
        _process(math_runs, out_math, model_obj, args.backend, device)
    else:
        print(f"[info] math runs not found at {math_runs}, skipping")

    if ood_runs.exists():
        _process(ood_runs, out_ood, model_obj, args.backend, device)
    else:
        print(f"[info] ood runs not found at {ood_runs}, skipping")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
