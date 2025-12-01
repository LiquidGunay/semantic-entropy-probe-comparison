#!/usr/bin/env python3
"""Generate multiple runs per question with vLLM, storing run-level metadata.

The script expects JSONL inputs with fields: {"id", "question", "answer"} for
both math and OOD sets. It writes one JSON line per run with token/entropy info.

Defaults are in `sep_marimo.config.ExperimentConfig`; override via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sep_marimo.config import ExperimentConfig, DEFAULT_CONFIG
from sep_marimo import utils as U


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_CONFIG.model_name, help="HF model name for vLLM")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="Generation backend")
    p.add_argument("--math-path", type=Path, default=None, help="Path to math_raw.jsonl")
    p.add_argument("--ood-path", type=Path, default=None, help="Path to ood_raw.jsonl")
    p.add_argument("--runs", type=int, default=DEFAULT_CONFIG.num_runs_per_question, help="Runs per question")
    p.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.temperature)
    p.add_argument("--top-p", type=float, default=DEFAULT_CONFIG.top_p)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_CONFIG.max_new_tokens)
    p.add_argument("--top-k-logprobs", type=int, default=DEFAULT_CONFIG.top_k_for_entropy)
    p.add_argument("--dtype", default="bfloat16", help="Model dtype for vLLM")
    p.add_argument("--max-model-len", type=int, default=None, help="Override model max sequence length for vLLM")
    p.add_argument("--tensor-parallel", type=int, default=1, help="vLLM tensor parallel size")
    p.add_argument("--gpu-mem-utilization", type=float, default=0.85, help="Fraction of GPU memory to use (vLLM gpu_memory_utilization)")
    p.add_argument("--max-num-seqs", type=int, default=1, help="Limit concurrent sequences for small GPUs")
    p.add_argument("--batch-size", type=int, default=8, help="Prompts per vLLM generate() call")
    p.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile in vLLM (useful on small GPUs)")
    p.add_argument("--out-math", type=Path, default=None, help="Override math_runs.jsonl output")
    p.add_argument("--out-ood", type=Path, default=None, help="Override ood_runs.jsonl output")
    p.add_argument("--limit", type=int, default=None, help="Optional question cap for smoke tests")
    return p.parse_args()


def _prepare_output_paths(cfg: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Path]:
    math_out = args.out_math or cfg.math_runs_path
    ood_out = args.out_ood or cfg.ood_runs_path
    cfg.ensure_dirs()
    return {"math": math_out, "ood": ood_out}


def _torch_dtype(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _load_dataset(path: Path, label: str) -> List[Dict]:
    if not path.exists():
        print(f"[warn] {label} dataset missing at {path}; skipping")
        return []
    return U.load_jsonl(path)


def _find_last_subseq(seq: List[int], subseq: List[int], start: int = 0) -> int:
    """Return last index where subseq appears in seq[start:], or -1."""
    last = -1
    if not subseq:
        return last
    for i in range(start, len(seq) - len(subseq) + 1):
        if seq[i : i + len(subseq)] == subseq:
            last = i
    return last


def _batched(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    batch_size = max(1, batch_size)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> int:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    U.seed_everything(cfg.random_seed)

    math_rows = _load_dataset(args.math_path or cfg.math_raw_path, "math")
    ood_rows = _load_dataset(args.ood_path or cfg.ood_raw_path, "ood")

    if not math_rows and not ood_rows:
        print("No input datasets found; nothing to do", file=sys.stderr)
        return 0

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    think_start_ids, think_end_ids = U.get_think_token_ids(
        tokenizer, cfg.think_start, cfg.think_end
    )
    prompt_template = lambda q: U.build_prompt(q, cfg.think_start, cfg.think_end)

    llm = None
    sampling = None
    if args.backend == "vllm":
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "vllm backend selected but vllm is not installed. Install with `pip install vllm`."
            ) from exc

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.runs,
            logprobs=args.top_k_logprobs,
        )
        # Let vLLM load the tokenizer internally; we only used the HF tokenizer
        # above to locate think-tag token IDs.
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            tensor_parallel_size=args.tensor_parallel,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_utilization,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            trust_remote_code=True,
        )

    outputs = _prepare_output_paths(cfg, args)

    def run_block(rows: List[Dict], dataset: str, out_path: Path) -> None:
        if not rows:
            return
        if args.limit:
            rows = rows[: args.limit]
        records = []
        if args.backend == "vllm":
            for batch in _batched(rows, args.batch_size):
                prompts = []
                meta = []
                for row in batch:
                    question_id = row.get("id") or row.get("question_id")
                    question = row.get("question", "")
                    gold_answer = row.get("answer", "")
                    prompts.append(prompt_template(question))
                    meta.append((question_id, gold_answer))
                generations = llm.generate(prompts, sampling_params=sampling)
                for (question_id, gold_answer), prompt, req in zip(meta, prompts, generations):
                    prompt_tokens = req.prompt_token_ids
                    for run_id, out in enumerate(req.outputs):
                        full_tokens = list(prompt_tokens) + list(out.token_ids)
                        full_text = prompt + out.text
                        think_text, answer_block = U.extract_think_and_answer(
                            full_text, cfg.think_start, cfg.think_end
                        )
                        answer_text = U.extract_answer_text(answer_block)
                        token_entropies = [None] * len(prompt_tokens)
                        for lp in out.logprobs:
                            token_entropies.append(U.token_entropy_from_logprobs(lp))
                        start_search = len(prompt_tokens)
                        think_start_idx = _find_last_subseq(full_tokens, think_start_ids, start=start_search)
                        if think_start_idx == -1:
                            think_start_idx = len(prompt_tokens)
                        think_end_idx = _find_last_subseq(full_tokens, think_end_ids, start=think_start_idx + 1)
                        if think_end_idx == -1 or think_end_idx < think_start_idx:
                            think_end_idx = len(full_tokens) - 1
                        mean_think_entropy = U.mean_entropy_in_span(
                            token_entropies, think_start_idx, think_end_idx
                        )
                        is_correct = U.compute_is_correct(answer_text, gold_answer, dataset)
                        records.append(
                            {
                                "dataset": dataset,
                                "question_id": question_id,
                                "run_id": run_id,
                                "prompt": prompt,
                                "output_text": full_text,
                                "think_text": think_text,
                                "answer_text": answer_text,
                                "tokens": full_tokens,
                                "think_token_indices": [think_start_idx, think_end_idx],
                                "token_entropies": token_entropies,
                                "mean_think_entropy": mean_think_entropy,
                                "is_correct": is_correct,
                            }
                        )
        else:  # HF fallback
            torch_dtype = _torch_dtype(args.dtype)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            model.eval()
            for row in rows:
                question_id = row.get("id") or row.get("question_id")
                question = row.get("question", "")
                gold_answer = row.get("answer", "")
                prompt = prompt_template(question)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
                gen_out = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.runs,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                for run_id in range(args.runs):
                    seq = gen_out.sequences[run_id].tolist()
                    scores = gen_out.scores
                    gen_len = len(seq) - input_ids.shape[1]
                    token_entropies = [None] * input_ids.shape[1]
                    for i in range(gen_len):
                        logits = scores[i][run_id]
                        probs = torch.softmax(logits, dim=-1)
                        entropy = float(-(probs * torch.log(probs + 1e-12)).sum().cpu())
                        token_entropies.append(entropy)
                    full_tokens = seq
                    gen_text = tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=False)
                    full_text = prompt + gen_text
                    think_text, answer_block = U.extract_think_and_answer(
                        full_text, cfg.think_start, cfg.think_end
                    )
                    answer_text = U.extract_answer_text(answer_block)
                    start_search = len(input_ids[0])
                    think_start_idx = _find_last_subseq(full_tokens, think_start_ids, start=start_search)
                    if think_start_idx == -1:
                        think_start_idx = len(input_ids[0])
                    think_end_idx = _find_last_subseq(full_tokens, think_end_ids, start=think_start_idx + 1)
                    if think_end_idx == -1 or think_end_idx < think_start_idx:
                        think_end_idx = len(full_tokens) - 1
                    mean_think_entropy = U.mean_entropy_in_span(
                        token_entropies, think_start_idx, think_end_idx
                    )
                    is_correct = U.compute_is_correct(answer_text, gold_answer, dataset)
                    records.append(
                        {
                            "dataset": dataset,
                            "question_id": question_id,
                            "run_id": run_id,
                            "prompt": prompt,
                            "output_text": full_text,
                            "think_text": think_text,
                            "answer_text": answer_text,
                            "tokens": full_tokens,
                            "think_token_indices": [think_start_idx, think_end_idx],
                            "token_entropies": token_entropies,
                            "mean_think_entropy": mean_think_entropy,
                            "is_correct": is_correct,
                        }
                    )

        U.write_jsonl(out_path, records)
        print(f"[{dataset}] wrote {len(records)} runs to {out_path}")

    run_block(math_rows, "math", outputs["math"])
    run_block(ood_rows, "ood", outputs["ood"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
