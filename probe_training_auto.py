import marimo

__generated_with = "0.7.13"
app = marimo.App(width="wide")


@app.cell
def __():
    import marimo as mo

    mo.md(
        """
        # Semantic Entropy Probes (GPT-2 + nnsight, marimo)

        This notebook approximates the method from *Semantic Entropy: Reliable Uncertainty from Language Models* (arXiv:2406.15927) using a tiny GPT-2 model, nnsight hooks, and a lightweight GSM8K subset. Heavy steps are gated behind run buttons to stay within a 4 GB VRAM budget. Configuration controls live below.
        """
    )
    return mo


@app.cell
def __(mo):
    import json
    import math
    import os
    import random
    from collections import defaultdict
    from functools import lru_cache
    from pathlib import Path
    from typing import Dict, Iterable, List, Tuple

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import torch
    from datasets import load_dataset
    from scipy import stats
    from sentence_transformers import SentenceTransformer
    from torch import nn
    from torch.nn import functional as F
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from nnsight import LanguageModel

    return (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Dict,
        F,
        Iterable,
        LanguageModel,
        List,
        Path,
        SentenceTransformer,
        Tuple,
        defaultdict,
        json,
        load_dataset,
        lru_cache,
        math,
        nn,
        np,
        os,
        pd,
        px,
        random,
        stats,
        torch,
        tqdm,
    )


@app.cell
def __(mo):
    k_slider = mo.ui.slider(2, 20, value=10, step=1, label="Samples per prompt (K)")
    temp_slider = mo.ui.slider(0.2, 1.5, value=0.7, step=0.05, label="Temperature")
    top_p_slider = mo.ui.slider(0.2, 1.0, value=0.9, step=0.05, label="Top-p")
    max_tokens_slider = mo.ui.slider(8, 96, value=48, step=8, label="Max new tokens")
    dataset_size_slider = mo.ui.slider(10, 200, value=40, step=10, label="Dataset size (GSM8K train)")
    threshold_slider = mo.ui.slider(0.3, 0.9, value=0.6, step=0.05, label="NLI entailment threshold")
    layer_slider = mo.ui.slider(0, 11, value=11, step=1, label="GPT-2 layer for probe (0=lowest)")
    seed_box = mo.ui.number(0, 10_000, value=42, step=1, label="Random seed")
    semantic_mode = mo.ui.dropdown(
        options=["nli", "cosine"],
        value="nli",
        label="Semantic equivalence scorer",
    )
    sample_btn = mo.ui.run_button(label="Generate samples & entropies")
    probe_btn = mo.ui.run_button(label="Train probe")

    mo.vstack(
        [
            mo.hstack([k_slider, temp_slider, top_p_slider, max_tokens_slider]),
            mo.hstack([dataset_size_slider, threshold_slider, layer_slider, seed_box]),
            mo.hstack([semantic_mode, sample_btn, probe_btn]),
        ]
    )
    return (
        dataset_size_slider,
        k_slider,
        layer_slider,
        max_tokens_slider,
        probe_btn,
        sample_btn,
        seed_box,
        semantic_mode,
        temp_slider,
        threshold_slider,
        top_p_slider,
    )


@app.cell
def __(
    dataset_size_slider,
    k_slider,
    layer_slider,
    max_tokens_slider,
    mo,
    seed_box,
    semantic_mode,
    temp_slider,
    threshold_slider,
    top_p_slider,
    torch,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "lm_name": "openai-community/gpt2",
        "nli_model": "MoritzLaurer/deberta-v3-small-mnli",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": device,
        "dataset_size": int(dataset_size_slider.value),
        "num_samples": int(k_slider.value),
        "temperature": float(temp_slider.value),
        "top_p": float(top_p_slider.value),
        "max_new_tokens": int(max_tokens_slider.value),
        "entailment_threshold": float(threshold_slider.value),
        "probe_layer": int(layer_slider.value),
        "seed": int(seed_box.value),
        "semantic_mode": semantic_mode.value,
    }
    mo.md(
        f"**Device:** {device} | **LM:** {config['lm_name']} | **NLI:** {config['nli_model']} | **K:** {config['num_samples']} | **Max new:** {config['max_new_tokens']}"
    )
    return config, device


@app.cell
def __(Path):
    cache_dir = Path("cache")
    outputs_dir = Path("outputs")
    cache_dir.mkdir(exist_ok=True, parents=True)
    outputs_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir, outputs_dir


@app.cell
def __(
    Dict,
    Iterable,
    List,
    Tuple,
    json,
    math,
    np,
    os,
    random,
    torch,
):
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def entropy_from_logprobs(logprobs: List[float]) -> float:
        """Compute entropy over unnormalized log-probabilities."""
        if len(logprobs) == 0:
            return float("nan")
        log_probs = torch.tensor(logprobs)
        norm = log_probs - torch.logsumexp(log_probs, dim=0)
        probs = torch.exp(norm)
        entropy = -(probs * norm).sum().item()
        return entropy

    def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in batch.items()}

    def save_json(path: str | os.PathLike, obj: Dict) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def load_json(path: str | os.PathLike) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    return entropy_from_logprobs, load_json, save_json, set_seed, to_device


@app.cell
def __(
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LanguageModel,
    SentenceTransformer,
    lru_cache,
    torch,
):
    @lru_cache(maxsize=1)
    def get_generation_model(model_name: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        model.eval()
        return tokenizer, model

    @lru_cache(maxsize=1)
    def get_nli_model(model_name: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        entailment_idx = 2  # MNLI label order: contradiction, neutral, entailment
        return tokenizer, model, entailment_idx

    @lru_cache(maxsize=1)
    def get_embed_model(model_name: str):
        return SentenceTransformer(model_name)

    @lru_cache(maxsize=1)
    def get_nnsight_model(model_name: str, device: str):
        # Device map is passed through to nnsight for tracing.
        return LanguageModel(model_name, device_map=device)

    return (
        get_embed_model,
        get_generation_model,
        get_nli_model,
        get_nnsight_model,
    )


@app.cell
def __(config, load_dataset, pd, set_seed):
    set_seed(config["seed"])
    ds = load_dataset("gsm8k", "main", split="train")
    subset = ds.shuffle(seed=config["seed"]).select(range(config["dataset_size"]))

    def format_prompt(row):
        return f"Question: {row['question'].strip()}\nAnswer:"

    records = []
    for idx, row in enumerate(subset):
        prompt = format_prompt(row)
        answer_text = row["answer"].split("####")[0].strip()
        records.append(
            {
                "prompt_id": idx,
                "question": row["question"],
                "answer_raw": row["answer"],
                "reference_answer": answer_text,
                "prompt": prompt,
            }
        )
    prompt_df = pd.DataFrame(records)
    prompt_df
    return prompt_df


@app.cell
def __(entropy_from_logprobs, torch):
    def logprobs_from_generate(outputs, prompt_len: int) -> torch.Tensor:
        """Compute sequence log-prob for each generated sample from generate() output."""
        scores = outputs.scores  # list length = generated tokens
        sequences = outputs.sequences
        batch = sequences.shape[0]
        logps = torch.zeros(batch, device=sequences.device, dtype=torch.float32)
        for t, score_t in enumerate(scores):
            logp_t = torch.log_softmax(score_t, dim=-1)
            token_ids = sequences[:, prompt_len + t]
            logps += logp_t[torch.arange(batch), token_ids]
        return logps

    def lexical_entropy(logprobs: torch.Tensor) -> float:
        return entropy_from_logprobs(logprobs.tolist())

    return lexical_entropy, logprobs_from_generate


@app.cell
def __(
    config,
    lexical_entropy,
    logprobs_from_generate,
    mo,
    pd,
    prompt_df,
    sample_btn,
    to_device,
    torch,
    tqdm,
    get_generation_model,
):
    if not sample_btn.value:
        mo.md("Click **Generate samples & entropies** to run sampling.")
        return None

    tokenizer, model = get_generation_model(config["lm_name"], config["device"])
    rows = []
    for row in tqdm(prompt_df.itertuples(), total=len(prompt_df)):
        inputs = tokenizer(
            row.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        inputs = to_device(inputs, config["device"])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_new_tokens=config["max_new_tokens"],
                num_return_sequences=config["num_samples"],
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        logps = logprobs_from_generate(outputs, prompt_len)
        sequences = outputs.sequences
        lex_ent = lexical_entropy(logps)
        for idx in range(sequences.shape[0]):
            completion_ids = sequences[idx, prompt_len:]
            completion_text = tokenizer.decode(
                completion_ids, skip_special_tokens=True
            ).strip()
            rows.append(
                {
                    "prompt_id": row.prompt_id,
                    "completion_id": idx,
                    "question": row.question,
                    "prompt": row.prompt,
                    "reference_answer": row.reference_answer,
                    "completion": completion_text,
                    "logprob": float(logps[idx].item()),
                    "lexical_entropy": lex_ent,
                }
            )

    sample_df = pd.DataFrame(rows)
    sample_df
    return sample_df


@app.cell
def __(
    SentenceTransformer,
    config,
    entropy_from_logprobs,
    get_embed_model,
    get_nli_model,
    np,
    torch,
):
    def nli_handles():
        tokenizer, model, entailment_idx = get_nli_model(
            config["nli_model"], device="cpu"
        )
        return tokenizer, model, entailment_idx, "cpu"

    def entailment_prob(
        premise: str, hypothesis: str, handles
    ) -> float:
        tokenizer, model, entailment_idx, device = handles
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, entailment_idx].item())

    def mutual_entails(a: str, b: str, handles, threshold: float) -> bool:
        if a.strip() == b.strip():
            return True
        pa = entailment_prob(a, b, handles)
        pb = entailment_prob(b, a, handles)
        return pa >= threshold and pb >= threshold

    def cosine_equiv_matrix(texts: list[str]) -> np.ndarray:
        embedder: SentenceTransformer = get_embed_model(config["embed_model"])
        embs = embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        embs = torch.nn.functional.normalize(embs, dim=-1)
        sim = (embs @ embs.T).cpu().numpy()
        return sim

    def cluster_equivalence(
        texts: list[str], logprobs: list[float], mode: str, threshold: float
    ):
        if mode == "nli":
            handles = nli_handles()

            def equivalent(i: int, j: int) -> bool:
                return mutual_entails(texts[i], texts[j], handles, threshold)

        else:
            sims = cosine_equiv_matrix(texts)

            def equivalent(i: int, j: int) -> bool:
                return sims[i, j] >= threshold

        clusters: list[list[int]] = []
        for idx, _ in enumerate(texts):
            placed = False
            for cluster in clusters:
                rep = cluster[0]
                if equivalent(idx, rep):
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])

        cluster_logweights = []
        for cluster in clusters:
            cluster_logweights.append(
                float(
                    torch.logsumexp(
                        torch.tensor([logprobs[i] for i in cluster]), dim=0
                    ).item()
                )
            )
        sem_entropy = entropy_from_logprobs(cluster_logweights)
        return clusters, sem_entropy

    return cluster_equivalence


@app.cell
def __(
    cluster_equivalence,
    config,
    mo,
    pd,
    sample_btn,
    sample_df,
):
    if sample_df is None or not sample_btn.value:
        mo.md("Semantic entropy pending sampling.")
        return None, None

    rows = []
    for prompt_id, group in sample_df.groupby("prompt_id"):
        texts = group["completion"].tolist()
        logps = group["logprob"].tolist()
        clusters, sem_entropy = cluster_equivalence(
            texts, logps, config["semantic_mode"], config["entailment_threshold"]
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "semantic_entropy": sem_entropy,
                "lexical_entropy": group["lexical_entropy"].iloc[0],
                "num_clusters": len(clusters),
                "num_samples": len(texts),
            }
        )
    se_df = pd.DataFrame(rows)
    se_table = se_df.merge(
        sample_df[
            ["prompt_id", "prompt", "question", "reference_answer"]
        ].drop_duplicates(),
        on="prompt_id",
        how="left",
    )
    se_table.sort_values("semantic_entropy", ascending=False, inplace=True)
    se_table
    return se_table, se_df


@app.cell
def __(
    config,
    get_nnsight_model,
    mo,
    np,
    pd,
    probe_btn,
    sample_btn,
    se_table,
    torch,
):
    if se_table is None or not sample_btn.value:
        mo.md("Run sampling first to collect prompts and entropies.")
        return None, None
    if not probe_btn.value:
        mo.md("Click **Train probe** to capture hidden states.")
        return None, None

    lm = get_nnsight_model(config["lm_name"], config["device"])
    features = []
    for row in se_table.itertuples():
        with lm.trace(row.prompt):
            hidden = lm.transformer.h[config["probe_layer"]].output[0].save()
        features.append(hidden[:, -1, :].detach().cpu().numpy()[0])

    X = np.stack(features)
    y = se_table["semantic_entropy"].to_numpy(dtype=np.float32)
    lexical = se_table["lexical_entropy"].to_numpy(dtype=np.float32)
    meta = se_table[["prompt_id", "question", "reference_answer", "prompt"]]
    mo.md(f"Captured hidden states shape: {X.shape}")
    return X, y, lexical, meta


@app.cell
def __(F, X, lexical, meta, mo, nn, np, pd, probe_btn, stats, torch, y):
    if X is None or not probe_btn.value:
        mo.md("Probe not trained yet.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, device=device)
    y_tensor = torch.tensor(y, device=device)

    class LinearProbe(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.head = nn.Linear(dim, 1)

        def forward(self, x):
            return self.head(x).squeeze(-1)

    probe = LinearProbe(X_tensor.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=5e-4)
    num_epochs = 200
    for epoch in range(num_epochs):
        opt.zero_grad()
        preds = probe(X_tensor)
        loss = F.mse_loss(preds, y_tensor)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = probe(X_tensor).detach().cpu().numpy()

    rmse = float(np.sqrt(((preds - y) ** 2).mean()))
    mae = float(np.abs(preds - y).mean())
    spearman = float(stats.spearmanr(preds, y).correlation)
    lex_rmse = float(np.sqrt(((lexical - y) ** 2).mean()))
    lex_mae = float(np.abs(lexical - y).mean())

    metrics = pd.DataFrame(
        [
            {"metric": "RMSE", "probe": rmse, "lexical_baseline": lex_rmse},
            {"metric": "MAE", "probe": mae, "lexical_baseline": lex_mae},
            {"metric": "Spearman", "probe": spearman, "lexical_baseline": np.nan},
        ]
    )

    scatter = pd.DataFrame({"true_SE": y, "pred_SE": preds})
    fig = mo.ui.plotly(
        px.scatter(
            scatter,
            x="true_SE",
            y="pred_SE",
            trendline="ols",
            title="Probe predictions vs Semantic Entropy",
        )
    )
    mo.vstack([metrics, fig])
    return preds, metrics


@app.cell
def __(mo, outputs_dir, sample_df, se_table):
    saver = mo.ui.run_button("Save artifacts to outputs/")
    if saver.value and sample_df is not None and se_table is not None:
        sample_path = outputs_dir / "samples.jsonl"
        se_path = outputs_dir / "semantic_entropies.csv"
        sample_df.to_json(sample_path, orient="records", lines=True, force_ascii=False)
        se_table.to_csv(se_path, index=False)
        mo.md(f"Saved samples → `{sample_path}` ; semantic entropy table → `{se_path}`")
    else:
        mo.md("Click save after running sampling to persist outputs.")
    return saver


if __name__ == "__main__":
    app.run()
