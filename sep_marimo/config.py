from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class ExperimentConfig:
    """Central config for the Qwen semantic-entropy probe pipeline."""

    model_name: str = "Qwen/Qwen3-0.6B-Instruct"
    # Sampling / decoding
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 2048
    num_runs_per_question: int = 10
    top_k_for_entropy: int = 20
    # Splits and randomness
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    random_seed: int = 42
    # Think tags
    think_start: str = "<think>"
    think_end: str = "</think>"
    # Paths (relative to repo root)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))

    @property
    def math_raw_path(self) -> Path:
        return self.data_dir / "math_raw.jsonl"

    @property
    def ood_raw_path(self) -> Path:
        return self.data_dir / "ood_raw.jsonl"

    @property
    def math_runs_path(self) -> Path:
        return self.data_dir / "math_runs.jsonl"

    @property
    def ood_runs_path(self) -> Path:
        return self.data_dir / "ood_runs.jsonl"

    @property
    def math_semantic_entropy_path(self) -> Path:
        return self.data_dir / "math_semantic_entropy.jsonl"

    @property
    def hidden_state_dir(self) -> Path:
        return self.artifacts_dir

    @property
    def math_hidden_path(self) -> Path:
        return self.hidden_state_dir / "math_hidden_states.npz"

    @property
    def ood_hidden_path(self) -> Path:
        return self.hidden_state_dir / "ood_hidden_states.npz"

    @property
    def probe_dataset_dir(self) -> Path:
        return self.artifacts_dir / "probe_datasets"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.probe_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = ExperimentConfig()
