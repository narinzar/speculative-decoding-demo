"""Configuration for the speculative decoding demo.

All knobs live here so the CLI scripts and benchmarks read from one place.
Models are named so they can be swapped without touching the algorithm code.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenConfig:
    """Generation and decoding settings."""

    # Model names. Both are public GPT-2 family checkpoints on the HF Hub.
    # gpt2-large is the intended target; gpt2-medium is a lighter fallback.
    draft_model: str = "distilgpt2"
    target_model: str = "gpt2-large"

    # Sampling.
    temperature: float = 1.0
    top_p: float = 1.0  # 1.0 disables nucleus filtering; kept for completeness.

    # Speculative decoding.
    k: int = 4          # proposed tokens per round (fixed-k mode).
    k_min: int = 1      # bounds for adaptive-k mode.
    k_max: int = 10

    # Length and reproducibility.
    max_new_tokens: int = 128
    seed: int = 0

    # Device selection. "auto" picks cuda when available, else cpu.
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


DEFAULT = GenConfig()
