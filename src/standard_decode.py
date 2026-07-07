"""Plain autoregressive sampling from the target model.

This is the latency baseline: one target forward pass per generated token. It
uses the exact same temperature / top-p sampling path as the speculative loop so
the comparison is apples to apples (same output distribution, different speed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from .spec_decode import NextLogitsFn, apply_temperature_top_p, _sample


@dataclass
class StandardResult:
    tokens: List[int] = field(default_factory=list)
    steps: int = 0  # target forward passes == generated tokens


def standard_generate(
    input_ids: torch.Tensor,
    target_next_logits: NextLogitsFn,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> StandardResult:
    """Sample max_new_tokens from the target, one forward pass per token."""
    device = input_ids.device
    seq = input_ids.clone()
    result = StandardResult()

    for _ in range(max_new_tokens):
        logits = target_next_logits(seq)
        probs = apply_temperature_top_p(logits, temperature, top_p)
        tok = _sample(probs, generator)
        result.tokens.append(tok)
        result.steps += 1
        seq = torch.cat(
            [seq, torch.tensor([[tok]], device=device, dtype=seq.dtype)], dim=1
        )
        if eos_token_id is not None and tok == eos_token_id:
            break

    return result
