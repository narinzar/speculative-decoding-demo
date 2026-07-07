"""Benchmark: standard vs speculative (fixed-k and adaptive-k) decoding.

Reports wall time, tokens/sec, and mean acceptance rate for each method. Timing
uses torch.cuda synchronization when on GPU so the numbers reflect real device
work rather than async launch latency.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .adaptive import AdaptiveK
from .models import full_logits, next_token_logits
from .spec_decode import speculative_generate
from .standard_decode import standard_generate


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class BenchRow:
    method: str
    wall_seconds: float
    tokens: int
    tokens_per_sec: float
    acceptance_rate: Optional[float] = None
    mean_k: Optional[float] = None


def _time_call(fn: Callable[[], object], device: str):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def run_benchmark(
    input_ids: torch.Tensor,
    draft,
    target,
    device: str,
    max_new_tokens: int,
    k: int,
    k_min: int,
    k_max: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    seed: int = 0,
) -> List[BenchRow]:
    """Run the three methods with matched seeds and return one BenchRow each."""

    def draft_next(ids):
        return next_token_logits(draft, ids)

    def target_next(ids):
        return next_token_logits(target, ids)

    def target_full(ids):
        return full_logits(target, ids)

    def fresh_gen():
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return g

    rows: List[BenchRow] = []

    # 1) Standard autoregressive baseline.
    out, dt = _time_call(
        lambda: standard_generate(
            input_ids,
            target_next,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            generator=fresh_gen(),
        ),
        device,
    )
    n = len(out.tokens)
    rows.append(
        BenchRow("standard", dt, n, n / dt if dt > 0 else 0.0)
    )

    # 2) Speculative decoding with fixed k.
    out, dt = _time_call(
        lambda: speculative_generate(
            input_ids,
            draft_next,
            target_full,
            max_new_tokens=max_new_tokens,
            k=k,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            generator=fresh_gen(),
        ),
        device,
    )
    n = len(out.tokens)
    rows.append(
        BenchRow(
            f"speculative_fixed_k={k}",
            dt,
            n,
            n / dt if dt > 0 else 0.0,
            acceptance_rate=out.acceptance_rate,
            mean_k=sum(out.k_history) / len(out.k_history) if out.k_history else None,
        )
    )

    # 3) Speculative decoding with adaptive k.
    controller = AdaptiveK(k_min=k_min, k_max=k_max, k_init=k)
    out, dt = _time_call(
        lambda: speculative_generate(
            input_ids,
            draft_next,
            target_full,
            max_new_tokens=max_new_tokens,
            k=k,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            generator=fresh_gen(),
            k_schedule=controller,
        ),
        device,
    )
    n = len(out.tokens)
    rows.append(
        BenchRow(
            "speculative_adaptive_k",
            dt,
            n,
            n / dt if dt > 0 else 0.0,
            acceptance_rate=out.acceptance_rate,
            mean_k=sum(out.k_history) / len(out.k_history) if out.k_history else None,
        )
    )

    return rows


def rows_to_dict(rows: List[BenchRow], meta: Dict) -> Dict:
    baseline = next((r for r in rows if r.method == "standard"), None)
    out_rows = []
    for r in rows:
        d = asdict(r)
        if baseline is not None and baseline.wall_seconds > 0:
            d["speedup_vs_standard"] = baseline.wall_seconds / r.wall_seconds
        out_rows.append(d)
    return {"meta": meta, "results": out_rows}
