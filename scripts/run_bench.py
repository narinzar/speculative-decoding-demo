"""Benchmark standard vs speculative (fixed and adaptive k); save outputs/bench.json.

Usage:
    python scripts/run_bench.py --max-new-tokens 128 --k 4 \
        --prompt "In a distant future"

Writes a JSON report with wall time, tokens/sec, speedup vs the standard
baseline, and mean acceptance rate for each method.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from src.bench import rows_to_dict, run_benchmark  # noqa: E402
from src.config import GenConfig  # noqa: E402
from src.models import load_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    cfg = GenConfig()
    p = argparse.ArgumentParser(description="Benchmark speculative decoding.")
    p.add_argument("--prompt", default="In a distant future, humanity")
    p.add_argument("--draft", default=cfg.draft_model)
    p.add_argument("--target", default=cfg.target_model)
    p.add_argument("--max-new-tokens", type=int, default=cfg.max_new_tokens)
    p.add_argument("--k", type=int, default=cfg.k)
    p.add_argument("--k-min", type=int, default=cfg.k_min)
    p.add_argument("--k-max", type=int, default=cfg.k_max)
    p.add_argument("--temperature", type=float, default=cfg.temperature)
    p.add_argument("--top-p", type=float, default=cfg.top_p)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--out", default=os.path.join("outputs", "bench.json"))
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = GenConfig(
        draft_model=args.draft,
        target_model=args.target,
        max_new_tokens=args.max_new_tokens,
        k=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
    )
    device = cfg.resolved_device()
    print(f"Loading draft={cfg.draft_model} target={cfg.target_model} on {device} ...")
    draft, target, tok = load_pair(cfg.draft_model, cfg.target_model, device)

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    rows = run_benchmark(
        input_ids,
        draft,
        target,
        device=device,
        max_new_tokens=cfg.max_new_tokens,
        k=cfg.k,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        eos_token_id=tok.eos_token_id,
        seed=cfg.seed,
    )

    meta = {
        "draft_model": cfg.draft_model,
        "target_model": cfg.target_model,
        "device": device,
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if device == "cuda" and torch.cuda.is_available()
            else None
        ),
        "prompt": args.prompt,
        "max_new_tokens": cfg.max_new_tokens,
        "k": cfg.k,
        "k_min": cfg.k_min,
        "k_max": cfg.k_max,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "seed": cfg.seed,
    }
    report = rows_to_dict(rows, meta)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Benchmark ===")
    for r in report["results"]:
        line = (
            f"{r['method']:>28}  {r['wall_seconds']:.3f}s  "
            f"{r['tokens_per_sec']:.2f} tok/s"
        )
        if r.get("speedup_vs_standard") is not None:
            line += f"  {r['speedup_vs_standard']:.2f}x"
        if r.get("acceptance_rate") is not None:
            line += f"  acc={r['acceptance_rate']:.3f}"
        if r.get("mean_k") is not None:
            line += f"  mean_k={r['mean_k']:.2f}"
        print(line)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
