"""Generate text with speculative decoding.

Usage:
    python scripts/run_generate.py --prompt "The history of computing" \
        --max-new-tokens 128 --k 4

By default draft=distilgpt2, target=gpt2-large. Swap models via --draft/--target
(for example --target gpt2-medium if gpt2-large is too heavy for your machine).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from src.config import GenConfig  # noqa: E402
from src.models import full_logits, load_pair, next_token_logits  # noqa: E402
from src.spec_decode import speculative_generate  # noqa: E402
from src.adaptive import AdaptiveK  # noqa: E402


def parse_args() -> argparse.Namespace:
    cfg = GenConfig()
    p = argparse.ArgumentParser(description="Speculative decoding text generation.")
    p.add_argument("--prompt", default="The history of computing began")
    p.add_argument("--draft", default=cfg.draft_model)
    p.add_argument("--target", default=cfg.target_model)
    p.add_argument("--max-new-tokens", type=int, default=cfg.max_new_tokens)
    p.add_argument("--k", type=int, default=cfg.k)
    p.add_argument("--temperature", type=float, default=cfg.temperature)
    p.add_argument("--top-p", type=float, default=cfg.top_p)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", default=cfg.device)
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive k (bounded by --k-min/--k-max) instead of fixed k.",
    )
    p.add_argument("--k-min", type=int, default=cfg.k_min)
    p.add_argument("--k-max", type=int, default=cfg.k_max)
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
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)

    k_schedule = None
    if args.adaptive:
        k_schedule = AdaptiveK(k_min=cfg.k_min, k_max=cfg.k_max, k_init=cfg.k)

    result = speculative_generate(
        input_ids,
        lambda ids: next_token_logits(draft, ids),
        lambda ids: full_logits(target, ids),
        max_new_tokens=cfg.max_new_tokens,
        k=cfg.k,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        eos_token_id=tok.eos_token_id,
        generator=gen,
        k_schedule=k_schedule,
    )

    text = tok.decode(result.tokens, skip_special_tokens=True)
    print("\n=== Generated ===")
    print(args.prompt + text)
    print("\n=== Stats ===")
    print(f"tokens generated : {len(result.tokens)}")
    print(f"rounds           : {result.rounds}")
    print(f"acceptance rate  : {result.acceptance_rate:.3f}")
    if result.k_history:
        mean_k = sum(result.k_history) / len(result.k_history)
        print(f"mean k           : {mean_k:.2f}")


if __name__ == "__main__":
    main()
