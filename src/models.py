"""Model loading helpers.

Kept separate from the algorithm so the decoding code depends only on a tiny
interface: a callable that maps input ids to next-token logits. This makes the
core loop testable with tiny stand-in models (see tests/).
"""
from __future__ import annotations

import os
from typing import Tuple

import torch


def load_pair(
    draft_name: str,
    target_name: str,
    device: str,
    hf_token: str | None = None,
):
    """Load the draft and target causal LMs and their shared tokenizer.

    The GPT-2 family shares one tokenizer/vocabulary, which is what lets a
    small draft propose tokens a larger target can score directly. If you swap
    in models with different vocabularies this helper (and the algorithm) would
    need per-model tokenizers and a vocabulary alignment step.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(draft_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft = AutoModelForCausalLM.from_pretrained(draft_name, token=token)
    target = AutoModelForCausalLM.from_pretrained(target_name, token=token)

    draft.to(device).eval()
    target.to(device).eval()
    return draft, target, tokenizer


@torch.no_grad()
def next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits for the next token given a (1, T) sequence: shape (vocab,)."""
    out = model(input_ids)
    return out.logits[0, -1, :]


@torch.no_grad()
def full_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits at every position: input (1, T) -> (T, vocab)."""
    out = model(input_ids)
    return out.logits[0]
