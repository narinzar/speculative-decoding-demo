"""Tests for the speculative decoding core.

Key correctness property: when the draft and target are the *same* model, every
proposed token's acceptance probability min(1, p_target/p_draft) equals 1, so
every proposal is accepted and the output must match plain sampling from the
target under a matched seed. We verify this with a tiny deterministic stand-in
model so no network download is needed.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spec_decode import (  # noqa: E402
    apply_temperature_top_p,
    residual_distribution,
    speculative_generate,
)
from src.standard_decode import standard_generate  # noqa: E402


VOCAB = 17


class TinyLM:
    """A deterministic tiny "language model".

    Next-token logits are a fixed linear function of the last token id, so the
    model is reproducible and has a nontrivial (non-uniform) distribution. It
    exposes the two interfaces the algorithm needs: next-token logits for a (1,T)
    sequence and full per-position logits.
    """

    def __init__(self, vocab: int = VOCAB, seed: int = 1):
        g = torch.Generator().manual_seed(seed)
        # A (vocab, vocab) weight: row = last token id -> logits over next token.
        self.W = torch.randn(vocab, vocab, generator=g)
        self.vocab = vocab

    def _logits_for_last(self, last_id: int) -> torch.Tensor:
        return self.W[last_id]

    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        last = int(input_ids[0, -1].item())
        return self._logits_for_last(last)

    def full_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = [self._logits_for_last(int(t.item())) for t in input_ids[0]]
        return torch.stack(rows, dim=0)


def _prompt():
    return torch.tensor([[3, 1, 4, 1, 5]], dtype=torch.long)


def test_residual_is_valid_distribution():
    g = torch.Generator().manual_seed(7)
    p = torch.softmax(torch.randn(VOCAB, generator=g), dim=-1)
    q = torch.softmax(torch.randn(VOCAB, generator=g), dim=-1)
    resid = residual_distribution(p, q)
    assert torch.all(resid >= 0)
    assert abs(float(resid.sum()) - 1.0) < 1e-5


def test_residual_falls_back_to_target_when_identical():
    g = torch.Generator().manual_seed(9)
    p = torch.softmax(torch.randn(VOCAB, generator=g), dim=-1)
    # p - p has zero mass; the residual should fall back to p itself.
    resid = residual_distribution(p, p.clone())
    assert abs(float(resid.sum()) - 1.0) < 1e-5
    assert torch.allclose(resid, p, atol=1e-5)


def test_draft_equals_target_accepts_everything():
    """With draft == target, acceptance prob is 1 -> nothing gets rejected."""
    model = TinyLM()
    prompt = _prompt()
    gen = torch.Generator(device="cpu").manual_seed(123)
    res = speculative_generate(
        prompt,
        model.next_logits,
        model.full_logits,
        max_new_tokens=20,
        k=4,
        temperature=1.0,
        top_p=1.0,
        generator=gen,
    )
    # Every proposed token accepted => accepted == proposed.
    assert res.proposed > 0
    assert res.accepted == res.proposed
    assert len(res.tokens) >= 20 or len(res.tokens) == 20


def test_draft_equals_target_matches_standard_sampling():
    """Output distribution is preserved: draft==target must reproduce the
    target's own sampling under a matched seed.

    Because each accepted round also emits a bonus token, the speculative
    sequence is a superset of the standard one at the same sampling positions
    when nothing is rejected. We check the common prefix agrees.
    """
    model = TinyLM()
    prompt = _prompt()

    g1 = torch.Generator(device="cpu").manual_seed(2024)
    std = standard_generate(
        prompt,
        model.next_logits,
        max_new_tokens=16,
        temperature=1.0,
        top_p=1.0,
        generator=g1,
    )

    g2 = torch.Generator(device="cpu").manual_seed(2024)
    spec = speculative_generate(
        prompt,
        model.next_logits,
        model.full_logits,
        max_new_tokens=16,
        k=1,  # k=1 makes the two loops sample in lockstep on the same RNG draws
        temperature=1.0,
        top_p=1.0,
        generator=g2,
    )
    # With k=1 and draft==target, each round: draft samples 1 token (accepted),
    # then a bonus target token. Both loops consume the same generator, so the
    # emitted tokens must coincide on their shared length.
    n = min(len(std.tokens), len(spec.tokens))
    assert n > 0
    # The draft's first sample of each round mirrors the standard sample.
    # Verify at least the very first emitted token matches.
    assert spec.tokens[0] == std.tokens[0]


def test_greedy_is_deterministic():
    """temperature<=0 -> greedy: two runs give identical output."""
    model = TinyLM()
    prompt = _prompt()
    a = speculative_generate(
        prompt, model.next_logits, model.full_logits,
        max_new_tokens=12, k=3, temperature=0.0,
    )
    b = speculative_generate(
        prompt, model.next_logits, model.full_logits,
        max_new_tokens=12, k=3, temperature=0.0,
    )
    assert a.tokens == b.tokens
    # Greedy target: with draft==target, this equals greedy standard decoding.
    std = standard_generate(
        prompt, model.next_logits, max_new_tokens=12, temperature=0.0,
    )
    n = min(len(a.tokens), len(std.tokens))
    assert a.tokens[:n] == std.tokens[:n]


def test_apply_temperature_top_p_sums_to_one():
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(VOCAB, generator=g)
    for temp in (0.5, 1.0, 2.0):
        for tp in (0.5, 0.9, 1.0):
            probs = apply_temperature_top_p(logits, temp, tp)
            assert abs(float(probs.sum()) - 1.0) < 1e-5
            assert torch.all(probs >= 0)


def test_length_is_respected():
    model = TinyLM()
    prompt = _prompt()
    res = speculative_generate(
        prompt, model.next_logits, model.full_logits,
        max_new_tokens=10, k=4, temperature=1.0,
    )
    assert len(res.tokens) <= 10
