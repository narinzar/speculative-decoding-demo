"""Speculative decoding, implemented from scratch.

The idea: a small fast "draft" model proposes k tokens autoregressively. The
large "target" model then scores all k proposed positions in a single forward
pass. Each proposed token is accepted with probability min(1, p_target/p_draft);
on the first rejection we resample that position from the normalized residual
max(0, p_target - p_draft), and stop. If all k are accepted we additionally
sample one bonus token from the target's distribution at the final position.

Why this is exact: the acceptance test plus residual resampling is constructed
so that the token emitted at every step is distributed exactly as if it had been
drawn from the target model directly. The draft only changes *speed*, never the
output distribution. The proof is the standard rejection-sampling identity:

    accept x ~ q with prob min(1, p(x)/q(x)), else draw from
    (p - q)_+ / sum((p - q)_+)   ==>   the emitted x ~ p.

Here q = p_draft, p = p_target.

KV-cache note: for clarity and correctness this implementation recomputes the
forward pass over the growing sequence each round rather than threading manual
past_key_values through both models. That keeps the accept/reject bookkeeping
readable and matches the from-scratch goal. The wins measured in bench.py come
from the target running once per round over k+1 positions instead of once per
token; caching would lower the constant factor further but not change the
algorithm or its output distribution. See README for the trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


# A scorer maps a (1, T) LongTensor of ids to next-token logits of shape (vocab,).
NextLogitsFn = Callable[[torch.Tensor], torch.Tensor]
FullLogitsFn = Callable[[torch.Tensor], torch.Tensor]


def apply_temperature_top_p(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Turn logits into a probability vector with optional temp + nucleus filter.

    Operates on a 1-D logits tensor and returns a 1-D probability tensor that
    sums to 1. temperature <= 0 is treated as greedy (a one-hot at the argmax).
    """
    if temperature is None or temperature <= 0:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest set whose cumulative mass reaches top_p.
        cutoff = cumulative > top_p
        # Always keep at least the top token.
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum()

    return probs


def residual_distribution(
    p_target: torch.Tensor, p_draft: torch.Tensor
) -> torch.Tensor:
    """Normalized residual max(0, p_target - p_draft) used on rejection.

    If the residual has zero mass (can happen when the two distributions match
    exactly), fall back to the target distribution itself, which is the correct
    limit and keeps the sampler well defined.
    """
    resid = torch.clamp(p_target - p_draft, min=0.0)
    total = resid.sum()
    if total <= 0:
        return p_target / p_target.sum()
    return resid / total


def _sample(probs: torch.Tensor, generator: Optional[torch.Generator]) -> int:
    """Sample one index from a probability vector.

    Sampling is done on CPU so a CPU torch.Generator gives reproducible draws
    even when the model logits live on a CUDA device (torch.multinomial requires
    the probs tensor and the generator to share a device).
    """
    p = probs.detach().to("cpu").float()
    return int(torch.multinomial(p, num_samples=1, generator=generator).item())


def _uniform(generator: Optional[torch.Generator]) -> float:
    """A single uniform(0,1) draw, on CPU, reproducible under `generator`."""
    return float(torch.rand(1, generator=generator).item())


@dataclass
class SpecResult:
    tokens: List[int] = field(default_factory=list)  # newly generated token ids
    accepted: int = 0        # proposed tokens accepted across all rounds
    proposed: int = 0        # proposed tokens offered across all rounds
    rounds: int = 0          # number of speculative rounds run
    k_history: List[int] = field(default_factory=list)  # k used each round

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def speculative_generate(
    input_ids: torch.Tensor,
    draft_next_logits: NextLogitsFn,
    target_full_logits: FullLogitsFn,
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    k_schedule: Optional[Callable[["SpecResult"], int]] = None,
) -> SpecResult:
    """Generate tokens by speculative decoding.

    Parameters
    ----------
    input_ids : LongTensor of shape (1, T0), the prompt.
    draft_next_logits : maps (1, T) -> (vocab,) next-token logits for the draft.
    target_full_logits : maps (1, T) -> (T, vocab) logits at every position for
        the target. We only need the last k+1 rows per round but scoring the
        whole sequence keeps the interface simple.
    k : proposed tokens per round (fixed unless k_schedule is given).
    k_schedule : optional callable that, given the running SpecResult, returns
        the k to use for the next round. Used for adaptive-k (see adaptive.py).

    Returns a SpecResult with the generated tokens and acceptance statistics.
    """
    device = input_ids.device
    seq = input_ids.clone()
    result = SpecResult()

    while len(result.tokens) < max_new_tokens:
        cur_k = k_schedule(result) if k_schedule is not None else k
        cur_k = max(1, cur_k)
        # Do not propose past the requested length.
        cur_k = min(cur_k, max_new_tokens - len(result.tokens))
        result.rounds += 1
        result.k_history.append(cur_k)

        # 1) Draft proposes cur_k tokens autoregressively, remembering its own
        #    probability for each proposed token (needed for the accept test).
        draft_seq = seq
        proposed_tokens: List[int] = []
        draft_probs_at: List[torch.Tensor] = []  # full prob vectors per position
        for _ in range(cur_k):
            d_logits = draft_next_logits(draft_seq)
            d_probs = apply_temperature_top_p(d_logits, temperature, top_p)
            tok = _sample(d_probs, generator)
            proposed_tokens.append(tok)
            draft_probs_at.append(d_probs)
            draft_seq = torch.cat(
                [draft_seq, torch.tensor([[tok]], device=device, dtype=seq.dtype)],
                dim=1,
            )

        # 2) Target scores the prompt + all proposed tokens in ONE forward pass.
        #    Row i of the target logits is the distribution for the token that
        #    follows position i. To score the j-th proposed token we need the
        #    row at the position just before it.
        t_full = target_full_logits(draft_seq)  # (T0 + cur_k, vocab)
        base = seq.shape[1]  # index of the row predicting the 1st proposed token

        # 3) Accept / reject each proposed token in order.
        n_accepted_this_round = 0
        rejected = False
        for j in range(cur_k):
            row = base + j - 1
            t_probs = apply_temperature_top_p(t_full[row], temperature, top_p)
            d_probs = draft_probs_at[j]
            tok = proposed_tokens[j]

            p_t = float(t_probs[tok])
            p_d = float(d_probs[tok])
            # Acceptance probability min(1, p_target / p_draft).
            if p_d <= 0:
                accept_prob = 1.0 if p_t > 0 else 0.0
            else:
                accept_prob = min(1.0, p_t / p_d)

            u = _uniform(generator)
            if u < accept_prob:
                result.tokens.append(tok)
                n_accepted_this_round += 1
                if eos_token_id is not None and tok == eos_token_id:
                    result.accepted += n_accepted_this_round
                    result.proposed += cur_k
                    return result
            else:
                # First rejection: resample from the normalized residual and stop
                # this round. This is the single token that "replaces" the
                # rejected proposal and keeps the output exact.
                resid = residual_distribution(t_probs, d_probs)
                new_tok = _sample(resid, generator)
                result.tokens.append(new_tok)
                rejected = True
                if eos_token_id is not None and new_tok == eos_token_id:
                    result.accepted += n_accepted_this_round
                    result.proposed += cur_k
                    return result
                break

        # 4) If every proposed token was accepted, sample one bonus token from
        #    the target's distribution at the final position (free extra token).
        if not rejected:
            row = base + cur_k - 1
            t_probs = apply_temperature_top_p(t_full[row], temperature, top_p)
            bonus = _sample(t_probs, generator)
            result.tokens.append(bonus)
            if eos_token_id is not None and bonus == eos_token_id:
                result.accepted += n_accepted_this_round
                result.proposed += cur_k
                return result

        result.accepted += n_accepted_this_round
        result.proposed += cur_k

        # Rebuild the running sequence from the accepted/emitted tokens.
        seq = torch.cat(
            [
                input_ids,
                torch.tensor([result.tokens], device=device, dtype=seq.dtype),
            ],
            dim=1,
        )

    # Trim in case the bonus token overshot the requested length.
    if len(result.tokens) > max_new_tokens:
        result.tokens = result.tokens[:max_new_tokens]
    return result
