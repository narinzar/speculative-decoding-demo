"""Adaptive proposal length.

Original addition to the standard algorithm: instead of a fixed k, watch the
running acceptance rate and grow k when the draft is agreeing with the target
often, shrink it when it is not. The intuition:

  - High acceptance means the draft is a good local approximation, so proposing
    more tokens per round amortizes the single target pass over more accepted
    tokens -> more speedup.
  - Low acceptance means most proposals are wasted work (the target pass scored
    tokens that got rejected), so a shorter proposal length is cheaper.

We use an exponential moving average of the per-round acceptance rate so the
controller reacts to recent behavior without thrashing on a single round.
"""
from __future__ import annotations

from dataclasses import dataclass

from .spec_decode import SpecResult


@dataclass
class AdaptiveK:
    """Controller that returns the next round's k from the running acceptance.

    Parameters
    ----------
    k_min, k_max : hard bounds on the proposal length.
    k_init : starting k.
    ema_beta : smoothing for the acceptance-rate EMA (0..1, higher = smoother).
    raise_above : if smoothed acceptance exceeds this, increase k.
    lower_below : if smoothed acceptance falls below this, decrease k.
    step : how many tokens to move k by when adjusting.
    """

    k_min: int = 1
    k_max: int = 10
    k_init: int = 4
    ema_beta: float = 0.7
    raise_above: float = 0.7
    lower_below: float = 0.4
    step: int = 1

    def __post_init__(self) -> None:
        self.k = int(min(max(self.k_init, self.k_min), self.k_max))
        self.ema: float | None = None
        self._last_proposed = 0
        self._last_accepted = 0

    def _observe(self, rate: float) -> None:
        if self.ema is None:
            self.ema = rate
        else:
            self.ema = self.ema_beta * self.ema + (1.0 - self.ema_beta) * rate

    def update(self, round_acceptance: float) -> int:
        """Feed one round's acceptance rate, return the k for the next round."""
        self._observe(round_acceptance)
        assert self.ema is not None
        if self.ema > self.raise_above:
            self.k = min(self.k + self.step, self.k_max)
        elif self.ema < self.lower_below:
            self.k = max(self.k - self.step, self.k_min)
        return self.k

    def __call__(self, result: SpecResult) -> int:
        """k_schedule hook for speculative_generate.

        Called at the start of each round. It derives the *previous* round's
        acceptance rate from the running totals in `result`, updates the
        controller, and returns the k to use now. On the first round there is no
        history yet, so it returns k_init.
        """
        proposed_delta = result.proposed - self._last_proposed
        accepted_delta = result.accepted - self._last_accepted
        if proposed_delta > 0:
            rate = accepted_delta / proposed_delta
            self.update(rate)
        self._last_proposed = result.proposed
        self._last_accepted = result.accepted
        return self.k
