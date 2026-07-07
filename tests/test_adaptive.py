"""Tests for the adaptive-k controller.

Properties checked:
  - Sustained high acceptance raises k (up to k_max).
  - Sustained low acceptance lowers k (down to k_min).
  - k never leaves [k_min, k_max].
  - The k_schedule hook derives per-round acceptance from a SpecResult's
    running totals correctly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive import AdaptiveK  # noqa: E402
from src.spec_decode import SpecResult  # noqa: E402


def test_high_acceptance_raises_k():
    ctrl = AdaptiveK(k_min=1, k_max=10, k_init=3, raise_above=0.7, lower_below=0.4)
    start = ctrl.k
    for _ in range(10):
        ctrl.update(1.0)  # perfect acceptance every round
    assert ctrl.k > start
    assert ctrl.k == 10  # should climb to the ceiling


def test_low_acceptance_lowers_k():
    ctrl = AdaptiveK(k_min=1, k_max=10, k_init=8, raise_above=0.7, lower_below=0.4)
    start = ctrl.k
    for _ in range(20):
        ctrl.update(0.0)  # nothing accepted
    assert ctrl.k < start
    assert ctrl.k == 1  # should fall to the floor


def test_k_stays_within_bounds():
    ctrl = AdaptiveK(k_min=2, k_max=6, k_init=4)
    import random

    random.seed(0)
    for _ in range(200):
        ctrl.update(random.random())
        assert 2 <= ctrl.k <= 6


def test_k_init_is_clamped():
    ctrl = AdaptiveK(k_min=3, k_max=5, k_init=99)
    assert ctrl.k == 5
    ctrl2 = AdaptiveK(k_min=3, k_max=5, k_init=1)
    assert ctrl2.k == 3


def test_schedule_hook_reads_running_totals():
    """The __call__ hook should compute the previous round's rate from deltas."""
    ctrl = AdaptiveK(k_min=1, k_max=10, k_init=4, raise_above=0.7, lower_below=0.4)

    res = SpecResult()
    # First round: no history -> returns k_init.
    assert ctrl(res) == 4

    # Simulate a round with full acceptance: proposed 4, accepted 4.
    res.proposed += 4
    res.accepted += 4
    k_after_high = ctrl(res)
    assert k_after_high >= 4  # high acceptance should not lower k

    # Now simulate several fully-rejected rounds and confirm k drops.
    for _ in range(15):
        res.proposed += 4
        res.accepted += 0  # zero accepted this round
        ctrl(res)
    assert ctrl.k < k_after_high


def test_ema_smooths_single_bad_round():
    """One bad round after many good ones should not immediately collapse k."""
    ctrl = AdaptiveK(
        k_min=1, k_max=10, k_init=5, ema_beta=0.8, raise_above=0.7, lower_below=0.4
    )
    for _ in range(10):
        ctrl.update(1.0)
    k_high = ctrl.k
    ctrl.update(0.0)  # a single bad round
    # EMA is still well above lower_below, so k should not have been lowered.
    assert ctrl.k >= k_high - 1
