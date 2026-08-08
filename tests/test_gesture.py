"""The gate between a gesture reading and the effect being on.

No model is loaded — ``EffectGate`` takes readings, so the sequences worth
testing can be written out directly.
"""

from __future__ import annotations

from deleteme.clock import ManualClock
from deleteme.config import GestureConfig
from deleteme.gesture import EffectGate, GestureReading, GestureState

CLOSED = GestureReading(GestureState.CLOSED, 0.9)
OPEN = GestureReading(GestureState.OPEN, 0.9)
NOTHING = GestureReading(GestureState.UNKNOWN, 0.0)


def run(gate, clock, reading, seconds, step=1 / 30):
    """Feed one reading for a span of time and return the final state."""
    state = gate.engaged
    elapsed = 0.0
    while elapsed < seconds:
        clock.advance(step)
        state = gate.update(reading, clock.now())
        elapsed += step
    return state


def test_the_effect_starts_off():
    assert not EffectGate().engaged


def test_a_held_fist_engages_the_effect():
    gate, clock = EffectGate(), ManualClock()
    assert run(gate, clock, CLOSED, 0.3)


def test_a_single_frame_of_fist_does_not_engage():
    gate, clock = EffectGate(), ManualClock()
    clock.advance(1 / 30)
    assert not gate.update(CLOSED, clock.now())


def test_an_open_palm_restores():
    gate, clock = EffectGate(), ManualClock()
    run(gate, clock, CLOSED, 0.3)
    assert not run(gate, clock, OPEN, 0.3)


def test_a_missing_hand_holds_the_effect_on():
    """The bug this replaces: a dropped detection read as "hand open", so any
    frame where the recogniser blinked brought the person straight back."""
    gate, clock = EffectGate(GestureConfig(lost_hand_timeout_s=5.0)), ManualClock()
    run(gate, clock, CLOSED, 0.3)

    assert run(gate, clock, NOTHING, 2.0)


def test_a_hand_gone_for_a_long_time_fails_safe():
    """Holding forever would strand someone who walked away mid-effect."""
    gate, clock = EffectGate(GestureConfig(lost_hand_timeout_s=1.0)), ManualClock()
    run(gate, clock, CLOSED, 0.3)

    assert not run(gate, clock, NOTHING, 2.0)


def test_restoring_is_quicker_than_vanishing():
    """The same elapsed time is enough to restore but not enough to vanish."""
    config = GestureConfig(engage_hold_s=0.2, release_hold_s=0.05)
    gate, clock = EffectGate(config), ManualClock()

    clock.advance(0.1)
    gate.update(CLOSED, clock.now())
    clock.advance(0.06)
    assert not gate.update(CLOSED, clock.now()), "0.06s is not enough to vanish"

    run(gate, clock, CLOSED, 0.4)
    assert gate.engaged

    gate.update(OPEN, clock.now())
    clock.advance(0.06)
    assert not gate.update(OPEN, clock.now()), "0.06s is enough to come back"


def test_reset_returns_to_visible():
    gate, clock = EffectGate(), ManualClock()
    run(gate, clock, CLOSED, 0.3)

    gate.reset()

    assert not gate.engaged


class TestReadingObservations:
    def test_a_fist_argues_for_the_effect(self):
        assert CLOSED.observation is True

    def test_an_open_palm_argues_against_it(self):
        assert OPEN.observation is False

    def test_nothing_seen_is_not_evidence_either_way(self):
        assert NOTHING.observation is None
