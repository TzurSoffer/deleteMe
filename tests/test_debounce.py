"""Hysteresis.

Every test here corresponds to a defect in the version this replaced, which
counted frames rather than seconds.
"""

from __future__ import annotations

import pytest

from deleteme.clock import ManualClock
from deleteme.debounce import Debouncer


def test_fresh_debouncer_does_not_flip_on_the_first_observation():
    """The original started its counter at zero and so had no debounce at all
    on the first event: one disagreeing frame flipped it immediately."""
    clock = ManualClock()
    debouncer = Debouncer(rise_s=0.1, fall_s=0.066, initial=False)

    assert debouncer.update(True, clock.now()) is False
    assert debouncer.pending


def test_change_is_accepted_only_after_the_hold_elapses():
    clock = ManualClock()
    debouncer = Debouncer(rise_s=0.1, fall_s=0.066)

    debouncer.update(True, clock.now())
    clock.advance(0.099)
    assert debouncer.update(True, clock.now()) is False
    clock.advance(0.002)
    assert debouncer.update(True, clock.now()) is True


@pytest.mark.parametrize("fps", [23, 30, 60, 88])
def test_hold_duration_is_independent_of_frame_rate(fps):
    """The old implementation measured its hold in frames, so the same code
    behaved differently on a fast machine than on a slow one — and its default
    of fps=1 reduced the hold to zero frames.

    Measured from the first observation of the change, not from t=0: the delay
    before that first observation is sampling, and is unavoidable at any rate.
    What must not vary is how long the evidence has to persist.
    """
    clock = ManualClock()
    debouncer = Debouncer(rise_s=0.1)
    step = 1.0 / fps

    first_seen = None
    flipped_at = None
    for _ in range(fps * 2):
        clock.advance(step)
        if first_seen is None:
            first_seen = clock.now()
        if debouncer.update(True, clock.now()):
            flipped_at = clock.now()
            break

    assert first_seen is not None and flipped_at is not None
    assert flipped_at - first_seen == pytest.approx(0.1, abs=step)


def test_rise_and_fall_are_independent():
    """Vanishing should feel considered; coming back should feel immediate.

    Times are nudged just past each boundary rather than sitting exactly on it:
    0.25 - 0.20 evaluates to 0.049999999999999996 in binary floating point, and
    a test that depends on which side of a boundary that lands is testing IEEE
    754 rather than this module.
    """
    clock = ManualClock()
    debouncer = Debouncer(rise_s=0.20, fall_s=0.05, initial=False)

    debouncer.update(True, clock.now())
    clock.advance(0.201)
    assert debouncer.update(True, clock.now()) is True

    debouncer.update(False, clock.now())
    clock.advance(0.051)
    assert debouncer.update(False, clock.now()) is False


def test_falling_does_not_have_to_wait_for_the_rise_time():
    """The asymmetry has to actually apply, not just be configured."""
    clock = ManualClock()
    debouncer = Debouncer(rise_s=1.0, fall_s=0.05, initial=True)

    debouncer.update(False, clock.now())
    clock.advance(0.06)

    assert debouncer.update(False, clock.now()) is False


def test_no_evidence_holds_rather_than_arguing_for_the_opposite():
    """A missed detection is not evidence the hand opened.

    The old code mapped "no hand found" to "hand open", so every dropped frame
    pushed the debouncer back towards cancelling the effect and the user had to
    hold a fist permanently and perfectly.
    """
    clock = ManualClock()
    debouncer = Debouncer(rise_s=0.1, fall_s=0.05)

    debouncer.update(True, clock.now())
    clock.advance(0.1)
    assert debouncer.update(True, clock.now()) is True

    for _ in range(100):
        clock.advance(0.033)
        assert debouncer.update(None, clock.now()) is True


def test_a_pending_change_is_abandoned_if_the_evidence_reverses():
    clock = ManualClock()
    debouncer = Debouncer(rise_s=0.1)

    debouncer.update(True, clock.now())
    clock.advance(0.09)
    debouncer.update(False, clock.now())  # agrees with current state; cancels
    clock.advance(0.02)
    assert debouncer.update(True, clock.now()) is False, "hold restarted, not resumed"


def test_negative_hold_times_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        Debouncer(rise_s=-1.0)


def test_manual_clock_refuses_to_go_backwards():
    with pytest.raises(ValueError, match="backwards"):
        ManualClock().advance(-1.0)
