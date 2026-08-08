"""Time-based hysteresis.

The previous implementation counted frames, so how long a gesture had to be
held silently depended on frame rate — and its default of ``fps=1`` reduced the
hold to zero frames, disabling hysteresis entirely for any new caller. It also
started its counter at zero, which meant a freshly constructed debouncer flipped
on its very first disagreeing observation.

This one measures seconds, starts fully charged, and distinguishes *evidence
against* the current state from *no evidence at all*.
"""

from __future__ import annotations


class Debouncer:
    """Requires a change to persist before accepting it.

    Rise and fall times are separate on purpose. In this application vanishing
    should feel deliberate while reappearing should feel immediate, so the two
    directions are not symmetric.
    """

    __slots__ = ("_pending_since", "_pending_value", "_state", "fall_s", "rise_s")

    def __init__(self, rise_s: float, fall_s: float | None = None, initial: bool = False) -> None:
        if rise_s < 0 or (fall_s is not None and fall_s < 0):
            raise ValueError("hold times cannot be negative")
        self.rise_s = float(rise_s)
        self.fall_s = float(rise_s if fall_s is None else fall_s)
        self._state = bool(initial)
        self._pending_since: float | None = None
        self._pending_value: bool | None = None

    @property
    def state(self) -> bool:
        return self._state

    @property
    def pending(self) -> bool:
        """True while a change is being considered but has not been accepted."""
        return self._pending_since is not None

    def update(self, observation: bool | None, now: float) -> bool:
        """Fold in one observation and return the debounced state.

        ``observation`` of ``None`` means *no evidence* — the detector saw
        nothing — and is not the same as evidence of the opposite. Mapping a
        detection dropout to False is what forced the old version's user to hold
        a fist permanently: every missed frame actively argued for coming back.
        """
        if observation is None:
            self._pending_since = None
            self._pending_value = None
            return self._state

        if observation == self._state:
            self._pending_since = None
            self._pending_value = None
            return self._state

        if self._pending_since is None or self._pending_value != observation:
            self._pending_since = now
            self._pending_value = observation
            return self._state

        required = self.rise_s if observation else self.fall_s
        if now - self._pending_since >= required:
            self._state = observation
            self._pending_since = None
            self._pending_value = None
        return self._state

    def force(self, state: bool) -> None:
        """Set the state outright, abandoning any pending change."""
        self._state = bool(state)
        self._pending_since = None
        self._pending_value = None
