"""Time source.

A seam, six lines of real code, and the reason the debounce tests can assert on
*time* instead of on frame counts. The previous implementation measured hold
duration in frames, so its behaviour silently changed with frame rate.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Monotonic time source, in seconds."""

    def now(self) -> float: ...


class MonotonicClock:
    """Real time, from :func:`time.monotonic`."""

    __slots__ = ()

    def now(self) -> float:
        return time.monotonic()


class ManualClock:
    """Test clock advanced by hand."""

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        """Move the clock forward and return the new time."""
        if seconds < 0:
            raise ValueError("ManualClock cannot move backwards")
        self._t += float(seconds)
        return self._t
