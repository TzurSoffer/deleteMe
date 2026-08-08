"""Windows timer resolution.

Windows schedules timers on a ~15.6 ms tick by default, which is half a frame
budget at 30 fps. Anything that asks to be woken "in 1 ms" — ``tkinter.after``,
``cv2.waitKey``, ``time.sleep`` — actually waits until the next tick. The old
version's ``cv2.waitKey(1)`` measured 15.5 ms for this reason, the second
largest cost in the whole application after inference.

``timeBeginPeriod`` is a documented, process-scoped, reversible request for a
finer tick. It must be paired with ``timeEndPeriod``, which is what the context
manager is for.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Generator

log = logging.getLogger(__name__)


@contextlib.contextmanager
def high_resolution_timer(milliseconds: int = 1) -> Generator[bool]:
    """Request a finer system timer for the duration of the block.

    Yields whether the request was granted. A no-op everywhere but Windows,
    where the default granularity is the problem.
    """
    if sys.platform != "win32":
        yield False
        return

    try:
        import ctypes

        winmm = ctypes.WinDLL("winmm")
    except (ImportError, OSError):  # pragma: no cover - non-standard Windows
        yield False
        return

    granted = winmm.timeBeginPeriod(milliseconds) == 0
    if not granted:  # pragma: no cover - driver dependent
        log.debug("timeBeginPeriod(%d) refused", milliseconds)
    try:
        yield granted
    finally:
        if granted:
            winmm.timeEndPeriod(milliseconds)
