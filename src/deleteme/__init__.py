"""DeleteMe — make yourself disappear from your own webcam feed.

The effect is simple to describe and easy to get wrong: hold up a closed fist
and the person in frame is replaced by a photograph of the empty room behind
them. Everything hard about it comes from that photograph — the *plate* — going
stale. Room lighting shifts, the camera's auto-exposure hunts, someone opens a
blind, the tripod gets nudged. A plate captured thirty seconds ago no longer
matches the pixels arriving now, and the composite shows a visible patch.

This package therefore treats the plate as a live model rather than a saved
image. :class:`~deleteme.background.BackgroundModel` re-fits the plate to the
current frame on every frame, tracks how well it still matches, refreshes it
where the scene is genuinely visible, and says so when it has given up.
"""

from deleteme.errors import (
    CameraError,
    DeleteMeError,
    ModelError,
    PlateError,
    PlateMismatchError,
)

__version__ = "0.2.0"

__all__ = [
    "CameraError",
    "DeleteMeError",
    "ModelError",
    "PlateError",
    "PlateMismatchError",
    "__version__",
]
