"""Exception types.

Every failure the user can actually cause — no camera, no models, a plate from
a different resolution — gets a distinct type carrying enough detail for the UI
to render an actionable message rather than a traceback.
"""

from __future__ import annotations


class DeleteMeError(Exception):
    """Base class for every error this package raises deliberately."""


class CameraError(DeleteMeError):
    """A camera could not be opened, configured, or read from."""

    def __init__(self, message: str, *, attempts: list[str] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []

    def __str__(self) -> str:
        base = super().__str__()
        if not self.attempts:
            return base
        return f"{base} (tried: {', '.join(self.attempts)})"


class ModelError(DeleteMeError):
    """A bundled model asset is missing or could not be loaded."""


class PlateError(DeleteMeError):
    """The background plate is missing, unreadable, or unusable."""


class PlateMismatchError(PlateError):
    """The plate does not match the shape of the frames being captured.

    This is raised in *both* directions. A plate larger than the live frame is
    the dangerous case: slicing it silently yields the wrong crop of the room,
    so the composite pastes the wrong part of the scene with no error at all.
    """

    def __init__(self, plate_shape: tuple[int, ...], frame_shape: tuple[int, ...]) -> None:
        self.plate_shape = plate_shape
        self.frame_shape = frame_shape
        super().__init__(
            f"Background plate is {plate_shape[1]}x{plate_shape[0]} but frames are "
            f"{frame_shape[1]}x{frame_shape[0]}. Recapture the background."
        )
