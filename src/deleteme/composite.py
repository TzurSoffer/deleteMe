"""Blending the corrected plate over the subject.

Two things matter here and they are both about restraint.

The first is arithmetic cost. A naive full-frame float blend,
``frame * (1 - a) + plate * a``, measured 20.5 ms at 640x480 — sixty-two percent
of a 30 fps budget spent on three multiplies. Working only inside the silhouette's
bounding box brings the same result down to a fraction of a millisecond, because
the subject occupies a small part of the frame and the rest of the image is
already correct.

The second is the transition. At 30 fps an instant cut lasts one frame, which on
video reads as a dropped frame or a bad edit rather than as an effect. A short
eased dissolve reads as intent.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from deleteme.config import CompositeConfig


def smoothstep(t: float) -> float:
    """Hermite ease, zero slope at both ends."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


class Dissolve:
    """Eases between visible and deleted.

    Driven by wall-clock time rather than a frame count, so a dropped frame
    stretches the animation instead of jumping it.
    """

    def __init__(self, duration_s: float) -> None:
        self.duration_s = max(1e-3, float(duration_s))
        self._from = 0.0
        self._to = 0.0
        self._started = 0.0
        self._value = 0.0

    def target(self, engaged: bool, now: float) -> None:
        goal = 1.0 if engaged else 0.0
        if goal == self._to:
            return
        # Start from wherever the ramp currently is, so reversing mid-dissolve
        # is continuous rather than a jump back to the beginning.
        self._from = self.value(now)
        self._to = goal
        self._started = now

    def value(self, now: float) -> float:
        if self._to == self._from:
            self._value = self._to
            return self._value
        progress = (now - self._started) / self.duration_s
        self._value = self._from + (self._to - self._from) * smoothstep(progress)
        return self._value

    @property
    def settled(self) -> bool:
        return abs(self._value - self._to) < 1e-3

    def reset(self, value: float = 0.0) -> None:
        self._from = self._to = self._value = float(value)


@dataclass(frozen=True)
class CompositeResult:
    image: np.ndarray
    strength: float
    roi: tuple[int, int, int, int] | None
    """``(x, y, w, h)`` actually blended, or ``None`` when nothing was."""


class Compositor:
    """Alpha-blends the plate over the subject, inside a bounding box."""

    def __init__(self, config: CompositeConfig | None = None) -> None:
        self.config = config or CompositeConfig()
        self._out: np.ndarray | None = None

    def _output_buffer(self, frame: np.ndarray) -> np.ndarray:
        if self._out is None or self._out.shape != frame.shape:
            self._out = np.empty_like(frame)
        np.copyto(self._out, frame)
        return self._out

    def compose(
        self,
        frame: np.ndarray,
        plate: np.ndarray,
        alpha: np.ndarray,
        strength: float = 1.0,
        margin: int = 12,
    ) -> CompositeResult:
        """Blend ``plate`` over ``frame`` weighted by ``alpha * strength``.

        ``alpha`` is float32 in [0, 1] at frame resolution. ``strength`` is the
        dissolve ramp; at 0 the frame is returned untouched, which is also the
        fast path for the overwhelming majority of frames.
        """
        out = self._output_buffer(frame)
        if strength <= 1e-3 or alpha.size == 0:
            return CompositeResult(image=out, strength=0.0, roi=None)

        if plate.shape != frame.shape:
            raise ValueError(f"plate {plate.shape} does not match frame {frame.shape}")

        box = cv2.boundingRect((alpha > (1.0 / 255.0)).astype(np.uint8))
        if box[2] == 0 or box[3] == 0:
            return CompositeResult(image=out, strength=0.0, roi=None)

        height, width = frame.shape[:2]
        x0 = max(0, box[0] - margin)
        y0 = max(0, box[1] - margin)
        x1 = min(width, box[0] + box[2] + margin)
        y1 = min(height, box[1] + box[3] + margin)

        weight = alpha[y0:y1, x0:x1]
        if strength < 1.0:
            weight = weight * float(strength)

        frame_roi = frame[y0:y1, x0:x1].astype(np.float32)
        plate_roi = plate[y0:y1, x0:x1].astype(np.float32)

        # frame + (plate - frame) * a, which is the same blend with one fewer
        # multiply than frame*(1-a) + plate*a.
        delta = cv2.subtract(plate_roi, frame_roi)
        delta = cv2.multiply(delta, cv2.merge((weight, weight, weight)))
        blended = cv2.add(frame_roi, delta)

        out[y0:y1, x0:x1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        return CompositeResult(image=out, strength=float(strength), roi=(x0, y0, x1 - x0, y1 - y0))
