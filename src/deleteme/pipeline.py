"""Per-frame wiring.

Order is not arbitrary. Two constraints fix most of it:

* Segmentation runs on **every** frame, not only while the effect is engaged.
  The background model needs to know where the person is in order to keep them
  out of the plate refresh; without that, someone standing still in an idle room
  gets slowly averaged into the background and then cannot be deleted.
* The background model updates on **every** frame too, for the same reason in
  reverse — the idle path is when the room is unobstructed and the model can
  actually learn something.

The gesture recogniser is the exception. It runs on its own thread, because at
16.6 ms it is half a frame budget on its own, and unlike segmentation its result
is a single boolean that does not need to be frame-accurate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

import cv2
import numpy as np

from deleteme.background import BackgroundModel, Health
from deleteme.composite import Compositor, Dissolve
from deleteme.config import AppConfig
from deleteme.frames import Frame
from deleteme.gesture import EffectGate, GestureReading, GestureState, GestureWorker
from deleteme.segment import MaskStabilizer, PersonMask, PersonSegmenter

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """One processed frame and everything worth knowing about it."""

    image: np.ndarray
    engaged: bool
    strength: float
    health: Health
    mask: PersonMask
    gesture: GestureReading
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def person_visible(self) -> bool:
        return self.mask.present


class EffectPipeline:
    """Runs one frame through gesture, segmentation, background and composite."""

    def __init__(
        self,
        background: BackgroundModel,
        segmenter: PersonSegmenter,
        gesture_worker: GestureWorker | None,
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.background = background
        self.segmenter = segmenter
        self.gesture_worker = gesture_worker

        self.stabilizer = MaskStabilizer(self.config.segmentation)
        self.gate = EffectGate(self.config.gesture)
        self.dissolve = Dissolve(self.config.composite.dissolve_s)
        self.compositor = Compositor(self.config.composite)

        self._feather = self.config.segmentation.feather_kernel_px | 1
        self._closed = False

        self.force: bool | None = None
        """Manual override of the gesture gate.

        Useful when filming without wanting a hand in shot, and when testing
        without a person in front of the camera. ``None`` returns control to the
        recogniser.
        """

    def process(self, frame: Frame) -> PipelineResult:
        timings: dict[str, float] = {}
        now = frame.timestamp_s
        image = frame.image

        # The worker always gets the newest frame and we always read whatever it
        # last finished; the two rates are deliberately decoupled.
        if self.gesture_worker is not None:
            self.gesture_worker.submit(image)
            reading = self.gesture_worker.reading
        else:
            reading = GestureReading(GestureState.UNKNOWN)

        gated = self.gate.update(reading, now)
        engaged = gated if self.force is None else self.force
        self.dissolve.target(engaged, now)
        strength = self.dissolve.value(now)

        mark = perf_counter()
        confidence = self.segmenter.confidence(image)
        mask = self.stabilizer.update(confidence)
        timings["segment"] = (perf_counter() - mark) * 1000.0

        mark = perf_counter()
        health = self.background.update(image, mask.binary if mask.present else None, now)
        timings["background"] = (perf_counter() - mark) * 1000.0

        if strength <= 1e-3:
            # Fully visible. Drop the stabiliser's history so the next
            # activation starts clean rather than fading in last time's shape.
            if self.dissolve.settled:
                self.stabilizer.reset()
            timings["composite"] = 0.0
            return PipelineResult(
                image=image,
                engaged=engaged,
                strength=0.0,
                health=health,
                mask=mask,
                gesture=reading,
                timings=timings,
            )

        mark = perf_counter()
        alpha = self._removal_alpha(image, mask)
        result = self.compositor.compose(
            image, self.background.corrected_plate(), alpha, strength=strength
        )
        timings["composite"] = (perf_counter() - mark) * 1000.0

        return PipelineResult(
            image=result.image,
            engaged=engaged,
            strength=strength,
            health=health,
            mask=mask,
            gesture=reading,
            timings=timings,
        )

    def _removal_alpha(self, image: np.ndarray, mask: PersonMask) -> np.ndarray:
        """Coverage to remove: the silhouette, plus what it casts.

        The change mask contributes the cast shadow — the detail that decides
        whether the effect reads as real. Its output is hard-edged, so it is
        feathered to match the silhouette before the two are combined, and
        combined with a maximum so it can only ever add coverage.
        """
        if not self.config.change_mask.enabled or not mask.present:
            return mask.alpha

        combined = self.background.change_mask(image, mask.binary, mask.area_fraction)
        if combined is mask.binary:
            return mask.alpha

        feathered = cv2.GaussianBlur(combined, (self._feather, self._feather), 0)
        alpha = feathered.astype(np.float32)
        alpha *= 1.0 / 255.0
        return np.maximum(alpha, mask.alpha)

    def reset(self) -> None:
        """Return to fully visible, forgetting mask and gesture history."""
        self.gate.reset()
        self.dissolve.reset(0.0)
        self.stabilizer.reset()

    def close(self) -> None:
        """Shut down in the only safe order.

        The worker thread is joined *before* any native task is closed. Closing
        a MediaPipe task while a call is in flight is an access violation rather
        than an exception — it is not something a try block can save.
        """
        if self._closed:
            return
        self._closed = True
        if self.gesture_worker is not None:
            self.gesture_worker.close()
        self.segmenter.close()
