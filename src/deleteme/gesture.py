"""Gesture recognition and the gate it drives.

The previous classifier counted extended fingers by comparing each fingertip's
distance from the wrist against the corresponding knuckle's. Three problems, all
real: it used normalised landmark coordinates, which are scaled independently by
width and height, so on a 4:3 frame the comparison is anisotropic and gets less
decisive as the wrist rolls; it ignored the thumb and required three extended
fingers to count as open, so pointing, a peace sign and a thumbs-up all read as
a closed fist and made you vanish; and it ran in IMAGE mode, so every frame was
solved from scratch.

MediaPipe's gesture recogniser answers the question directly, and measured
*cheaper* than the landmark model it replaces — 16.6 ms against the old
pipeline's hand landmarker.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import processors as mp_processors

from deleteme.config import GestureConfig
from deleteme.debounce import Debouncer
from deleteme.errors import ModelError
from deleteme.paths import GESTURE_MODEL, model_path

log = logging.getLogger(__name__)

CLOSED_FIST = "Closed_Fist"
OPEN_PALM = "Open_Palm"


class GestureState(Enum):
    """What the recogniser saw."""

    CLOSED = "closed"
    OPEN = "open"
    UNKNOWN = "unknown"
    """No hand, or nothing confident enough.

    Critically *not* the same as OPEN. The old code mapped a detection dropout
    to "hand open", which actively argued for cancelling the effect, so a single
    missed frame during a wave brought the person back.
    """


@dataclass(frozen=True)
class GestureReading:
    state: GestureState
    score: float = 0.0

    @property
    def observation(self) -> bool | None:
        """Evidence for the debouncer: True, False, or "no evidence"."""
        if self.state is GestureState.CLOSED:
            return True
        if self.state is GestureState.OPEN:
            return False
        return None


class GestureReader:
    """MediaPipe gesture recognition in VIDEO mode."""

    def __init__(
        self,
        model_file: Path | None = None,
        config: GestureConfig | None = None,
        fps: int = 30,
    ) -> None:
        self.config = config or GestureConfig()
        path = model_file or model_path(GESTURE_MODEL)
        try:
            self._recognizer = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.config.num_hands,
                    min_hand_detection_confidence=self.config.detection_confidence,
                    min_hand_presence_confidence=self.config.presence_confidence,
                    min_tracking_confidence=self.config.tracking_confidence,
                    canned_gesture_classifier_options=mp_processors.ClassifierOptions(
                        score_threshold=min(self.config.engage_score, self.config.release_score),
                        category_allowlist=[CLOSED_FIST, OPEN_PALM],
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - native failure
            raise ModelError(f"could not load the gesture model from {path}: {exc}") from exc

        self._step_ms = max(1, round(1000 / max(1, fps)))
        self._timestamp_ms = 0
        self._closed = False

    def read(self, frame_bgr: np.ndarray) -> GestureReading:
        """Classify one frame.

        Note that once :meth:`close` has been called MediaPipe returns an empty
        result rather than raising — which would read as UNKNOWN, hold the
        current state, and quietly do the right thing. That is deliberate: the
        alternative interpretation would make the person flicker back into view
        during teardown.
        """
        if self._closed:
            return GestureReading(GestureState.UNKNOWN)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += self._step_ms
        result = self._recognizer.recognize_for_video(image, self._timestamp_ms)

        best_closed = 0.0
        best_open = 0.0
        for hand in result.gestures or []:
            for category in hand:
                if category.category_name == CLOSED_FIST:
                    best_closed = max(best_closed, float(category.score))
                elif category.category_name == OPEN_PALM:
                    best_open = max(best_open, float(category.score))

        cfg = self.config
        if best_closed >= cfg.engage_score and best_closed >= best_open:
            return GestureReading(GestureState.CLOSED, best_closed)
        if best_open >= cfg.release_score and best_open > best_closed:
            return GestureReading(GestureState.OPEN, best_open)
        return GestureReading(GestureState.UNKNOWN, max(best_closed, best_open))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._recognizer.close()

    def __enter__(self) -> GestureReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class EffectGate:
    """Turns a stream of gesture readings into a stable on/off decision."""

    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        self._debouncer = Debouncer(
            rise_s=self.config.engage_hold_s,
            fall_s=self.config.release_hold_s,
            initial=False,
        )
        self._unknown_since: float | None = None

    @property
    def engaged(self) -> bool:
        return self._debouncer.state

    def update(self, reading: GestureReading, now: float) -> bool:
        """Fold in one reading and return whether the effect should be on."""
        observation = reading.observation

        if observation is None:
            if self._unknown_since is None:
                self._unknown_since = now
            # Hands leave frame constantly. Holding indefinitely would strand a
            # user who walked away mid-effect, so fail safe back to visible
            # after a while — but a whole second later, not on the first miss.
            elif (
                self._debouncer.state
                and now - self._unknown_since >= self.config.lost_hand_timeout_s
            ):
                self._debouncer.force(False)
                self._unknown_since = None
                log.debug("no hand for %.1fs — restoring", self.config.lost_hand_timeout_s)
        else:
            self._unknown_since = None

        return self._debouncer.update(observation, now)

    def reset(self) -> None:
        self._debouncer.force(False)
        self._unknown_since = None


class GestureWorker:
    """Runs a :class:`GestureReader` on its own thread.

    Recognition measured 16.6 ms per frame. Run inline that is half the frame
    budget spent before compositing has started; on a worker it overlaps with
    segmentation and rendering, and the gesture simply updates at whatever rate
    it manages.

    The thread must be joined before the reader is closed — closing a MediaPipe
    task while a call is in flight is a native crash, not an exception.
    """

    def __init__(self, reader: GestureReader) -> None:
        self._reader = reader
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._reading = GestureReading(GestureState.UNKNOWN)
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="deleteme-gesture", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Offer a frame. Replaces any frame not yet picked up."""
        with self._lock:
            self._pending = frame
        self._wake.set()

    @property
    def reading(self) -> GestureReading:
        with self._lock:
            return self._reading

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.1):
                continue
            self._wake.clear()
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            try:
                reading = self._reader.read(frame)
            except Exception as exc:  # pragma: no cover - native failure
                self.error = exc
                log.exception("gesture worker failed")
                return
            with self._lock:
                self._reading = reading

    def stop(self) -> None:
        """Stop and join. Safe to call more than once."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():  # pragma: no cover - pathological hang
                log.warning("gesture worker did not stop within 2s")
        self._thread = None

    def close(self) -> None:
        """Stop the thread, *then* release the native task.

        The order is mandatory rather than tidy: closing a MediaPipe task while
        a recognition call is still running is an access violation in native
        code, which no Python-level exception handling can catch.
        """
        self.stop()
        self._reader.close()
