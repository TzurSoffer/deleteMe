"""Capturing a background plate on a worker thread.

Separate from the window for the same reason :mod:`deleteme.cli` is: none of
this is GUI code. Warming the camera and probing its exposure takes seconds and
must not block an event loop, but the logic itself is about cameras and plates,
and keeping it here means it can be tested on a machine with no display.
"""

from __future__ import annotations

import logging
import threading
from time import perf_counter

import numpy as np

from deleteme.camera import CameraSession, PhotometryLock
from deleteme.config import AppConfig
from deleteme.errors import CameraError, DeleteMeError
from deleteme.frames import FrameSource
from deleteme.plate import FrameQuality, Plate, PlateCapture
from deleteme.segment import MaskStabilizer, PersonMask, PersonSegmenter

log = logging.getLogger(__name__)


class PlateCaptureWorker(threading.Thread):
    """Warms the camera, pins its photometry, and collects a plate.

    On a thread because the exposure probe genuinely takes seconds: it changes
    a camera setting and waits to see whether the picture responds, several
    times over. Doing that on the UI thread would freeze the window during the
    one operation the user is actively watching.
    """

    def __init__(
        self,
        source: FrameSource,
        segmenter: PersonSegmenter,
        config: AppConfig,
        relock: bool,
    ) -> None:
        super().__init__(name="deleteme-capture-plate", daemon=True)
        self._source = source
        self._segmenter = segmenter
        self._config = config
        self._relock = relock

        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._preview: np.ndarray | None = None
        self._progress = 0.0
        self._message = "starting the camera"
        self.plate: Plate | None = None
        self.error: BaseException | None = None
        self.finished = threading.Event()
        self.occupied_fraction = 0.0

    # ------------------------------------------------------------- published

    @property
    def preview(self) -> np.ndarray | None:
        with self._lock:
            return self._preview

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    def _publish(self, preview: np.ndarray | None, progress: float, message: str) -> None:
        with self._lock:
            if preview is not None:
                self._preview = preview
            self._progress = progress
            self._message = message

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------- run

    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self.error = exc
            log.exception("background capture failed")
        finally:
            self.finished.set()

    def _run(self) -> None:
        camera = self._source if isinstance(self._source, CameraSession) else None

        if camera is not None:
            self._publish(None, 0.0, "letting the camera settle")
            # The cancel token goes all the way down. Both of these run for
            # seconds — the exposure probe can take 96 camera reads against a
            # driver that accepts every request without honouring it — and a
            # shutdown that cannot get this thread out in time would end up
            # releasing the capture while it is still inside cap.read().
            camera.warm_up(cancel=self._cancel)
            if self._cancel.is_set():
                return
            if self._relock:
                self._publish(None, 0.0, "checking what this camera will let us pin down")
                lock = camera.lock_photometry(cancel=self._cancel)
                if self._cancel.is_set():
                    return
                if not lock.verified:
                    log.info("exposure could not be pinned; software correction will carry it")

        capture = PlateCapture(self._config.plate)
        stabilizer = MaskStabilizer(self._config.segmentation)
        started = perf_counter()
        self.occupied_fraction = 0.0

        while not self._cancel.is_set():
            frame = self._source.read()
            if frame is None:
                if perf_counter() - started > self._config.plate.timeout_s:
                    raise CameraError("the camera stopped delivering frames during capture")
                continue

            # Decided by the same component that decides it everywhere else, so
            # "is a person present" has one definition rather than two. The
            # previous test here was `confidence.max() < core_threshold`, under
            # which a single pixel anywhere above threshold marked the room as
            # occupied indefinitely — with a message blaming a person who had
            # already left, and no way to tell the difference.
            mask = stabilizer.update(self._segmenter.confidence(frame.image))
            self.occupied_fraction = mask.area_fraction

            quality = capture.offer(frame.image, not mask.present, perf_counter())
            self._publish(frame.image, capture.progress(), self._guidance(quality, mask))

            if capture.complete:
                lock = camera.lock if camera is not None else PhotometryLock()
                backend = camera.backend_name if camera is not None else "replay"
                index = camera.config.index if camera is not None else 0
                self.plate = capture.build(backend, index, lock)
                return

            if capture.timed_out(perf_counter()):
                raise DeleteMeError(self._timeout_message(capture, mask))

    def _guidance(self, quality: FrameQuality, mask: PersonMask) -> str:
        """What the user should physically do, not merely what is wrong."""
        if quality.ok:
            return "Hold still — capturing the background"
        if mask.present:
            return (
                f"Step out of shot — the camera can still see someone "
                f"({mask.area_fraction:.0%} of the frame)"
            )
        return quality.reason

    def _timeout_message(self, capture: PlateCapture, mask: PersonMask) -> str:
        seconds = self._config.plate.timeout_s
        if mask.present:
            return (
                f"The camera could still see someone after {seconds:.0f} seconds "
                f"({mask.area_fraction:.0%} of the frame).\n\n"
                "The background has to be photographed with nobody in it. Start the capture, "
                "then step out of the camera's view until it says it has finished."
            )
        return f"Could not get a clean background in {seconds:.0f} seconds — {capture.last_reason}."
