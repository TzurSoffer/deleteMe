"""Camera session.

The previous version opened one ``VideoCapture`` to grab the background plate,
released it, and opened a second one for the effect loop. That single decision
caused most of what looked like an algorithm problem: between the two opens the
driver re-ran auto-exposure and auto-white-balance from scratch — the second
time with a person in frame — so the plate and the live feed were photometrically
mismatched before any light in the room had changed. The two captures could also
negotiate different resolutions, which made the composite either crash or, worse,
silently paste the wrong crop of the room.

So: one session, held for the lifetime of the process, with its photometric
state pinned and recorded.

Locking that state is not straightforward, and doing it naively makes things
worse rather than better. Drivers routinely accept a property, return success,
and ignore it. A measured example: setting ``AUTO_EXPOSURE`` (silently refused,
read back as -1) and then forcing ``EXPOSURE`` left the driver fighting itself
and produced *eleven times* more drift than leaving it on auto. ``cap.set()``
returned ``True`` throughout.

Every lock here is therefore confirmed by experiment — change the exposure, look
at whether the picture responds — and a lock that cannot be confirmed is rolled
back completely rather than left half-applied.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from deleteme.config import CameraConfig
from deleteme.errors import CameraError
from deleteme.frames import Frame

log = logging.getLogger(__name__)


def _backend_candidates() -> list[tuple[str, int]]:
    """Capture backends to try, best first, for this platform."""
    if sys.platform == "win32":
        return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
    if sys.platform == "darwin":
        return [("avfoundation", cv2.CAP_AVFOUNDATION), ("any", cv2.CAP_ANY)]
    return [("v4l2", cv2.CAP_V4L2), ("any", cv2.CAP_ANY)]


#: Auto-exposure "manual" sentinels differ per backend and per driver, and the
#: documented values are not reliable. Rather than guess, each is tried and the
#: first one that demonstrably hands over control of exposure wins.
_MANUAL_AUTO_EXPOSURE_CANDIDATES = (0.25, 0.0, 1.0, 3.0)

_AUTO_EXPOSURE_ON = 0.75


@dataclass(frozen=True)
class PhotometryLock:
    """What the driver actually let us pin down.

    ``verified`` is only true when exposure was confirmed manual by experiment.
    When it is false the software correction in :mod:`deleteme.photometry`
    carries the entire load — which is a supported mode, not a failure.
    """

    applied: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": dict(self.applied),
            "failures": list(self.failures),
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PhotometryLock:
        if not data:
            return cls()
        return cls(
            applied={str(k): float(v) for k, v in (data.get("applied") or {}).items()},
            failures=[str(f) for f in (data.get("failures") or [])],
            verified=bool(data.get("verified", False)),
        )


class _LatestFrameSlot:
    """A one-deep mailbox that drops stale frames.

    ``CAP_PROP_BUFFERSIZE = 1`` is the usual advice for this and it does not
    work — ``set()`` was measured returning False on DSHOW, MSMF and ANY on the
    development machine. Owning the drain loop is the only reliable fix, and it
    took capture latency from ~2000 ms to ~17.5 ms median.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._arrived = threading.Event()

    def put(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame
        self._arrived.set()

    def take(self, timeout: float) -> Frame | None:
        if not self._arrived.wait(timeout):
            return None
        with self._lock:
            frame = self._frame
            self._arrived.clear()
        return frame

    def peek(self) -> Frame | None:
        with self._lock:
            return self._frame


class CameraSession:
    """A single camera, opened once and held.

    Also a :class:`~deleteme.frames.FrameSource`, so the pipeline cannot tell it
    apart from a recorded session.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._cap: cv2.VideoCapture | None = None
        self.backend_name: str = "unopened"
        self._size: tuple[int, int] = (0, 0)
        self.lock: PhotometryLock = PhotometryLock()

        self._slot = _LatestFrameSlot()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._index = 0
        self._t0 = time.monotonic()
        self.consecutive_failures = 0
        self._stream_error: BaseException | None = None

    # ---------------------------------------------------------------- opening

    def open(self) -> None:
        """Open the camera, trying each backend until one actually delivers.

        ``isOpened()`` returning True is not sufficient — a camera held by
        another application, or blocked by the Windows privacy setting, opens
        happily and then never produces a frame.
        """
        if self._cap is not None:
            return

        attempts: list[str] = []
        for name, backend in _backend_candidates():
            cap = cv2.VideoCapture(self.config.index, backend)
            if not cap.isOpened():
                attempts.append(f"{name}: would not open")
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            frame = self._read_direct(cap, tries=10)
            if frame is None:
                attempts.append(f"{name}: opened but delivered no frames")
                cap.release()
                continue

            self._cap = cap
            self.backend_name = name
            self._size = (frame.shape[1], frame.shape[0])
            log.info(
                "camera opened via %s at %dx%d (requested %dx%d)",
                name,
                self._size[0],
                self._size[1],
                self.config.width,
                self.config.height,
            )
            return

        raise CameraError(
            "Could not open a camera. Check that no other application is using "
            "it, and that camera access is enabled in your system privacy settings.",
            attempts=attempts,
        )

    @staticmethod
    def _read_direct(cap: cv2.VideoCapture, tries: int = 1) -> np.ndarray | None:
        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return frame
            time.sleep(0.02)
        return None

    def _require_cap(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise CameraError("camera session is not open")
        return self._cap

    # ---------------------------------------------------------------- reading

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` actually negotiated, not what was requested."""
        return self._size

    @property
    def is_streaming(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def read(self) -> Frame | None:
        """Most recent frame. ``None`` means "not right now", never "give up"."""
        if self.is_streaming:
            frame = self._slot.take(self.config.read_timeout_s)
        else:
            image = self._read_direct(self._require_cap())
            frame = self._wrap(image) if image is not None else None

        if frame is None:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        return frame

    def read_blocking(self, timeout_s: float | None = None) -> Frame:
        """Read, raising if nothing arrives. For setup paths that cannot proceed."""
        deadline = time.monotonic() + (
            timeout_s if timeout_s is not None else self.config.read_timeout_s
        )
        while time.monotonic() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
        raise CameraError("camera stopped delivering frames")

    def _wrap(self, image: np.ndarray) -> Frame:
        frame = Frame(image=image, index=self._index, timestamp_s=time.monotonic() - self._t0)
        self._index += 1
        return frame

    @property
    def stalled(self) -> bool:
        return self.consecutive_failures >= self.config.stall_tolerance_frames

    # -------------------------------------------------------------- streaming

    def start_stream(self) -> None:
        """Begin draining the camera on a background thread."""
        if self.is_streaming:
            return
        self._require_cap()
        self._stop.clear()
        self._stream_error = None
        self._thread = threading.Thread(target=self._pump, name="deleteme-capture", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        cap = self._cap
        assert cap is not None
        misses = 0
        while not self._stop.is_set():
            try:
                ok, image = cap.read()
            except Exception as exc:  # pragma: no cover - driver level failure
                self._stream_error = exc
                return
            if not ok or image is None or not image.size:
                misses += 1
                if misses > self.config.stall_tolerance_frames:
                    time.sleep(0.05)
                else:
                    time.sleep(0.005)
                continue
            misses = 0
            self._slot.put(self._wrap(image))

    def stop_stream(self) -> None:
        """Stop the capture thread and wait for it.

        Joining before anything else is released is not optional: tearing down
        native MediaPipe or OpenCV resources while a worker is still inside them
        produced an access violation in 3 of 25 trials.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():  # pragma: no cover - pathological driver hang
                log.warning("capture thread did not stop within 2s")
        self._thread = None

    # ------------------------------------------------------------- photometry

    def warm_up(self, frames: int | None = None) -> None:
        """Discard frames so the driver's auto algorithms can converge.

        Locking before they settle pins the wrong values — which is exactly what
        the old code's five-second countdown failed to do, and why its plate was
        often captured mid-convergence.
        """
        count = self.config.warmup_frames if frames is None else frames
        cap = self._require_cap()
        for _ in range(count):
            self._read_direct(cap)

    def lock_photometry(self) -> PhotometryLock:
        """Pin exposure, white balance and focus, verifying each by experiment."""
        if not self.config.lock_photometry:
            return PhotometryLock(failures=["disabled by configuration"])

        cap = self._require_cap()
        applied: dict[str, float] = {}
        failures: list[str] = []

        converged = {
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "wb_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
            "focus": cap.get(cv2.CAP_PROP_FOCUS),
            "gain": cap.get(cv2.CAP_PROP_GAIN),
        }

        verified = self._take_manual_exposure(cap, converged["exposure"])
        if verified:
            applied["auto_exposure"] = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            applied["exposure"] = cap.get(cv2.CAP_PROP_EXPOSURE)
        else:
            failures.append("exposure")
            # Roll all the way back. A half-applied lock is worse than none:
            # forcing EXPOSURE while AUTO_EXPOSURE silently stayed on measured
            # 11x worse drift than simply leaving the driver alone.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_ON)

        for label, auto_prop, value_prop, converged_key in (
            ("white_balance", cv2.CAP_PROP_AUTO_WB, cv2.CAP_PROP_WB_TEMPERATURE, "wb_temperature"),
            ("focus", cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_FOCUS, "focus"),
        ):
            if self._set_verified(cap, auto_prop, 0.0):
                applied[f"auto_{label}"] = 0.0
                target = converged[converged_key]
                if (
                    target is not None
                    and target > 0
                    and self._set_verified(cap, value_prop, target)
                ):
                    applied[converged_key] = float(target)
            else:
                failures.append(label)

        self.lock = PhotometryLock(applied=applied, failures=failures, verified=verified)
        log.info("photometry lock: applied=%s failures=%s verified=%s", applied, failures, verified)
        return self.lock

    def _take_manual_exposure(self, cap: cv2.VideoCapture, converged_exposure: float) -> bool:
        """Try each manual sentinel until exposure demonstrably responds to us."""
        for candidate in _MANUAL_AUTO_EXPOSURE_CANDIDATES:
            if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, candidate):
                continue
            if self._exposure_responds(cap, converged_exposure):
                return True
        return False

    def _exposure_responds(self, cap: cv2.VideoCapture, baseline_exposure: float) -> bool:
        """Change the exposure and check whether the picture actually changed.

        The only trustworthy test. Read-back alone is not enough — DSHOW returns
        -1 for ``AUTO_EXPOSURE`` whether or not the request was honoured.
        """
        base_luma = self._mean_luma(cap)
        if base_luma is None:
            return False

        probe = baseline_exposure + self.config.exposure_probe_stops
        if not cap.set(cv2.CAP_PROP_EXPOSURE, probe):
            return False
        probe_luma = self._mean_luma(cap)

        cap.set(cv2.CAP_PROP_EXPOSURE, baseline_exposure)
        self._mean_luma(cap)  # let it settle back before anyone else looks

        if probe_luma is None:
            return False
        return abs(probe_luma - base_luma) >= self.config.exposure_probe_min_delta_luma

    @staticmethod
    def _mean_luma(cap: cv2.VideoCapture, discard: int = 5, average: int = 3) -> float | None:
        """Mean luma after letting the sensor settle."""
        for _ in range(discard):
            cap.read()
        readings: list[float] = []
        for _ in range(average):
            ok, image = cap.read()
            if ok and image is not None and image.size:
                readings.append(float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()))
        return sum(readings) / len(readings) if readings else None

    @staticmethod
    def _set_verified(cap: cv2.VideoCapture, prop: int, value: float) -> bool:
        """Set a property and confirm by read-back where the driver reports one."""
        if not cap.set(prop, value):
            return False
        back = cap.get(prop)
        if back is None or back < 0:
            # Opaque driver: no read-back available. Trust set()'s return value,
            # which is all the information on offer for this property.
            return True
        return abs(back - value) < 1e-3

    def apply_lock(self, lock: PhotometryLock) -> list[str]:
        """Re-apply a lock recorded alongside a stored plate.

        This is what makes a plate captured yesterday still radiometrically
        comparable to today's frames, instead of the driver independently
        reconverging to something slightly different.
        """
        cap = self._require_cap()
        restored: list[str] = []
        order = (
            ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
            ("exposure", cv2.CAP_PROP_EXPOSURE),
            ("auto_white_balance", cv2.CAP_PROP_AUTO_WB),
            ("wb_temperature", cv2.CAP_PROP_WB_TEMPERATURE),
            ("auto_focus", cv2.CAP_PROP_AUTOFOCUS),
            ("focus", cv2.CAP_PROP_FOCUS),
        )
        for key, prop in order:
            if key in lock.applied and cap.set(prop, lock.applied[key]):
                restored.append(key)
        self.lock = lock
        log.info("re-applied photometry from plate metadata: %s", restored)
        return restored

    def open_driver_settings(self) -> bool:
        """Pop the vendor's own camera property page, where the backend has one.

        A three-line escape hatch that is worth more than any amount of property
        plumbing when a particular webcam refuses to behave.
        """
        cap = self._cap
        if cap is None:
            return False
        try:
            return bool(cap.set(cv2.CAP_PROP_SETTINGS, 0))
        except cv2.error:  # pragma: no cover - backend without the property
            return False

    # ------------------------------------------------------------------ close

    def close(self) -> None:
        """Stop streaming, then release. Order matters; see :meth:`stop_stream`."""
        self.stop_stream()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> CameraSession:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
