"""Capturing, validating and storing the background plate.

The old capture was a five-second countdown that kept whatever frame happened to
be in hand when the timer expired. It never checked that the room was actually
empty, that the exposure was sane, that the lens had focused, or that the image
had stopped changing. If you were still walking out of shot, that was the plate.

This one accepts a plate when it is good rather than when time is up, and says
which check is blocking it while you wait.

The accepted frames are then combined by a per-pixel median. That is free noise
reduction — roughly sqrt(N) — and, unlike a mean, it is immune to one bad frame
sneaking into the stack. The spread of those same frames also gives the sensor's
noise floor for nothing, which is what calibrates the change-detection threshold
later instead of a hardcoded guess that only suits one camera.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deleteme.camera import PhotometryLock
from deleteme.config import PlateConfig
from deleteme.errors import PlateError, PlateMismatchError

log = logging.getLogger(__name__)

PLATE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FrameQuality:
    """Verdict on one candidate frame, with the numbers behind it."""

    scene_empty: bool
    mean_luma: float
    p99_luma: float
    highlight_clip_fraction: float
    focus_variance: float
    stable: bool

    exposure_ok: bool
    focus_ok: bool

    @property
    def ok(self) -> bool:
        return self.scene_empty and self.exposure_ok and self.focus_ok and self.stable

    @property
    def reason(self) -> str:
        """Why this frame was rejected, phrased for a human to act on."""
        if not self.scene_empty:
            return "someone is still in frame"
        if not self.exposure_ok:
            if self.mean_luma < 1e-6:
                return "no signal from the camera"
            if self.highlight_clip_fraction > 0 and self.p99_luma > 250:
                return "the scene is blown out — reduce the light or lower exposure"
            return "the scene is too dark" if self.mean_luma < 100 else "the scene is too bright"
        if not self.focus_ok:
            return "the image is out of focus or motion-blurred"
        if not self.stable:
            return "the picture is still settling"
        return "ok"


def measure_frame(image: np.ndarray, config: PlateConfig | None = None) -> dict[str, float]:
    """Exposure and focus statistics for one frame."""
    cfg = config or PlateConfig()
    del cfg  # thresholds are applied by the caller; this only measures

    luma = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "mean_luma": float(luma.mean()),
        "p99_luma": float(np.percentile(luma, 99)),
        "highlight_clip_fraction": float(np.count_nonzero(luma >= 254) / luma.size),
        "focus_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


class _StabilityWindow:
    """Tracks whether the picture has stopped moving.

    Two signals, because they fail differently: the spread of mean luma catches
    an auto-exposure still hunting, and the mean absolute frame difference
    catches something physically moving in shot.
    """

    def __init__(self, size: int) -> None:
        self.size = max(2, size)
        self._luma: list[float] = []
        self._deltas: list[float] = []
        self._previous_gray: np.ndarray | None = None

    def push(self, image: np.ndarray, mean_luma: float) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is not None:
            self._deltas.append(float(cv2.absdiff(gray, self._previous_gray).mean()))
            del self._deltas[: -self.size]
        self._previous_gray = gray
        self._luma.append(mean_luma)
        del self._luma[: -self.size]

    def verdict(self, max_luma_spread: float, max_frame_delta: float) -> bool:
        if len(self._luma) < self.size or len(self._deltas) < self.size - 1:
            return False
        spread = max(self._luma) - min(self._luma)
        delta = sum(self._deltas) / len(self._deltas)
        return spread <= max_luma_spread and delta <= max_frame_delta

    def reset(self) -> None:
        self._luma.clear()
        self._deltas.clear()
        self._previous_gray = None


@dataclass
class PlateMetadata:
    """Everything needed to decide whether a stored plate is still usable.

    ``locked`` matters most. Re-applying the camera settings the plate was shot
    under is what makes today's frames radiometrically comparable to it, rather
    than letting the driver reconverge to something subtly different.
    """

    version: int = PLATE_FORMAT_VERSION
    width: int = 0
    height: int = 0
    backend: str = "unknown"
    camera_index: int = 0
    lock: PhotometryLock = field(default_factory=PhotometryLock)
    mean_luma: float = 0.0
    focus_variance: float = 0.0
    noise_sigma: float = 0.0
    frames_averaged: int = 0
    capture_method: str = "median"
    captured_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "width": self.width,
            "height": self.height,
            "camera": {"backend": self.backend, "index": self.camera_index},
            "locked": self.lock.as_dict(),
            "stats": {
                "mean_luma": round(self.mean_luma, 3),
                "focus_variance": round(self.focus_variance, 3),
                "noise_sigma": round(self.noise_sigma, 4),
            },
            "frames_averaged": self.frames_averaged,
            "capture_method": self.capture_method,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlateMetadata:
        camera = data.get("camera") or {}
        stats = data.get("stats") or {}
        return cls(
            version=int(data.get("version", 0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            backend=str(camera.get("backend", "unknown")),
            camera_index=int(camera.get("index", 0)),
            lock=PhotometryLock.from_dict(data.get("locked")),
            mean_luma=float(stats.get("mean_luma", 0.0)),
            focus_variance=float(stats.get("focus_variance", 0.0)),
            noise_sigma=float(stats.get("noise_sigma", 0.0)),
            frames_averaged=int(data.get("frames_averaged", 0)),
            capture_method=str(data.get("capture_method", "unknown")),
            captured_at=str(data.get("captured_at", "")),
        )


@dataclass
class Plate:
    """A background plate and the conditions it was shot under."""

    image: np.ndarray
    metadata: PlateMetadata

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    def require_compatible(self, frame: np.ndarray) -> None:
        """Raise unless this plate matches the frames being captured.

        Checked in *both* directions. A plate larger than the frame is the
        quietly dangerous case: the slice succeeds and pastes the wrong part of
        the room, with no error anywhere.
        """
        if self.image.shape[:2] != frame.shape[:2]:
            raise PlateMismatchError(self.image.shape, frame.shape)

    def matches_camera(self, backend: str, index: int) -> bool:
        """Whether this plate was shot on the same camera and backend.

        Backends are not interchangeable: DSHOW and MSMF were measured producing
        materially different pixel values for the same scene.
        """
        return self.metadata.backend == backend and self.metadata.camera_index == index


class PlateCapture:
    """Collects consecutive good frames and stacks them into a plate.

    Consecutive is deliberate. A run of ten good frames means the scene really
    has settled; ten good frames scattered among rejects means someone is
    wandering in and out of shot.
    """

    def __init__(self, config: PlateConfig | None = None) -> None:
        self.config = config or PlateConfig()
        self._accepted: list[np.ndarray] = []
        self._stability = _StabilityWindow(self.config.stability_window)
        self._started: float | None = None
        self._last_reason = "waiting for the first frame"

    @property
    def accepted(self) -> int:
        return len(self._accepted)

    @property
    def required(self) -> int:
        return self.config.required_good_frames

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def progress(self) -> float:
        return min(1.0, self.accepted / max(1, self.required))

    def reset(self) -> None:
        self._accepted.clear()
        self._stability.reset()
        self._started = None
        self._last_reason = "waiting for the first frame"

    def offer(self, image: np.ndarray, scene_empty: bool, now: float) -> FrameQuality:
        """Consider one frame. Returns the verdict, including why it failed."""
        cfg = self.config
        if self._started is None:
            self._started = now

        stats = measure_frame(image, cfg)
        self._stability.push(image, stats["mean_luma"])

        exposure_ok = (
            cfg.min_mean_luma < stats["mean_luma"] < cfg.max_mean_luma
            and stats["p99_luma"] < cfg.max_p99_luma
            and stats["highlight_clip_fraction"] < cfg.max_highlight_clip_fraction
        )
        focus_ok = stats["focus_variance"] > cfg.min_focus_variance
        stable = self._stability.verdict(cfg.max_luma_spread, cfg.max_frame_delta)

        quality = FrameQuality(
            scene_empty=scene_empty,
            mean_luma=stats["mean_luma"],
            p99_luma=stats["p99_luma"],
            highlight_clip_fraction=stats["highlight_clip_fraction"],
            focus_variance=stats["focus_variance"],
            stable=stable,
            exposure_ok=exposure_ok,
            focus_ok=focus_ok,
        )
        self._last_reason = quality.reason

        if quality.ok:
            self._accepted.append(image.copy())
        else:
            self._accepted.clear()

        return quality

    @property
    def complete(self) -> bool:
        return self.accepted >= self.required

    def timed_out(self, now: float) -> bool:
        if self._started is None:
            return False
        return (now - self._started) >= self.config.timeout_s

    def build(self, backend: str, camera_index: int, lock: PhotometryLock) -> Plate:
        """Stack the accepted frames into a plate."""
        if not self._accepted:
            raise PlateError("no frames were accepted")

        stack = np.stack(self._accepted).astype(np.float32)
        median = np.median(stack, axis=0)
        image = np.clip(median, 0, 255).astype(np.uint8)

        # Temporal spread across the stack is this sensor's noise floor at this
        # exposure. It calibrates the change-detection threshold for free, on
        # this camera, instead of a constant tuned on someone else's.
        noise_sigma = float(np.median(stack.std(axis=0))) if len(self._accepted) > 1 else 0.0

        stats = measure_frame(image, self.config)
        metadata = PlateMetadata(
            width=image.shape[1],
            height=image.shape[0],
            backend=backend,
            camera_index=camera_index,
            lock=lock,
            mean_luma=stats["mean_luma"],
            focus_variance=stats["focus_variance"],
            noise_sigma=noise_sigma,
            frames_averaged=len(self._accepted),
            capture_method="median",
            captured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        log.info(
            "plate built from %d frames (noise sigma %.3f, mean luma %.1f)",
            len(self._accepted),
            noise_sigma,
            stats["mean_luma"],
        )
        return Plate(image=image, metadata=metadata)


class PlateStore:
    """Reads and writes the plate and its sidecar."""

    def __init__(self, image_path: Path, metadata_path: Path) -> None:
        self.image_path = Path(image_path)
        self.metadata_path = Path(metadata_path)

    def save(self, plate: Plate) -> None:
        """Write plate and metadata.

        PNG, not JPEG. The composite puts these pixels directly against live
        camera pixels; JPEG ringing along that boundary is visible, and the old
        code compounded it by replacing the pristine in-memory plate with the
        decoded one.
        """
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.image_path), plate.image):
            raise PlateError(f"could not write the background plate to {self.image_path}")
        self.metadata_path.write_text(
            json.dumps(plate.metadata.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        log.info("plate saved to %s", self.image_path)

    def load(self) -> Plate | None:
        """Load a stored plate, or ``None`` if there is not a usable one.

        Never raises for an absent or corrupt plate — a missing background is an
        ordinary first-run state, not an error.
        """
        if not self.image_path.is_file():
            return None
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("stored plate at %s could not be decoded; ignoring", self.image_path)
            return None

        metadata = PlateMetadata(width=image.shape[1], height=image.shape[0])
        if self.metadata_path.is_file():
            try:
                metadata = PlateMetadata.from_dict(
                    json.loads(self.metadata_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                log.warning(
                    "plate metadata at %s is unreadable (%s); ignoring", self.metadata_path, exc
                )

        if metadata.version > PLATE_FORMAT_VERSION:
            log.warning("plate was written by a newer version of DeleteMe; ignoring")
            return None

        if (metadata.width, metadata.height) != (image.shape[1], image.shape[0]):
            # Trust the pixels over the sidecar.
            metadata.width, metadata.height = image.shape[1], image.shape[0]

        return Plate(image=image, metadata=metadata)

    def clear(self) -> None:
        self.image_path.unlink(missing_ok=True)
        self.metadata_path.unlink(missing_ok=True)
