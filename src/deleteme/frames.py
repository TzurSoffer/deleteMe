"""Frame sources.

The single most useful seam in the project. Everything downstream of here takes
a :class:`FrameSource`, so the whole pipeline can be driven from synthetic
numpy arrays in a test, from a recorded session on disk, or from a live camera,
without any of those paths knowing about each other.

Recordings are PNG sequences rather than video. An mp4v round-trip has a mean
error of ~2.3 grey levels, which is the same order of magnitude as the
photometric drift this project exists to measure — a lossy fixture would make
the photometry tests measure the codec instead of the code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured image and when it arrived.

    ``image`` is BGR uint8, the OpenCV convention, and is not copied. Consumers
    that intend to mutate it must copy first.
    """

    image: np.ndarray
    index: int
    timestamp_s: float

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``."""
        return self.image.shape[0], self.image.shape[1]


class FrameSource(Protocol):
    """Anything that can hand out frames in order."""

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the frames this source produces."""
        ...

    def read(self) -> Frame | None:
        """Next frame, or ``None`` when exhausted or temporarily unavailable."""
        ...

    def close(self) -> None: ...


class SyntheticFrameSource:
    """Replays an in-memory sequence of images at a fixed nominal frame rate.

    Used by the tests to drive the pipeline deterministically, including the
    cases that are impractical to stage in front of a real camera: a lighting
    step change, a camera bump, a person entering and leaving.
    """

    def __init__(self, images: Sequence[np.ndarray], fps: float = 30.0, loop: bool = False) -> None:
        if not images:
            raise ValueError("SyntheticFrameSource needs at least one image")
        first = images[0]
        for image in images:
            if image.shape != first.shape:
                raise ValueError("all synthetic frames must share a shape")
        self._images = list(images)
        self._dt = 1.0 / float(fps)
        self._loop = loop
        self._i = 0

    @property
    def size(self) -> tuple[int, int]:
        h, w = self._images[0].shape[:2]
        return w, h

    def read(self) -> Frame | None:
        if self._i >= len(self._images):
            if not self._loop:
                return None
            self._i = 0
        image = self._images[self._i % len(self._images)]
        frame = Frame(image=image, index=self._i, timestamp_s=self._i * self._dt)
        self._i += 1
        return frame

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


class RecordingSink:
    """Writes frames losslessly so a live session can be replayed offline.

    This is what turns "it looked wrong when I waved" into a fixture.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._timestamps: list[float] = []

    def write(self, frame: Frame) -> None:
        path = self.directory / f"{self._count:06d}.png"
        if not cv2.imwrite(str(path), frame.image):
            raise OSError(f"could not write recording frame to {path}")
        self._timestamps.append(frame.timestamp_s)
        self._count += 1

    def close(self) -> None:
        manifest = {"version": 1, "frames": self._count, "timestamps": self._timestamps}
        (self.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def __enter__(self) -> RecordingSink:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ReplayFrameSource:
    """Reads back a :class:`RecordingSink` directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._paths = sorted(self.directory.glob("*.png"))
        if not self._paths:
            raise FileNotFoundError(f"no PNG frames found in {self.directory}")

        manifest_path = self.directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._timestamps = list(manifest.get("timestamps") or [])
        else:
            self._timestamps = []

        probe = cv2.imread(str(self._paths[0]))
        if probe is None:
            raise OSError(f"could not decode {self._paths[0]}")
        self._size = (probe.shape[1], probe.shape[0])
        self._i = 0

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def read(self) -> Frame | None:
        if self._i >= len(self._paths):
            return None
        image = cv2.imread(str(self._paths[self._i]))
        if image is None:
            raise OSError(f"could not decode {self._paths[self._i]}")
        if self._i < len(self._timestamps):
            timestamp = float(self._timestamps[self._i])
        else:
            timestamp = self._i / 30.0
        frame = Frame(image=image, index=self._i, timestamp_s=timestamp)
        self._i += 1
        return frame

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def solid_frame(width: int, height: int, colour: tuple[int, int, int]) -> np.ndarray:
    """A flat BGR image. Convenient for tests; useless as a real plate."""
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :] = colour
    return frame


def textured_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A deterministic textured image with enough variance to fit gain against.

    Photometric fitting needs contrast: on a flat wall the gain term is
    unidentifiable. Tests that exercise the real fit path must use this rather
    than :func:`solid_frame`.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, size=(height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
