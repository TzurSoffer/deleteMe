"""Deciding when the room is empty enough to photograph.

No camera and no window. ``PlateCaptureWorker`` lives outside ``app.py``
precisely so this can run anywhere.

The rule under test used to be ``confidence.max() < core_threshold``: a single
pixel anywhere in the frame above threshold marked the room as occupied, for as
long as it lasted, while the status line blamed a person who may have left
minutes earlier. Everywhere else in the project decides "is a person present" by
area, because one pixel is not a person.
"""

from __future__ import annotations

import numpy as np
import pytest

from deleteme.capture import PlateCaptureWorker
from deleteme.config import AppConfig, PlateConfig, SegmentationConfig
from deleteme.frames import SyntheticFrameSource, textured_frame

WIDTH, HEIGHT = 320, 240


class StubSegmenter:
    """Returns a fixed confidence map. No model, no inference."""

    def __init__(self, confidence: np.ndarray) -> None:
        self.confidence_map = confidence

    def confidence(self, frame: np.ndarray) -> np.ndarray:
        return self.confidence_map


def empty_room_frames(count: int = 400, sigma: float = 0.4):
    """A still, well-exposed, in-focus room with a little sensor noise."""
    room = textured_frame(WIDTH, HEIGHT, seed=61)
    rng = np.random.default_rng(3)
    return [
        np.clip(room.astype(np.float32) + rng.normal(0, sigma, room.shape), 0, 255).astype(np.uint8)
        for _ in range(count)
    ]


def run_worker(confidence: np.ndarray, frames=None, timeout_s: float = 5.0):
    config = AppConfig(plate=PlateConfig(required_good_frames=8, timeout_s=timeout_s))
    source = SyntheticFrameSource(frames or empty_room_frames(), loop=True)
    worker = PlateCaptureWorker(source, StubSegmenter(confidence), config, relock=False)  # type: ignore[arg-type]
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "the capture worker hung"
    return worker


def no_person() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), np.float32)


def a_person() -> np.ndarray:
    field = np.zeros((HEIGHT, WIDTH), np.float32)
    field[60:220, 110:210] = 1.0
    return field


class TestEmptinessIsDecidedByArea:
    def test_an_empty_room_is_captured(self):
        worker = run_worker(no_person())

        assert worker.error is None, worker.error
        assert worker.plate is not None
        assert worker.plate.metadata.frames_averaged >= 8

    def test_a_stray_hot_pixel_does_not_block_capture(self):
        """The regression.

        Under the old rule this frame was 'occupied' forever, so capture could
        never complete and the message blamed a person who was not there.
        """
        field = no_person()
        field[100, 200] = 1.0
        field[10:12, 30:32] = 1.0  # and a four-pixel speck

        worker = run_worker(field)

        assert worker.error is None, worker.error
        assert worker.plate is not None

    def test_a_person_still_blocks_capture(self):
        """The rule must not have been loosened into uselessness."""
        worker = run_worker(a_person(), timeout_s=2.0)

        assert worker.plate is None
        assert worker.error is not None
        assert "step out" in str(worker.error).lower()

    def test_the_failure_says_how_much_of_the_frame_is_occupied(self):
        """So a wrong verdict is diagnosable instead of merely frustrating."""
        worker = run_worker(a_person(), timeout_s=2.0)

        assert worker.error is not None
        assert "%" in str(worker.error)
        assert worker.occupied_fraction > 0.1

    def test_a_blob_below_the_area_floor_is_ignored(self):
        cfg = SegmentationConfig()
        field = no_person()
        side = int((cfg.min_component_frame_fraction * HEIGHT * WIDTH) ** 0.5) - 3
        field[50 : 50 + side, 50 : 50 + side] = 1.0

        worker = run_worker(field)

        assert worker.plate is not None


class TestGuidance:
    def test_the_message_tells_the_user_what_to_do(self):
        config = AppConfig(plate=PlateConfig(required_good_frames=8, timeout_s=2.0))
        source = SyntheticFrameSource(empty_room_frames(), loop=True)
        worker = PlateCaptureWorker(source, StubSegmenter(a_person()), config, relock=False)  # type: ignore[arg-type]
        worker.start()
        worker.join(timeout=30)

        # An instruction, not a diagnosis.
        assert "step out of shot" in worker.message.lower()

    def test_progress_is_published_while_capturing(self):
        worker = run_worker(no_person())
        assert worker.progress == pytest.approx(1.0)

    def test_a_preview_is_published_for_the_window_to_draw(self):
        worker = run_worker(no_person())
        assert worker.preview is not None
        assert worker.preview.shape == (HEIGHT, WIDTH, 3)


class TestCancellation:
    def test_cancelling_stops_without_a_plate_or_an_error(self):
        config = AppConfig(plate=PlateConfig(required_good_frames=8, timeout_s=30.0))
        source = SyntheticFrameSource(empty_room_frames(), loop=True)
        worker = PlateCaptureWorker(source, StubSegmenter(a_person()), config, relock=False)  # type: ignore[arg-type]
        worker.start()
        worker.cancel()
        worker.join(timeout=10)

        assert not worker.is_alive()
        assert worker.plate is None
        assert worker.error is None, "cancelling is not a failure"
        assert worker.finished.is_set()
