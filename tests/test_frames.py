"""Frame sources and the recording round trip."""

from __future__ import annotations

import numpy as np
import pytest

from deleteme.frames import (
    Frame,
    RecordingSink,
    ReplayFrameSource,
    SyntheticFrameSource,
    solid_frame,
    textured_frame,
)


def test_synthetic_source_hands_out_frames_in_order():
    images = [solid_frame(32, 24, (i, i, i)) for i in range(5)]
    source = SyntheticFrameSource(images, fps=30)

    assert source.size == (32, 24)
    for i in range(5):
        frame = source.read()
        assert frame is not None
        assert frame.index == i
        assert frame.timestamp_s == pytest.approx(i / 30)
    assert source.read() is None


def test_synthetic_source_can_loop():
    source = SyntheticFrameSource([solid_frame(8, 8, (1, 1, 1))], loop=True)
    for _ in range(4):
        assert source.read() is not None


def test_synthetic_source_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="share a shape"):
        SyntheticFrameSource([solid_frame(8, 8, (0, 0, 0)), solid_frame(16, 16, (0, 0, 0))])


def test_synthetic_source_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one"):
        SyntheticFrameSource([])


def test_a_recording_replays_bit_for_bit(tmp_path):
    """Recordings are PNG sequences, not video, on purpose.

    An mp4v round trip has a mean error of about 2.3 grey levels — the same
    order as the photometric drift this project measures. A lossy fixture would
    turn the photometry tests into a measurement of the codec.
    """
    originals = [textured_frame(64, 48, seed=i) for i in range(4)]
    with RecordingSink(tmp_path) as sink:
        for i, image in enumerate(originals):
            sink.write(Frame(image=image, index=i, timestamp_s=i / 30))

    replay = ReplayFrameSource(tmp_path)
    assert replay.size == (64, 48)
    for i, original in enumerate(originals):
        frame = replay.read()
        assert frame is not None
        assert np.array_equal(frame.image, original), "a recorded frame changed on the way back"
        assert frame.timestamp_s == pytest.approx(i / 30)
    assert replay.read() is None


def test_replaying_a_directory_without_frames_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayFrameSource(tmp_path)


def test_frame_reports_its_shape():
    frame = Frame(image=solid_frame(64, 48, (0, 0, 0)), index=0, timestamp_s=0.0)
    assert frame.shape == (48, 64)


class TestSyntheticImagery:
    def test_a_solid_frame_is_flat(self):
        image = solid_frame(16, 16, (10, 20, 30))
        assert image.shape == (16, 16, 3)
        assert list(image[0, 0]) == [10, 20, 30]
        assert image.reshape(-1, 3).var(axis=0).max() == 0

    def test_a_textured_frame_is_deterministic(self):
        assert np.array_equal(textured_frame(64, 48, seed=3), textured_frame(64, 48, seed=3))

    def test_a_textured_frame_has_enough_contrast_to_fit_gain_against(self):
        """On a flat wall the gain term is unidentifiable, so the fixture that
        photometry tests use must not be flat."""
        from deleteme.config import PhotometryConfig

        image = textured_frame(320, 240, seed=4)
        assert image.reshape(-1, 3).var(axis=0).min() > PhotometryConfig().min_plate_variance * 10

    def test_a_textured_frame_reads_as_in_focus(self):
        """The plate quality gate rejects a blurred scene, so a fixture built
        by smooth interpolation would be refused — correctly, but uselessly."""
        from deleteme.config import PlateConfig
        from deleteme.plate import measure_frame

        stats = measure_frame(textured_frame(320, 240, seed=5))
        assert stats["focus_variance"] > PlateConfig().min_focus_variance
