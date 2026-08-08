"""The plate: quality gate, stacking, storage, and shape safety."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from deleteme.camera import PhotometryLock
from deleteme.config import PlateConfig
from deleteme.errors import PlateMismatchError
from deleteme.frames import solid_frame, textured_frame
from deleteme.plate import Plate, PlateCapture, PlateMetadata, PlateStore, measure_frame


@pytest.fixture
def room():
    """A plausible room: textured, mid-exposure, in focus."""
    return textured_frame(320, 240, seed=21)


def noisy(image, rng, sigma=0.5):
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(
        np.uint8
    )


def feed(capture, image, count, *, empty=True, start=0.0, rng=None):
    """Offer a run of frames, returning the last verdict."""
    rng = rng or np.random.default_rng(0)
    verdict = None
    for i in range(count):
        verdict = capture.offer(noisy(image, rng), empty, start + i / 30)
    return verdict


class TestShapeSafety:
    def test_a_larger_plate_is_rejected(self):
        """The quiet direction. Slicing a too-large plate succeeds and pastes
        the wrong crop of the room, with no error raised anywhere."""
        plate = Plate(textured_frame(640, 480), PlateMetadata(width=640, height=480))
        with pytest.raises(PlateMismatchError):
            plate.require_compatible(textured_frame(320, 240))

    def test_a_smaller_plate_is_rejected(self):
        plate = Plate(textured_frame(320, 240), PlateMetadata(width=320, height=240))
        with pytest.raises(PlateMismatchError):
            plate.require_compatible(textured_frame(640, 480))

    def test_a_matching_plate_is_accepted(self):
        plate = Plate(textured_frame(320, 240), PlateMetadata(width=320, height=240))
        plate.require_compatible(textured_frame(320, 240))

    def test_the_error_says_both_sizes(self):
        plate = Plate(textured_frame(640, 480), PlateMetadata(width=640, height=480))
        with pytest.raises(PlateMismatchError) as info:
            plate.require_compatible(textured_frame(320, 240))
        assert "640x480" in str(info.value)
        assert "320x240" in str(info.value)


class TestQualityGate:
    def test_a_clean_settled_scene_is_accepted(self, room):
        capture = PlateCapture(PlateConfig(required_good_frames=5))
        feed(capture, room, 20)
        assert capture.complete

    def test_a_scene_with_someone_in_it_is_refused(self, room):
        capture = PlateCapture(PlateConfig(required_good_frames=5))
        verdict = feed(capture, room, 20, empty=False)
        assert not capture.complete
        assert verdict is not None and "still in frame" in verdict.reason

    def test_a_dark_scene_is_refused(self):
        capture = PlateCapture(PlateConfig(required_good_frames=5))
        verdict = feed(capture, solid_frame(320, 240, (3, 3, 3)), 20)
        assert not capture.complete
        assert verdict is not None and not verdict.exposure_ok

    def test_a_blurred_scene_is_refused(self, room):
        capture = PlateCapture(PlateConfig(required_good_frames=5))
        blurred = cv2.GaussianBlur(room, (31, 31), 0)
        verdict = feed(capture, blurred, 20)
        assert not capture.complete
        assert verdict is not None and not verdict.focus_ok

    def test_acceptance_requires_a_consecutive_run(self, room):
        """Ten good frames scattered among rejects means someone is wandering
        in and out of shot, not that the scene has settled."""
        capture = PlateCapture(PlateConfig(required_good_frames=5))
        rng = np.random.default_rng(1)
        for i in range(40):
            capture.offer(noisy(room, rng), i % 2 == 0, i / 30)
        assert not capture.complete

    def test_progress_is_reported_while_waiting(self, room):
        capture = PlateCapture(PlateConfig(required_good_frames=10))
        feed(capture, room, 12)
        assert 0.0 < capture.progress() <= 1.0

    def test_timeout_is_reported_rather_than_hanging(self, room):
        capture = PlateCapture(PlateConfig(required_good_frames=5, timeout_s=1.0))
        capture.offer(room, False, 0.0)
        assert not capture.timed_out(0.5)
        assert capture.timed_out(1.5)


class TestStacking:
    def test_the_stack_is_a_median_and_suppresses_noise(self, room):
        capture = PlateCapture(PlateConfig(required_good_frames=15))
        rng = np.random.default_rng(2)
        feed(capture, room, 30, rng=rng)
        assert capture.complete

        plate = capture.build("test", 0, PhotometryLock())

        assert plate.metadata.capture_method == "median"
        assert plate.metadata.frames_averaged >= 15
        assert plate.image.shape == room.shape

    def test_noise_sigma_is_measured_from_the_stack(self, room):
        """This is what calibrates change detection to *this* camera rather
        than to a constant tuned on someone else's."""
        capture = PlateCapture(PlateConfig(required_good_frames=10))
        rng = np.random.default_rng(3)
        for i in range(25):
            capture.offer(noisy(room, rng, sigma=2.0), True, i / 30)

        plate = capture.build("test", 0, PhotometryLock())

        assert plate.metadata.noise_sigma == pytest.approx(2.0, rel=0.4)

    def test_building_without_accepted_frames_is_an_error(self):
        from deleteme.errors import PlateError

        with pytest.raises(PlateError):
            PlateCapture().build("test", 0, PhotometryLock())


class TestStorage:
    def test_round_trip_preserves_pixels_and_metadata(self, tmp_path, room):
        store = PlateStore(tmp_path / "plate.png", tmp_path / "plate.json")
        lock = PhotometryLock(applied={"exposure": -6.0}, failures=["focus"], verified=True)
        metadata = PlateMetadata(
            width=320, height=240, backend="dshow", camera_index=1, lock=lock, noise_sigma=0.75
        )

        store.save(Plate(room, metadata))
        loaded = store.load()

        assert loaded is not None
        # PNG, not JPEG: the composite seam puts these pixels against live
        # camera pixels and JPEG ringing along that boundary is visible.
        assert np.array_equal(loaded.image, room)
        assert loaded.metadata.backend == "dshow"
        assert loaded.metadata.camera_index == 1
        assert loaded.metadata.noise_sigma == pytest.approx(0.75)
        assert loaded.metadata.lock.applied["exposure"] == pytest.approx(-6.0)
        assert loaded.metadata.lock.failures == ["focus"]
        assert loaded.metadata.lock.verified is True

    def test_a_missing_plate_is_not_an_error(self, tmp_path):
        assert PlateStore(tmp_path / "none.png", tmp_path / "none.json").load() is None

    def test_unreadable_metadata_does_not_lose_the_plate(self, tmp_path, room):
        store = PlateStore(tmp_path / "plate.png", tmp_path / "plate.json")
        store.save(Plate(room, PlateMetadata(width=320, height=240)))
        (tmp_path / "plate.json").write_text("{not json", encoding="utf-8")

        loaded = store.load()

        assert loaded is not None
        assert np.array_equal(loaded.image, room)

    def test_a_plate_from_a_newer_version_is_ignored(self, tmp_path, room):
        store = PlateStore(tmp_path / "plate.png", tmp_path / "plate.json")
        store.save(Plate(room, PlateMetadata(width=320, height=240)))
        data = json.loads((tmp_path / "plate.json").read_text(encoding="utf-8"))
        data["version"] = 999
        (tmp_path / "plate.json").write_text(json.dumps(data), encoding="utf-8")

        assert store.load() is None

    def test_camera_identity_is_part_of_compatibility(self, room):
        """DSHOW and MSMF were measured producing materially different pixels
        for the same scene, so a plate is not portable between them."""
        plate = Plate(room, PlateMetadata(backend="dshow", camera_index=0))
        assert plate.matches_camera("dshow", 0)
        assert not plate.matches_camera("msmf", 0)
        assert not plate.matches_camera("dshow", 1)

    def test_clear_removes_both_files(self, tmp_path, room):
        store = PlateStore(tmp_path / "plate.png", tmp_path / "plate.json")
        store.save(Plate(room, PlateMetadata(width=320, height=240)))
        store.clear()
        assert store.load() is None


def test_measure_frame_reports_the_statistics_the_gate_uses(room):
    stats = measure_frame(room)
    assert set(stats) == {"mean_luma", "p99_luma", "highlight_clip_fraction", "focus_variance"}
    assert 0 < stats["mean_luma"] < 255
    assert stats["focus_variance"] > 0
