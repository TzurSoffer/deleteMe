"""The background model: photometric tracking, the change mask, and health.

Every scenario here is staged with numpy rather than in front of a camera. A
lighting step, a camera bump and a person entering are all trivial to write and
impossible to reproduce on demand in a room.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deleteme.background import BackgroundModel, BumpDetector, HealthState
from deleteme.config import AppConfig, ChangeMaskConfig
from deleteme.errors import PlateMismatchError
from deleteme.frames import solid_frame, textured_frame
from deleteme.plate import Plate, PlateMetadata

WIDTH, HEIGHT = 320, 240


@pytest.fixture
def room():
    return textured_frame(WIDTH, HEIGHT, seed=31)


@pytest.fixture
def model(room):
    model = BackgroundModel(AppConfig())
    model.adopt(Plate(room, PlateMetadata(width=WIDTH, height=HEIGHT, noise_sigma=0.5)))
    return model


def relight(image, gain):
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def person_mask(cx=WIDTH // 2, cy=HEIGHT // 2, rx=30, ry=70):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask


class TestPhotometricTracking:
    def test_the_model_follows_a_lighting_change(self, model, room):
        frame = relight(room, 1.25)
        for i in range(80):
            model.update(frame, None, now=i / 30)

        assert model.photometry.gain.mean() == pytest.approx(1.25, abs=0.05)

    def test_a_large_change_is_learned_rather_than_refused(self, model, room):
        """The deadlock this guards against is subtle and self-concealing.

        The fit normally ignores pixels that differ from the corrected plate,
        so a moved chair cannot drag it. But before any correction exists, a
        real lighting change makes most of the frame differ — 57.5% of pixels
        at a 25% brightening — and the strict mask starves the fit. The model
        then declines to learn the exact event it was built for, and declines
        harder the larger the event is.
        """
        frame = relight(room, 1.25)

        model.update(frame, None, now=0.0)

        assert model.fit_relaxed, "the strict mask should have starved the fit"
        assert not model.photometry.frozen, "and the fallback should have rescued it"

    def test_the_corrected_plate_matches_the_new_lighting(self, model, room):
        frame = relight(room, 1.25)
        for i in range(80):
            model.update(frame, None, now=i / 30)

        before = np.median(cv2.absdiff(room, frame))
        after = np.median(cv2.absdiff(model.corrected_plate(), frame))
        assert after < before / 3

    def test_health_recovers_once_the_fit_has_caught_up(self, model, room):
        frame = relight(room, 1.2)
        for i in range(120):
            health = model.update(frame, None, now=i / 30)

        assert health.state is HealthState.GOOD
        assert health.score is not None and health.score > 70

    def test_a_person_is_kept_out_of_the_fit(self, model, room):
        """Without the exclusion the subject drags the correction with them."""
        mask = person_mask()
        frame = relight(room, 1.2)
        frame[mask > 0] = (20, 18, 16)

        for i in range(80):
            model.update(frame, mask, now=i / 30)

        assert model.photometry.gain.mean() == pytest.approx(1.2, abs=0.06)

    def test_shape_mismatch_is_raised_on_update(self, model):
        with pytest.raises(PlateMismatchError):
            model.update(textured_frame(640, 480), None, now=0.0)


class TestHealthReporting:
    def test_no_plate_reports_unknown_rather_than_a_number(self):
        model = BackgroundModel(AppConfig())
        health = model.update(textured_frame(WIDTH, HEIGHT), None, now=0.0)

        assert health.state is HealthState.UNKNOWN
        assert health.score is None
        assert health.label == "BG --"

    def test_an_unmeasurable_frame_reports_unknown(self, model, room):
        """Inventing a number when the evidence is missing is worse than
        admitting there is none."""
        health = model.update(room, np.full((HEIGHT, WIDTH), 255, np.uint8), now=0.0)

        assert health.state is HealthState.UNKNOWN
        assert health.score is None

    def test_a_replaced_scene_is_eventually_reported_as_bad(self, model):
        """Not a lighting change — a different room entirely."""
        other = textured_frame(WIDTH, HEIGHT, seed=99)
        for i in range(120):
            health = model.update(other, None, now=i / 30)

        assert health.state is HealthState.BAD
        assert health.reason


class TestChangeMask:
    def test_the_silhouette_is_always_removed(self, model, room):
        mask = person_mask()
        frame = room.copy()
        frame[mask > 0] = (20, 18, 16)
        model.update(frame, mask, now=0.0)

        result = model.change_mask(frame, mask, float((mask > 0).mean()))

        assert result[HEIGHT // 2, WIDTH // 2] > 0

    def test_a_shadow_touching_the_subject_is_removed(self, model, room):
        """The gate reaches about one characteristic body size, sqrt(area).

        A real cast shadow falls roughly a body's width away; this one is
        staged at that distance. A shadow several body-widths long is beyond
        what a bounded gate can reach, and reaching further would mean the
        area guard doing all the safety work on its own.
        """
        mask = person_mask(cx=120)
        frame = room.copy()
        frame[mask > 0] = (20, 18, 16)
        shadow = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.ellipse(shadow, (172, 165), (42, 24), 0, 0, 360, 255, -1)
        only_shadow = (shadow > 0) & (mask == 0)
        frame[only_shadow] = (frame[only_shadow] * 0.5).astype(np.uint8)
        model.update(frame, mask, now=0.0)

        result = model.change_mask(frame, mask, float((mask > 0).mean()))

        covered = (result > 0)[only_shadow].mean()
        assert covered > 0.85, f"only {covered:.0%} of the cast shadow was removed"

    def test_an_object_not_touching_the_subject_is_kept(self, model, room):
        """The connectivity rule, on its own.

        The object is placed deliberately *inside* the spatial gate. Put it
        outside, and the gate alone keeps it — so the test passes even with
        connectivity filtering disabled entirely, and pins nothing. Here the
        only thing standing between the mug and deletion is that it does not
        touch the silhouette.
        """
        mask = person_mask(cx=120)
        silhouette_area = float((mask > 0).sum())
        roi_radius = self.model_roi_radius(model, silhouette_area)

        frame = room.copy()
        frame[mask > 0] = (20, 18, 16)
        # ~25 px clear of the body, comfortably within the gate.
        frame[110:135, 178:218] = (245, 245, 245)
        distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        assert distance[110:135, 178:218].min() < roi_radius, "object must be inside the gate"

        model.update(frame, mask, now=0.0)
        result = model.change_mask(frame, mask, silhouette_area / (HEIGHT * WIDTH))

        assert int((result[110:135, 178:218] > 0).sum()) == 0
        assert not model.change_mask_suppressed, "kept by connectivity, not by the area guard"

    @staticmethod
    def model_roi_radius(model, silhouette_area: float) -> float:
        cfg = model.config.change_mask
        radius = cfg.roi_area_coefficient * float(np.sqrt(silhouette_area))
        return min(max(radius, cfg.roi_min_px), cfg.roi_max_px)

    def test_change_beyond_the_gate_is_not_removed_even_when_connected(self, model, room):
        """The spatial gate, on its own.

        A darkened streak running from the subject to the frame edge is a single
        connected component, so connectivity would happily take all of it, and
        it is small enough that the area guard never fires. Only the gate stops
        it — which matters because without a bound, one exposure step turns the
        whole picture into wallpaper.
        """
        mask = person_mask(cx=90, rx=25, ry=55)
        silhouette_area = float((mask > 0).sum())
        roi_radius = self.model_roi_radius(model, silhouette_area)

        frame = room.copy()
        frame[mask > 0] = (20, 18, 16)
        streak = np.zeros((HEIGHT, WIDTH), np.uint8)
        streak[118:126, 90:WIDTH] = 255
        only_streak = (streak > 0) & (mask == 0)
        frame[only_streak] = (frame[only_streak] * 0.45).astype(np.uint8)

        model.update(frame, mask, now=0.0)
        result = model.change_mask(frame, mask, silhouette_area / (HEIGHT * WIDTH))

        far_end = result[118:126, WIDTH - 20 :]
        assert int((far_end > 0).sum()) == 0, "the gate must stop the streak short"
        assert not model.change_mask_suppressed, "stopped by the gate, not the area guard"
        near = result[118:126, 90 : 90 + int(roi_radius) // 2]
        assert int((near > 0).sum()) > 0, "the part inside the gate should still go"

    def test_an_uncorrected_exposure_step_cannot_take_the_frame(self, model, room):
        """The failure this guard exists for.

        With a plate that has not been fitted, every pixel differs after an
        exposure step. An unbounded change mask would replace the whole picture
        with wallpaper; this must fall back to the bare silhouette instead.
        """
        mask = person_mask()
        blown = np.clip(room.astype(np.float32) * 1.7 + 30, 0, 255).astype(np.uint8)
        blown[mask > 0] = (20, 18, 16)

        result = model.change_mask(blown, mask, float((mask > 0).mean()))

        assert model.change_mask_suppressed
        assert np.array_equal(result, mask)

    def test_disabling_the_change_mask_returns_the_silhouette(self, room):
        config = AppConfig(change_mask=ChangeMaskConfig(enabled=False))
        model = BackgroundModel(config)
        model.adopt(Plate(room, PlateMetadata(width=WIDTH, height=HEIGHT)))
        mask = person_mask()

        assert np.array_equal(model.change_mask(room, mask, 0.1), mask)


class TestBumpDetection:
    def test_a_shifted_frame_is_measured(self, model, room):
        shifted = np.roll(room, shift=(0, 6), axis=(0, 1))
        for i in range(40):
            model.update(shifted, None, now=i * 0.2)

        assert model.bump.measurable
        assert model.bump.shift_px > 2.0

    def test_a_still_camera_raises_no_alarm(self, model, room):
        for i in range(60):
            model.update(room, None, now=i * 0.2)

        assert not model.bump.alarm

    def test_a_sustained_shift_eventually_raises_the_alarm(self):
        """Asserted positively, because `alarm` is False in almost every state.

        A test that only ever checks `not alarm` would pass against a detector
        hard-wired to False, which is no test of a detector at all.
        """
        detector = BumpDetector(AppConfig())
        plate = textured_frame(WIDTH, HEIGHT, seed=77)
        plate_gray = self.small_gray(plate)
        shifted_gray = self.small_gray(np.roll(plate, shift=(0, 8), axis=(0, 1)))

        for _ in range(AppConfig().background.drift_sustain_samples + 2):
            detector.update(plate_gray, shifted_gray)

        assert detector.measurable
        assert detector.shift_px > AppConfig().background.drift_trigger_px
        assert detector.alarm, "a sustained shift must eventually be reported"

    def test_the_alarm_stands_down_once_the_camera_is_still_again(self):
        detector = BumpDetector(AppConfig())
        plate = textured_frame(WIDTH, HEIGHT, seed=78)
        plate_gray = self.small_gray(plate)
        shifted_gray = self.small_gray(np.roll(plate, shift=(0, 8), axis=(0, 1)))

        for _ in range(AppConfig().background.drift_sustain_samples + 2):
            detector.update(plate_gray, shifted_gray)
        assert detector.alarm

        for _ in range(AppConfig().background.drift_sustain_samples + 2):
            detector.update(plate_gray, plate_gray)

        assert not detector.alarm

    @staticmethod
    def small_gray(image: np.ndarray) -> np.ndarray:
        divisor = AppConfig().background.drift_scale_divisor
        size = (image.shape[1] // divisor, image.shape[0] // divisor)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def test_a_textureless_scene_is_reported_as_unmeasurable(self):
        """Phase correlation on a blank wall returns a number, and it means
        nothing. Say so rather than acting on it."""
        flat = solid_frame(WIDTH, HEIGHT, (128, 128, 128))
        model = BackgroundModel(AppConfig())
        model.adopt(Plate(flat, PlateMetadata(width=WIDTH, height=HEIGHT)))

        for i in range(30):
            model.update(flat, None, now=i * 0.2)

        assert not model.bump.measurable


class TestChangedFraction:
    def test_the_subject_is_not_counted_as_a_changed_scene(self, model, room):
        """`changed_fraction` measures the scene, not the person.

        The subject differs from the plate by definition — that is what makes
        them the subject. Counting their silhouette made the reading a measure
        of how large they are in frame, which drove the health score down at
        every coverage and produced a red "recapture" prompt in front of a
        perfectly good plate.
        """
        mask = person_mask(rx=70, ry=110)
        frame = room.copy()
        frame[mask > 0] = (15, 12, 10)  # very different from the plate

        health = model.update(frame, mask, now=0.0)

        assert health.changed_fraction < 0.05, (
            f"a good plate reported {health.changed_fraction:.0%} of the scene changed"
        )

    def test_a_genuinely_changed_scene_is_still_counted(self, model, room):
        other = textured_frame(WIDTH, HEIGHT, seed=98)

        health = model.update(other, None, now=0.0)

        assert health.changed_fraction > 0.5

    def test_a_close_up_subject_over_a_good_plate_is_not_called_broken(self, model, room):
        """The user-visible symptom: lean towards the camera, get told the
        background is wrong."""
        mask = np.full((HEIGHT, WIDTH), 255, np.uint8)
        mask[:8, :8] = 0  # a sliver of visible background
        frame = room.copy()
        frame[mask > 0] = (15, 12, 10)

        health = model.update(frame, mask, now=0.0)

        assert health.state is not HealthState.BAD


class TestPlateRefresh:
    def test_a_lasting_scene_change_is_eventually_learned(self, room):
        """Continuous relearning. Without it the plate is a photograph again."""
        config = AppConfig()
        model = BackgroundModel(config)
        model.adopt(Plate(room.copy(), PlateMetadata(width=WIDTH, height=HEIGHT, noise_sigma=0.5)))

        moved = room.copy()
        moved[150:200, 40:100] = (200, 60, 40)  # an object put down and left

        before = int(cv2.absdiff(model.plate_image, moved)[150:200, 40:100].sum())
        for i in range(900):
            model.update(moved, None, now=i / 30)
        after = int(cv2.absdiff(model.plate_image, moved)[150:200, 40:100].sum())

        assert after < before / 2, "the plate never learned a change that stayed put"

    def test_a_person_who_holds_still_cannot_burn_into_the_plate(self, room):
        """Anti-burn-in. A stationary subject is excluded by the person mask,
        and the two-speed commit is the backstop for when that mask fails."""
        model = BackgroundModel(AppConfig())
        model.adopt(Plate(room.copy(), PlateMetadata(width=WIDTH, height=HEIGHT, noise_sigma=0.5)))

        mask = person_mask()
        frame = room.copy()
        frame[mask > 0] = (18, 16, 14)

        untouched = model.plate_image[mask > 0].copy()
        for i in range(900):
            model.update(frame, mask, now=i / 30)

        assert np.array_equal(model.plate_image[mask > 0], untouched), (
            "a stationary person was averaged into the background"
        )

    def test_relearning_stands_down_while_the_light_is_moving(self, room):
        """Averaging a half-adjusted exposure into the plate bakes in a value
        that was never correct at any moment."""
        model = BackgroundModel(AppConfig())
        model.adopt(Plate(room.copy(), PlateMetadata(width=WIDTH, height=HEIGHT, noise_sigma=0.5)))
        original = model.plate_image.copy()

        for i in range(60):
            gain = 1.0 + 0.3 * (i / 60)  # light moving continuously
            model.update(relight(room, gain), None, now=i / 30)

        assert np.array_equal(model.plate_image, original), "the plate was updated mid-transition"


class TestIdleTracking:
    def test_an_empty_room_accumulates_idle_time(self, model, room):
        for i in range(30):
            model.update(room, None, now=i / 30)
        assert model.empty_for(1.0) > 0.9

    def test_someone_entering_resets_it(self, model, room):
        for i in range(30):
            model.update(room, None, now=i / 30)
        model.update(room, person_mask(), now=1.0)

        assert model.empty_for(1.0) == 0.0
