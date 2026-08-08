"""Blending and the transition."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deleteme.composite import Compositor, Dissolve, smoothstep
from deleteme.frames import solid_frame, textured_frame


def test_zero_alpha_returns_the_frame_bit_for_bit():
    frame = textured_frame(320, 240, seed=1)
    plate = textured_frame(320, 240, seed=2)
    alpha = np.zeros((240, 320), np.float32)

    result = Compositor().compose(frame, plate, alpha, strength=1.0)

    assert np.array_equal(result.image, frame)
    assert result.roi is None


def test_full_alpha_returns_the_plate_bit_for_bit():
    frame = textured_frame(320, 240, seed=1)
    plate = textured_frame(320, 240, seed=2)
    alpha = np.ones((240, 320), np.float32)

    result = Compositor().compose(frame, plate, alpha, strength=1.0)

    assert np.array_equal(result.image, plate)


def test_zero_strength_short_circuits_even_with_a_full_mask():
    frame = textured_frame(320, 240, seed=1)
    plate = textured_frame(320, 240, seed=2)
    alpha = np.ones((240, 320), np.float32)

    result = Compositor().compose(frame, plate, alpha, strength=0.0)

    assert np.array_equal(result.image, frame)
    assert result.strength == 0.0


def test_only_the_masked_region_is_touched():
    frame = solid_frame(320, 240, (10, 20, 30))
    plate = solid_frame(320, 240, (200, 210, 220))
    alpha = np.zeros((240, 320), np.float32)
    alpha[100:140, 100:140] = 1.0

    out = Compositor().compose(frame, plate, alpha, strength=1.0).image

    assert np.array_equal(out[100:140, 100:140], plate[100:140, 100:140])
    assert np.array_equal(out[:90, :90], frame[:90, :90])
    assert np.array_equal(out[200:, 200:], frame[200:, 200:])


def test_half_alpha_lands_between_the_two():
    frame = solid_frame(64, 64, (0, 0, 0))
    plate = solid_frame(64, 64, (200, 200, 200))
    alpha = np.full((64, 64), 0.5, np.float32)

    out = Compositor().compose(frame, plate, alpha, strength=1.0).image

    assert out[32, 32, 0] == pytest.approx(100, abs=1)


def test_no_pure_green_pixel_appears_in_the_output():
    """A regression guard for the shipped effect's most visible defect.

    The previous version drew a green debug rectangle and *then* pasted the
    background inside it, so a two-pixel green outline survived around the very
    thing being hidden — 1291 pixels of it, in the published demo.
    """
    frame = textured_frame(320, 240, seed=3)
    plate = textured_frame(320, 240, seed=4)
    alpha = np.zeros((240, 320), np.float32)
    alpha[50:200, 50:250] = 1.0

    out = Compositor().compose(frame, plate, alpha, strength=1.0).image

    pure_green = np.all(out == np.array([0, 255, 0], np.uint8), axis=2)
    assert not pure_green.any()


def test_output_buffer_is_reused_across_frames():
    """Allocating a fresh full-frame buffer every frame is measurable."""
    compositor = Compositor()
    frame = textured_frame(320, 240, seed=5)
    plate = textured_frame(320, 240, seed=6)
    alpha = np.zeros((240, 320), np.float32)

    first = compositor.compose(frame, plate, alpha).image
    second = compositor.compose(frame, plate, alpha).image

    assert first is second


def test_mismatched_plate_is_rejected():
    compositor = Compositor()
    frame = textured_frame(320, 240)
    alpha = np.ones((240, 320), np.float32)

    with pytest.raises(ValueError, match="does not match"):
        compositor.compose(frame, textured_frame(640, 480), alpha, strength=1.0)


def test_dissolve_eases_from_zero_to_one():
    dissolve = Dissolve(0.2)
    dissolve.target(True, now=0.0)

    assert dissolve.value(0.0) == pytest.approx(0.0)
    assert dissolve.value(0.1) == pytest.approx(0.5, abs=0.01)
    assert dissolve.value(0.2) == pytest.approx(1.0)
    assert dissolve.value(5.0) == pytest.approx(1.0)


def test_reversing_mid_dissolve_is_continuous():
    """Reversing must resume from where the ramp is, not snap to an end."""
    dissolve = Dissolve(0.2)
    dissolve.target(True, now=0.0)
    midpoint = dissolve.value(0.1)

    dissolve.target(False, now=0.1)

    assert dissolve.value(0.1) == pytest.approx(midpoint, abs=1e-6)
    assert dissolve.value(0.3) == pytest.approx(0.0)


def test_smoothstep_has_zero_slope_at_both_ends():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(-5.0) == 0.0
    assert smoothstep(5.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
    # Eased, so it moves slower than linear near the start.
    assert smoothstep(0.1) < 0.1


def test_composite_roi_covers_the_mask():
    frame = textured_frame(320, 240, seed=7)
    plate = textured_frame(320, 240, seed=8)
    alpha = np.zeros((240, 320), np.float32)
    alpha[60:100, 70:130] = 1.0

    result = Compositor().compose(frame, plate, alpha, strength=1.0, margin=4)

    assert result.roi is not None
    x, y, w, h = result.roi
    assert x <= 70 and y <= 60
    assert x + w >= 130 and y + h >= 100


def test_a_feathered_edge_produces_intermediate_values():
    frame = solid_frame(128, 128, (0, 0, 0))
    plate = solid_frame(128, 128, (255, 255, 255))
    hard = np.zeros((128, 128), np.uint8)
    hard[40:90, 40:90] = 255
    alpha = cv2.GaussianBlur(hard, (9, 9), 0).astype(np.float32) / 255.0

    out = Compositor().compose(frame, plate, alpha, strength=1.0).image

    values = set(np.unique(out))
    assert len(values - {0, 255}) > 0, "a feathered mask must produce a gradient"
