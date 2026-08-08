"""Fitting the plate to the current light.

The two tests that matter most here are the ones that would have caught a bug
that nearly shipped: ``cv2.convertScaleAbs`` inverting negative corrections, and
a fit quietly poisoned by the person it was supposed to exclude.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deleteme.config import PhotometryConfig
from deleteme.frames import solid_frame, textured_frame
from deleteme.photometry import PhotometryTracker, apply_lut, build_lut, fit_gain_bias


def relight(image, gain, bias):
    """Apply a known photometric change, the way a room's lighting would."""
    return np.clip(image.astype(np.float32) * np.array(gain) + np.array(bias), 0, 255).astype(
        np.uint8
    )


def test_fit_recovers_a_known_gain_and_bias():
    plate = textured_frame(640, 480, seed=1)
    gain, bias = (1.20, 1.15, 1.25), (-10.0, -6.0, -14.0)
    frame = relight(plate, gain, bias)

    fit = fit_gain_bias(plate, frame, None)

    assert fit is not None
    assert fit.gain == pytest.approx(gain, abs=0.02)
    assert fit.bias == pytest.approx(bias, abs=1.0)
    assert not fit.degenerate


def test_correction_removes_the_difference_it_measured():
    plate = textured_frame(640, 480, seed=2)
    frame = relight(plate, (1.2, 1.2, 1.2), (-10.0, -10.0, -10.0))

    fit = fit_gain_bias(plate, frame, None)
    assert fit is not None
    corrected = apply_lut(plate, build_lut(fit.gain, fit.bias))

    before = np.median(cv2.absdiff(plate, frame))
    after = np.median(cv2.absdiff(corrected, frame))
    assert after < 2.0
    assert after < before / 4


def test_lut_clamps_a_negative_bias_instead_of_taking_its_absolute_value():
    """cv2.convertScaleAbs(10, alpha=1, beta=-30) returns 20, not 0.

    Using it here would invert every darkening correction: a plate that needed
    to get darker would get brighter instead. This is the guard against anyone
    reintroducing it as an "optimisation".
    """
    lut = build_lut(np.ones(3, np.float32), np.full(3, -30.0, np.float32))
    probe = np.full((1, 1, 3), 10, np.uint8)

    assert int(apply_lut(probe, lut)[0, 0, 0]) == 0
    assert int(cv2.convertScaleAbs(probe, alpha=1, beta=-30)[0, 0, 0]) == 20


def test_lut_saturates_instead_of_wrapping():
    """(200 * 2) as uint8 wraps to 144 — bright pixels would turn dark."""
    lut = build_lut(np.full(3, 2.0, np.float32), np.zeros(3, np.float32))
    probe = np.full((1, 1, 3), 200, np.uint8)

    assert int(apply_lut(probe, lut)[0, 0, 0]) == 255
    assert np.uint8(np.int32(200 * 2).astype(np.uint8)) == 144  # the trap, documented


def test_lut_is_applied_per_channel():
    lut = build_lut(np.ones(3, np.float32), np.array([0.0, 50.0, 100.0], np.float32))
    probe = np.full((1, 1, 3), 10, np.uint8)

    assert list(apply_lut(probe, lut)[0, 0]) == [10, 60, 110]


def test_gain_is_pinned_on_a_flat_plate():
    """On a blank wall gain is unidentifiable.

    A true gain of 1.1 fits as gain 1.0002 with a bias of +11.97 — perfect on
    those pixels, meaningless as parameters, and catastrophic to extrapolate to
    a brighter part of the scene.
    """
    plate = solid_frame(320, 240, (128, 128, 128))
    frame = relight(plate, (1.1, 1.1, 1.1), (0.0, 0.0, 0.0))

    fit = fit_gain_bias(plate, frame, None)

    assert fit is not None
    assert fit.degenerate
    assert fit.gain == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert fit.bias == pytest.approx([12.8, 12.8, 12.8], abs=1.5)


def test_the_mask_actually_gates_the_fit():
    """Half of this test is the negative control.

    Asserting only that the masked fit is right would pass even if the mask
    were ignored entirely on some inputs. Asserting that the *unmasked* fit is
    measurably wrong proves the mask is load-bearing.
    """
    plate = textured_frame(640, 480, seed=3)
    gain, bias = (1.2, 1.2, 1.2), (-10.0, -10.0, -10.0)
    frame = relight(plate, gain, bias)

    frame[:, :260] = 0  # a large dark subject
    mask = np.ones((480, 640), np.uint8)
    mask[:, :300] = 0  # excluded, with margin

    gated = fit_gain_bias(plate, frame, mask)
    ungated = fit_gain_bias(plate, frame, None)

    assert gated is not None and ungated is not None
    assert gated.gain == pytest.approx(gain, abs=0.02)
    assert ungated.gain != pytest.approx(gain, abs=0.05)


def test_fit_is_refused_when_too_little_background_is_visible():
    plate = textured_frame(320, 240, seed=4)
    frame = relight(plate, (1.1, 1.1, 1.1), (0.0, 0.0, 0.0))
    mask = np.zeros((240, 320), np.uint8)
    mask[:4, :4] = 1

    assert fit_gain_bias(plate, frame, mask) is None


def test_tracker_holds_its_parameters_when_the_fit_is_refused():
    plate = textured_frame(320, 240, seed=5)
    frame = relight(plate, (1.2, 1.2, 1.2), (0.0, 0.0, 0.0))
    tracker = PhotometryTracker()

    for i in range(60):
        tracker.update(plate, frame, None, now=i / 30)
    settled = tracker.gain.copy()
    assert settled.mean() == pytest.approx(1.2, abs=0.05)

    blind = np.zeros((240, 320), np.uint8)
    assert tracker.update(plate, frame, blind, now=3.0) is None
    assert tracker.frozen
    assert tracker.gain == pytest.approx(settled, abs=1e-6)


def test_tracker_flags_a_correction_that_means_the_scene_changed():
    """Beyond a point the difference is not the light, it is the room."""
    plate = textured_frame(320, 240, seed=6)
    tracker = PhotometryTracker()

    tracker.update(plate, np.zeros_like(plate), None, now=0.0)

    assert tracker.out_of_range
    assert tracker.gain == pytest.approx([1.0, 1.0, 1.0], abs=1e-6), "parameters left alone"


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="differ"):
        fit_gain_bias(textured_frame(320, 240), textured_frame(640, 480), None)


def test_subsampling_does_not_cost_accuracy_on_a_textured_scene():
    plate = textured_frame(640, 480, seed=8)
    frame = relight(plate, (1.2, 1.2, 1.2), (-8.0, -8.0, -8.0))

    fine = fit_gain_bias(plate, frame, None, PhotometryConfig(subsample_stride=2))
    coarse = fit_gain_bias(plate, frame, None, PhotometryConfig(subsample_stride=5))

    assert fine is not None and coarse is not None
    assert coarse.gain == pytest.approx(fine.gain, abs=0.01)
    assert coarse.pixels < fine.pixels / 4
