"""Mask stabilisation.

No model is loaded here. The stabiliser takes a confidence array, so the
interesting cases — two people, a jittering edge, a distant subject — can be
written directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deleteme.config import SegmentationConfig
from deleteme.segment import MaskStabilizer

WIDTH, HEIGHT = 320, 240


def confidence_with(*blobs, value=1.0):
    """A confidence map containing the given ``(cx, cy, rx, ry)`` ellipses."""
    field = np.zeros((HEIGHT, WIDTH), np.float32)
    for cx, cy, rx, ry in blobs:
        cv2.ellipse(field, (cx, cy), (rx, ry), 0, 0, 360, float(value), -1)
    return field


def settle(stabilizer, field, frames=30):
    """Run the EMA to convergence."""
    result = None
    for _ in range(frames):
        result = stabilizer.update(field)
    return result


def test_an_empty_confidence_map_produces_no_mask():
    result = MaskStabilizer().update(np.zeros((HEIGHT, WIDTH), np.float32))

    assert not result.present
    assert result.area_fraction == 0.0
    assert result.alpha.max() == 0.0


def test_a_confident_blob_becomes_a_silhouette():
    stabilizer = MaskStabilizer()
    result = settle(stabilizer, confidence_with((160, 120, 30, 70)))

    assert result.present
    assert result.alpha[120, 160] == pytest.approx(1.0, abs=0.01)
    assert result.binary[120, 160] == 255


def test_two_people_are_both_kept():
    """An area rule, not "keep the largest".

    The version this replaced kept only the biggest detection, so a second
    person in shot was never removed. Largest-only would silently reintroduce
    that, which is why the smaller blob here is well above the area floor.
    """
    stabilizer = MaskStabilizer()
    result = settle(stabilizer, confidence_with((80, 120, 30, 60), (240, 120, 22, 45)))

    assert result.binary[120, 80] == 255
    assert result.binary[120, 240] == 255


def test_a_speck_below_the_area_floor_is_dropped():
    stabilizer = MaskStabilizer()
    result = settle(stabilizer, confidence_with((160, 120, 40, 70), (300, 20, 2, 2)))

    assert result.binary[120, 160] == 255
    assert result.binary[20, 300] == 0


def test_a_medium_confidence_blob_alone_is_ignored():
    """Hysteresis: shell pixels join only if they connect to a confident core."""
    stabilizer = MaskStabilizer(SegmentationConfig(core_threshold=0.65, shell_threshold=0.35))
    result = settle(stabilizer, confidence_with((160, 120, 40, 70), value=0.5))

    assert not result.present


def test_a_shell_connected_to_a_core_is_included():
    field = confidence_with((160, 120, 40, 70), value=0.5)
    cv2.ellipse(field, (160, 120), (15, 30), 0, 0, 360, 0.9, -1)

    result = settle(MaskStabilizer(), field)

    assert result.present
    assert result.binary[120, 130] == 255, "the 0.5 shell joined its 0.9 core"


def test_the_mask_is_smoothed_over_time_rather_than_snapping():
    """The network runs at 256x256 and is upsampled about 2.5x, so a pixel of
    model jitter becomes several pixels of crawling edge without this."""
    stabilizer = MaskStabilizer()
    field = confidence_with((160, 120, 40, 70))

    first = stabilizer.update(np.zeros((HEIGHT, WIDTH), np.float32))
    second = stabilizer.update(field)
    eventual = settle(stabilizer, field)

    assert first.area_fraction == 0.0
    assert second.area_fraction < eventual.area_fraction


def test_reset_clears_the_history():
    """So no ghost of the last silhouette fades into the next activation."""
    stabilizer = MaskStabilizer()
    field = confidence_with((160, 120, 40, 70))
    settle(stabilizer, field)

    stabilizer.reset()
    after = stabilizer.update(np.zeros((HEIGHT, WIDTH), np.float32))

    assert not after.present


class TestAdaptiveDilation:
    def test_dilation_grows_with_the_subject(self):
        stabilizer = MaskStabilizer()
        near = stabilizer.dilation_radius(60000)
        far = stabilizer.dilation_radius(3000)
        assert near > far

    def test_dilation_is_clamped_at_both_ends(self):
        cfg = SegmentationConfig()
        stabilizer = MaskStabilizer(cfg)

        assert stabilizer.dilation_radius(0) == cfg.dilate_min_px
        assert stabilizer.dilation_radius(10_000_000) == cfg.dilate_max_px

    def test_the_dilated_mask_is_larger_than_the_silhouette(self):
        """Erring large is nearly free: the pixels pasted are the real
        background, so the seam lands where plate and frame agree. Erring small
        leaves a rim of the subject's own colour, which is the classic halo."""
        stabilizer = MaskStabilizer()
        field = confidence_with((160, 120, 40, 70))
        result = settle(stabilizer, field)

        core = int((field > 0.65).sum())
        assert int((result.binary > 0).sum()) > core


def test_alpha_is_normalised_and_feathered():
    result = settle(MaskStabilizer(), confidence_with((160, 120, 40, 70)))

    assert result.alpha.dtype == np.float32
    assert 0.0 <= result.alpha.min() <= result.alpha.max() <= 1.0
    intermediate = np.count_nonzero((result.alpha > 0.05) & (result.alpha < 0.95))
    assert intermediate > 0, "a feathered edge must have partial coverage"
