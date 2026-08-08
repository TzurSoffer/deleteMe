"""Structuring-element geometry.

This module exists because conflating a radius with a kernel size silently
halved every dilation in the project, which left a cast shadow 27 px from its
owner outside a gate documented as reaching 40 px.
"""

from __future__ import annotations

import cv2
import numpy as np

from deleteme.frames import textured_frame
from deleteme.morphology import channel_max, dilate_scaled, ellipse_kernel


def test_kernel_radius_is_a_radius_not_a_size():
    assert ellipse_kernel(3).shape == (7, 7)
    assert ellipse_kernel(1).shape == (3, 3)
    assert ellipse_kernel(0).shape == (1, 1)


def test_dilating_by_a_radius_grows_a_shape_by_that_radius():
    mask = np.zeros((101, 101), np.uint8)
    mask[50, 50] = 255

    grown = cv2.dilate(mask, ellipse_kernel(6))

    # The row through the centre spans 2r+1 pixels.
    assert int((grown[50] > 0).sum()) == 13


def test_scaled_dilation_reaches_approximately_the_requested_radius():
    mask = np.zeros((480, 640), np.uint8)
    cv2.circle(mask, (320, 240), 20, 255, -1)

    grown = dilate_scaled(mask, radius_px=40, divisor=4)

    distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    reached = distance[grown > 0].max()
    assert 30 <= reached <= 50, f"asked for 40 px, reached {reached:.0f}"


def test_scaled_dilation_is_a_no_op_for_a_zero_radius():
    mask = np.zeros((64, 64), np.uint8)
    mask[30:34, 30:34] = 255
    assert np.array_equal(dilate_scaled(mask, 0), mask)


def test_channel_max_matches_the_obvious_spelling():
    """Same answer as image.max(axis=2), which measured three times slower."""
    image = textured_frame(320, 240, seed=11)
    assert np.array_equal(channel_max(image), image.max(axis=2))


def test_channel_max_passes_single_channel_images_through():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert np.array_equal(channel_max(gray), gray)


def test_channel_max_is_sensitive_to_a_single_channel_change():
    """A colour shift must not be averaged away by the other two channels."""
    image = np.zeros((4, 4, 3), np.uint8)
    image[:, :, 2] = 90  # red only
    assert channel_max(image).max() == 90
    assert cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).max() < 60
