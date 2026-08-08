"""Shared image-geometry helpers.

Small module, but it exists for a specific reason: structuring elements are
specified by *size* in OpenCV and reasoned about as *radius* by humans, and
conflating the two silently halves every dilation. That bug hid a cast shadow
27 px from its owner behind a gate that was documented as reaching 40 px and
actually reached 20.

Everything here takes a radius and says so.
"""

from __future__ import annotations

import cv2
import numpy as np


def ellipse_kernel(radius_px: int) -> np.ndarray:
    """An elliptical structuring element of the given *radius*.

    A radius of ``r`` produces a ``2r+1`` square kernel, so ``ellipse_kernel(3)``
    grows a shape by three pixels in every direction.
    """
    r = max(0, int(radius_px))
    size = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def dilate_scaled(mask: np.ndarray, radius_px: float, divisor: int = 4) -> np.ndarray:
    """Dilate by ``radius_px`` at reduced resolution.

    A 81x81 structuring element over a full frame is needlessly expensive when
    the result is an exclusion zone tens of pixels wide — a few pixels of
    quantisation makes no difference to it. Working at a quarter scale is about
    an order of magnitude cheaper for the same practical outcome.
    """
    if radius_px <= 0:
        return mask
    height, width = mask.shape[:2]
    small_size = (max(1, width // divisor), max(1, height // divisor))
    small = cv2.resize(mask, small_size, interpolation=cv2.INTER_NEAREST)
    small = cv2.dilate(small, ellipse_kernel(max(1, round(radius_px / divisor))))
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def channel_max(image: np.ndarray) -> np.ndarray:
    """Largest value across the colour channels, per pixel.

    ``image.max(axis=2)`` is the obvious spelling and measured 5.92 ms on a
    640x480 frame — reducing along the fastest-varying axis defeats numpy's
    vectorisation. Splitting and taking two pairwise maxima does the same work
    in 1.98 ms.

    Maximum rather than a luma conversion because a change confined to one
    channel — a colour shift rather than a brightness one — must not be averaged
    away by the other two.
    """
    if image.ndim == 2:
        return image
    blue, green, red = cv2.split(image)
    return cv2.max(cv2.max(blue, green), red)
