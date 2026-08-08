"""Person segmentation and mask stabilisation.

The old pipeline detected a bounding box and pasted a rectangle of background
over it — then stretched that rectangle to the bottom of the frame, because the
box cut off legs. So it also deleted the desk.

A segmentation model gives a per-pixel confidence instead, and happens to be
much cheaper than the detector it replaces: ``selfie_segmenter`` measured 9.9 ms
against ``efficientdet_lite0``'s 37.1 ms, at 244 KB against 13.8 MB. It also
removes two bugs for nothing — a second person in shot is now handled, and there
is no leg problem to work around.

Raw model output is not usable directly. The network runs at 256x256 and its
result is bilinearly upsampled about 2.5x, so a single pixel of network jitter
becomes two or three pixels of crawling edge. Everything in
:class:`MaskStabilizer` after the threshold exists to stop that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from deleteme.config import SegmentationConfig
from deleteme.errors import ModelError
from deleteme.morphology import ellipse_kernel
from deleteme.paths import SEGMENTER_MODEL, model_path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonMask:
    """A stabilised silhouette, in the several forms the pipeline needs."""

    alpha: np.ndarray
    """Feathered coverage in [0, 1], float32. What the compositor blends with."""

    binary: np.ndarray
    """Hard, dilated silhouette as uint8 0/255. Used for exclusion zones and as
    the seed for the change mask's connectivity test."""

    area_fraction: float
    """Silhouette area as a fraction of the frame, before dilation. Drives the
    distance-dependent behaviour: a small value means the subject is far away,
    where the segmenter is least reliable."""

    @property
    def present(self) -> bool:
        return self.area_fraction > 0.0


class PersonSegmenter:
    """MediaPipe selfie segmentation in VIDEO mode.

    VIDEO rather than IMAGE because the task then carries state between frames.
    The old code asked for IMAGE mode while passing ``min_tracking_confidence``,
    which is inert outside VIDEO and LIVE_STREAM — so every frame was solved from
    scratch with no temporal continuity at all.

    Timestamps must strictly increase or the task raises, so they come from a
    counter owned by this object that never resets for its lifetime. Deriving
    them from the wall clock is unsafe: two frames can land in the same
    millisecond, and an equal timestamp is rejected just as a decreasing one is.
    """

    def __init__(
        self,
        model_file: Path | None = None,
        config: SegmentationConfig | None = None,
        fps: int = 30,
    ) -> None:
        self.config = config or SegmentationConfig()
        path = model_file or model_path(SEGMENTER_MODEL)
        try:
            self._segmenter = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                    running_mode=vision.RunningMode.VIDEO,
                    output_confidence_masks=True,
                    output_category_mask=False,
                )
            )
        except Exception as exc:  # pragma: no cover - native failure
            raise ModelError(f"could not load the segmentation model from {path}: {exc}") from exc

        self._step_ms = max(1, round(1000 / max(1, fps)))
        self._timestamp_ms = 0
        self._closed = False

    def confidence(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel person confidence in [0, 1], already at frame resolution."""
        if self._closed:
            raise RuntimeError("segmenter has been closed")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += self._step_ms
        result = self._segmenter.segment_for_video(image, self._timestamp_ms)
        if not result.confidence_masks:
            return np.zeros(frame_bgr.shape[:2], np.float32)
        mask = result.confidence_masks[0].numpy_view()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return np.ascontiguousarray(mask, dtype=np.float32)

    def close(self) -> None:
        """Release the native task.

        Must not be called while a call is in flight — that produced an access
        violation in 3 of 25 trials. Join any worker thread first.
        """
        if not self._closed:
            self._closed = True
            self._segmenter.close()

    def __enter__(self) -> PersonSegmenter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class MaskStabilizer:
    """Turns a noisy per-frame confidence map into a stable silhouette."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()
        self._ema: np.ndarray | None = None
        self._close_kernel = ellipse_kernel(self.config.close_radius_px)

    def reset(self) -> None:
        """Forget the previous mask.

        Called when the effect switches off, so no ghost of the last silhouette
        carries into the next activation.
        """
        self._ema = None

    def update(self, confidence: np.ndarray) -> PersonMask:
        cfg = self.config
        height, width = confidence.shape[:2]

        # Smooth the float confidence *before* thresholding. Smoothing after
        # would only blur an already-jittering binary edge.
        if self._ema is None or self._ema.shape != confidence.shape:
            self._ema = confidence.astype(np.float32).copy()
        else:
            cv2.addWeighted(
                confidence,
                cfg.mask_ema_alpha,
                self._ema,
                1.0 - cfg.mask_ema_alpha,
                0.0,
                dst=self._ema,
            )
        smoothed = self._ema

        # Hysteresis: a high-confidence core anchors the silhouette, and lower
        # confidence pixels join only if they connect to a core. Isolated
        # medium-confidence blobs elsewhere in the room are ignored.
        core = (smoothed > cfg.core_threshold).astype(np.uint8)
        shell = (smoothed > cfg.shell_threshold).astype(np.uint8)

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(shell, connectivity=8)
        keep = np.unique(labels[core > 0])
        keep = keep[keep != 0]

        # An area rule, not "keep the largest". Largest-only silently drops a
        # second person, which is the bug the segmentation swap just fixed.
        min_area = cfg.min_component_frame_fraction * height * width
        table = np.zeros(max(count, 1), np.uint8)
        silhouette_area = 0
        for label in keep:
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                table[label] = 255
                silhouette_area += area

        if silhouette_area == 0:
            empty = np.zeros((height, width), np.uint8)
            return PersonMask(
                alpha=np.zeros((height, width), np.float32), binary=empty, area_fraction=0.0
            )

        silhouette = table[labels]
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, self._close_kernel)

        area_fraction = silhouette_area / float(height * width)
        dilated = cv2.dilate(silhouette, ellipse_kernel(self.dilation_radius(silhouette_area)))

        feather = cfg.feather_kernel_px | 1
        alpha_u8 = cv2.GaussianBlur(dilated, (feather, feather), 0)
        alpha = alpha_u8.astype(np.float32)
        alpha *= 1.0 / 255.0

        return PersonMask(alpha=alpha, binary=dilated, area_fraction=area_fraction)

    def dilation_radius(self, area_px: float) -> int:
        """Dilation scaled to how big the subject is in frame.

        A fixed radius is wrong at both ends: it swallows a distant figure and
        barely touches a close-up. Scaling with the square root of area keeps it
        a roughly constant fraction of the subject.

        Erring large is close to free here, because the pixels being pasted are
        the real background — pushing the seam outward lands it where plate and
        frame show the same thing. Erring small leaves a rim of the subject's
        own colour, which is the halo everyone recognises.
        """
        cfg = self.config
        radius = cfg.dilate_area_coefficient * float(np.sqrt(max(0.0, area_px)))
        return round(min(max(radius, cfg.dilate_min_px), cfg.dilate_max_px))
