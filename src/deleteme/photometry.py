"""Fitting the stored plate to the light arriving now.

The plate is a photograph taken at one moment. Every frame after that is lit
slightly differently — the sun goes behind a cloud, a monitor changes what it is
showing, the driver nudges its gain. Pasting the plate unmodified puts a patch
of *then* into a picture of *now*, and the seam shows.

The correction is a per-channel affine map, ``out = gain * plate + bias``, fitted
in closed form against the pixels that are known to still be background. Fitted
on the development machine against a synthetic ground truth of gain 1.2 and bias
-10, it recovers (1.200, -10.02) and takes the median residual from 8.0 to 0.00.

Three details are load-bearing:

* **Apply through a lookup table, not** ``cv2.convertScaleAbs``. That function
  takes the absolute value of its result, so ``convertScaleAbs(10, alpha=1,
  beta=-30)`` returns 20 rather than 0 — every darkening correction comes out
  inverted. A clipped LUT is also faster.
* **Smooth the parameters, not the images.** Averaging corrected plates smears
  detail; averaging two gains and two biases costs nothing and cannot blur.
* **Refuse to fit when the fit is meaningless.** On a flat wall the gain term is
  unidentifiable: a true gain of 1.1 comes back as 1.0002 with a bias of +11.97,
  which reproduces the pixels perfectly and would be catastrophic to extrapolate.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from deleteme.config import PhotometryConfig

log = logging.getLogger(__name__)

_RAMP = np.arange(256, dtype=np.float32)


@dataclass(frozen=True)
class GainBias:
    """A per-channel affine photometric correction, in BGR order."""

    gain: np.ndarray
    bias: np.ndarray
    pixels: int
    degenerate: bool = False
    """True when the plate was too flat to identify gain, so it was pinned to 1."""

    @classmethod
    def identity(cls) -> GainBias:
        return cls(gain=np.ones(3, np.float32), bias=np.zeros(3, np.float32), pixels=0)

    @property
    def is_identity(self) -> bool:
        return bool(
            np.allclose(self.gain, 1.0, atol=1e-4) and np.allclose(self.bias, 0.0, atol=1e-3)
        )


def fit_gain_bias(
    plate: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray | None,
    config: PhotometryConfig | None = None,
) -> GainBias | None:
    """Least-squares fit of ``frame ~= gain * plate + bias`` per channel.

    ``mask`` selects the pixels believed to still show background; passing
    ``None`` fits over everything, which is only correct when nothing is
    occluding the scene. Returns ``None`` when there is not enough evidence,
    which callers should treat as "keep the previous parameters".
    """
    cfg = config or PhotometryConfig()
    if plate.shape != frame.shape:
        raise ValueError(f"plate {plate.shape} and frame {frame.shape} differ")

    stride = max(1, cfg.subsample_stride)
    plate_s = plate[::stride, ::stride]
    frame_s = frame[::stride, ::stride]

    if mask is None:
        selected_plate = plate_s.reshape(-1, 3)
        selected_frame = frame_s.reshape(-1, 3)
    else:
        sel = mask[::stride, ::stride].astype(bool, copy=False)
        if sel.ndim == 3:
            sel = sel[:, :, 0]
        selected_plate = plate_s[sel]
        selected_frame = frame_s[sel]

    count = int(selected_plate.shape[0])
    if count < cfg.min_fit_pixels:
        return None

    plate_f = selected_plate.astype(np.float32, copy=False)
    frame_f = selected_frame.astype(np.float32, copy=False)

    gain = np.ones(3, np.float32)
    bias = np.zeros(3, np.float32)
    degenerate = False

    for channel in range(3):
        p = plate_f[:, channel]
        f = frame_f[:, channel]
        g, b, chan_degenerate = _fit_channel(p, f, cfg)

        # One reject-and-refit pass. A handful of pixels that changed for real
        # — a shadow edge, a reflection — would otherwise drag the whole fit.
        residual = f - (g * p + b)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        if mad > 1e-6:
            keep = np.abs(residual - median) <= cfg.outlier_sigma * 1.4826 * mad
            if int(keep.sum()) >= cfg.min_fit_pixels:
                g, b, chan_degenerate = _fit_channel(p[keep], f[keep], cfg)

        gain[channel] = np.clip(g, cfg.gain_min, cfg.gain_max)
        bias[channel] = np.clip(b, cfg.bias_min, cfg.bias_max)
        degenerate = degenerate or chan_degenerate

    return GainBias(gain=gain, bias=bias, pixels=count, degenerate=degenerate)


def _fit_channel(p: np.ndarray, f: np.ndarray, cfg: PhotometryConfig) -> tuple[float, float, bool]:
    """Closed-form affine fit for one channel."""
    p_mean = float(p.mean())
    f_mean = float(f.mean())
    p_centred = p - p_mean
    denominator = float(np.dot(p_centred, p_centred))
    variance = denominator / max(1, p.size)

    if variance < cfg.min_plate_variance:
        # Flat region: gain is unidentifiable. Fit the offset only.
        return 1.0, f_mean - p_mean, True

    gain = float(np.dot(p_centred, f - f_mean) / denominator)
    return gain, f_mean - gain * p_mean, False


def build_lut(gain: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """A ``(1, 256, 3)`` uint8 table applying ``gain * x + bias`` per channel.

    Clipping happens in float before the cast. A naive ``(200 * 2).astype(uint8)``
    wraps to 144, which produces bright pixels that suddenly turn dark.

    Rounded, not truncated. ``astype(uint8)`` floors, which throws away up to a
    full level on every entry and discards a sub-1.0 bias correction outright:
    ``build_lut(gain=1, bias=0.9)`` came out bit-identical to the identity
    table, even though a bias move of only 0.3 is enough to trigger a rebuild.
    That left a permanent ~1 level floor under the health residual, and made
    this module's own documented "median residual 8.0 to 0.00" untrue of its
    implementation.
    """
    table = np.empty((1, 256, 3), dtype=np.uint8)
    for channel in range(3):
        values = _RAMP * float(gain[channel]) + float(bias[channel])
        table[0, :, channel] = np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)
    return table


def apply_lut(image: np.ndarray, lut: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
    """Apply a table built by :func:`build_lut`."""
    return cv2.LUT(image, lut, dst=dst)


class PhotometryTracker:
    """Tracks the plate-to-frame correction over time.

    Holds the exponentially smoothed parameters, decides when the lookup table
    is stale enough to be worth rebuilding, and reports how fast the light is
    currently moving so the plate refresh can stand down during a transition.
    """

    def __init__(self, config: PhotometryConfig | None = None) -> None:
        self.config = config or PhotometryConfig()
        self.gain = np.ones(3, np.float32)
        self.bias = np.zeros(3, np.float32)

        self._lut = build_lut(self.gain, self.bias)
        self._lut_gain = self.gain.copy()
        self._lut_bias = self.bias.copy()

        self.last_fit: GainBias | None = None
        self.frozen = False
        """True when the last frame produced no usable fit and the previous
        parameters were held. Not an error — a person filling the frame leaves
        nothing to fit against."""
        self.out_of_range = False
        """True when the fit wanted a correction beyond the sane limits, which
        means the *scene* changed rather than the light."""
        self.used_fallback = False
        """True when the preferred mask starved the fit and the fallback was
        used instead. See :meth:`update`."""

        self._history: deque[tuple[float, float]] = deque(maxlen=64)

    @property
    def lut(self) -> np.ndarray:
        return self._lut

    def update(
        self,
        plate: np.ndarray,
        frame: np.ndarray,
        mask: np.ndarray | None,
        now: float,
        fallback_mask: np.ndarray | None = None,
    ) -> GainBias | None:
        """Fit this frame and fold the result into the smoothed parameters.

        ``mask`` is the preferred selection. ``fallback_mask`` is a wider one
        tried only if the preferred mask leaves too little to fit against.

        That fallback resolves a deadlock rather than being belt-and-braces.
        The caller's preferred mask excludes pixels that differ from the
        *corrected* plate, so before any correction has been learned a genuine
        lighting change marks most of the frame as changed: a 25% brightening
        puts 57.5% of pixels over the threshold, leaving too few to fit, so the
        model would refuse to learn precisely the change it exists to handle —
        and the bigger the change, the more certainly it would refuse.

        Deciding this by trying the fit, rather than by a separate coverage
        threshold, means there is no second number to keep in agreement with
        ``min_fit_pixels``. An earlier version had exactly that bug: it relaxed
        below 12% coverage while the fit needed 32.6%.
        """
        cfg = self.config
        fit = fit_gain_bias(plate, frame, mask, cfg)
        self.used_fallback = False
        if fit is None and fallback_mask is not None:
            fit = fit_gain_bias(plate, frame, fallback_mask, cfg)
            self.used_fallback = fit is not None

        if fit is None:
            self.frozen = True
            # Cleared too: a refused fit is not evidence that the scene changed.
            # Leaving it latched meant one clipped fit followed by a subject
            # close enough to starve the fit kept reporting "the scene has
            # changed, not just the light" — which is checked before every other
            # explanation, so it masked whatever was actually wrong.
            self.out_of_range = False
            self._history.append((now, float(self.gain.mean())))
            return None

        self.frozen = False
        self.last_fit = fit

        # A correction outside these limits is not a lighting change — it is the
        # scene itself having moved. Flag it and leave the parameters alone;
        # applying it would smear a wrong plate over the frame more confidently.
        raw_gain_extreme = bool(
            np.any(fit.gain <= cfg.gain_min + 1e-6) or np.any(fit.gain >= cfg.gain_max - 1e-6)
        )
        raw_bias_extreme = bool(
            np.any(fit.bias <= cfg.bias_min + 1e-6) or np.any(fit.bias >= cfg.bias_max - 1e-6)
        )
        self.out_of_range = raw_gain_extreme or raw_bias_extreme
        if self.out_of_range:
            self._history.append((now, float(self.gain.mean())))
            return fit

        alpha = cfg.ema_alpha
        self.gain += alpha * (fit.gain - self.gain)
        self.bias += alpha * (fit.bias - self.bias)
        self._history.append((now, float(self.gain.mean())))

        if (
            np.max(np.abs(self.gain - self._lut_gain)) > cfg.gain_deadband
            or np.max(np.abs(self.bias - self._lut_bias)) > cfg.bias_deadband
        ):
            self._lut = build_lut(self.gain, self.bias)
            self._lut_gain = self.gain.copy()
            self._lut_bias = self.bias.copy()

        return fit

    def correct(self, plate: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
        """Return the plate mapped into the current frame's light."""
        return apply_lut(plate, self._lut, dst=dst)

    def gain_movement(self, now: float, window_s: float = 0.5) -> float:
        """Peak-to-peak movement of mean gain over the recent past.

        Used to stand the plate refresh down mid-transition, so a half-adjusted
        exposure never gets averaged into the stored background.
        """
        recent = [g for t, g in self._history if now - t <= window_s]
        if len(recent) < 2:
            return 0.0
        return max(recent) - min(recent)

    def reset(self) -> None:
        """Return to identity. Called when a fresh plate is adopted."""
        self.gain = np.ones(3, np.float32)
        self.bias = np.zeros(3, np.float32)
        self._lut = build_lut(self.gain, self.bias)
        self._lut_gain = self.gain.copy()
        self._lut_bias = self.bias.copy()
        self.last_fit = None
        self.frozen = False
        self.out_of_range = False
        self.used_fallback = False
        self._history.clear()
