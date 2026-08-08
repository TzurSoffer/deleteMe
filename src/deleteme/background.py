"""The background and lighting model.

This is the part the rest of the project exists to serve. It owns the plate and
everything about whether the plate is still telling the truth.

The single most important property of this class is that :meth:`update` runs on
*every* frame, whether or not the effect is switched on. The idle path — nobody
gesturing, the room simply sitting there — is where the model learns best,
because nothing is occluding the scene. The previous version wrote its plate
once and never looked at it again.

Two different "background" masks come out of each update, and conflating them
would break one job or the other:

``fit`` mask
    Pixels believed to show *the same background as the plate*. Anything that
    differs is excluded, because a moved chair would drag the photometric fit.

``refresh`` mask
    Pixels that simply are not the person. Changed pixels stay *in*, because
    learning that the chair moved is the entire point of refreshing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from math import exp, hypot

import cv2
import numpy as np

from deleteme.config import AppConfig
from deleteme.debounce import Debouncer
from deleteme.morphology import channel_max, dilate_scaled, ellipse_kernel
from deleteme.photometry import PhotometryTracker
from deleteme.plate import Plate

log = logging.getLogger(__name__)


class HealthState(StrEnum):
    """How much the plate can currently be trusted."""

    GOOD = "good"
    WARN = "warn"
    BAD = "bad"
    UNKNOWN = "unknown"
    """Not "bad" — "the measurement itself is not trustworthy right now".

    Shown as ``BG --``. Inventing a number when the evidence is missing is worse
    than admitting there is none.
    """


@dataclass(frozen=True)
class Health:
    """A reading on the plate's fidelity, plus the numbers behind it."""

    state: HealthState
    score: float | None
    reason: str

    residual: float = 0.0
    chroma_shift: float = 0.0
    shift_px: float = 0.0
    changed_fraction: float = 0.0
    stale_fraction: float = 0.0
    fit_pixels: int = 0
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_moved: bool = False
    frozen: bool = False

    @property
    def label(self) -> str:
        """The single line shown on the HUD."""
        if self.score is None:
            return "BG --"
        return f"BG {self.score:3.0f}"

    @classmethod
    def unknown(cls, reason: str) -> Health:
        return cls(state=HealthState.UNKNOWN, score=None, reason=reason)


class BumpDetector:
    """Notices that the camera itself has moved.

    Phase correlation on a downscaled pair. The scale matters more than it
    looks: at a divisor of 4 (160x120) a true 3 px shift was measured back as
    0.69 px on one axis, a 65% error. At a divisor of 2 it reads 3.24, -1.94.

    Deliberately reports rather than corrects. Warping the plate back into
    alignment means inventing border pixels that were never observed, and a
    bump large enough to matter usually changes the lighting geometry too.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = (config or AppConfig()).background
        self._window: np.ndarray | None = None
        self._hits = 0
        self.shift_px = 0.0
        self.alarm = False
        self.measurable = False

    def reset(self) -> None:
        self._hits = 0
        self.shift_px = 0.0
        self.alarm = False
        self.measurable = False

    def update(self, plate_gray: np.ndarray, frame_gray: np.ndarray) -> float:
        """Compare a plate and frame, both already downscaled and float32."""
        cfg = self.config
        if self._window is None or self._window.shape != plate_gray.shape:
            self._window = cv2.createHanningWindow(
                (plate_gray.shape[1], plate_gray.shape[0]), cv2.CV_32F
            )

        # Phase correlation needs structure to correlate. On a blank wall the
        # answer is meaningless, so gate on how much detail the plate actually
        # has rather than on the correlation response, which does not
        # generalise between scenes.
        energy = float(cv2.Laplacian(plate_gray, cv2.CV_32F).var())
        if energy < cfg.drift_min_gradient_energy:
            self.measurable = False
            self._hits = 0
            self.alarm = False
            return self.shift_px

        self.measurable = True
        (dx, dy), _response = cv2.phaseCorrelate(plate_gray, frame_gray, self._window)
        self.shift_px = hypot(dx, dy) * cfg.drift_scale_divisor

        if self.shift_px > cfg.drift_trigger_px:
            self._hits += 1
        else:
            self._hits = max(0, self._hits - 2)
        self.alarm = self._hits >= cfg.drift_sustain_samples
        return self.shift_px


class BackgroundModel:
    """Owns the plate, keeps it honest, and reports on its condition."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.photometry = PhotometryTracker(self.config.photometry)
        self.bump = BumpDetector(self.config)

        self._plate: Plate | None = None
        self._plate_u8: np.ndarray | None = None
        self._candidate: np.ndarray | None = None
        self._committed: np.ndarray | None = None
        self._previous_small: np.ndarray | None = None
        self._stable_ticks: np.ndarray | None = None
        self._age_s: np.ndarray | None = None

        self._corrected: np.ndarray | None = None
        self._corrected_from: np.ndarray | None = None

        self._frames = 0
        self._last_update: float | None = None
        self._last_drift_check = 0.0
        self._refresh_frozen_until = 0.0
        self._adopted_at = 0.0

        hold = self.config.health.flag_hold_s
        self._warn_flag = Debouncer(hold, hold)
        self._bad_flag = Debouncer(hold, hold)

        self.empty_scene_since: float | None = None
        self.last_health = Health.unknown("no background captured yet")
        self.fit_relaxed = False
        """Set when the last update fitted without the changed-pixel exclusion,
        because keeping it would have left too little to fit against."""
        self.change_mask_suppressed = False
        """Set when the last :meth:`change_mask` call overreached and was
        discarded. Surfaced on the HUD, because silently doing less than asked
        is how a feature gets a reputation for being unreliable."""

    # ------------------------------------------------------------------ plate

    @property
    def has_plate(self) -> bool:
        return self._plate is not None

    @property
    def plate(self) -> Plate | None:
        return self._plate

    @property
    def plate_image(self) -> np.ndarray | None:
        """The current committed plate, uncorrected."""
        return self._plate_u8

    def adopt(self, plate: Plate, now: float = 0.0) -> None:
        """Take a new plate and discard everything learned about the old one."""
        self._plate = plate
        self._plate_u8 = plate.image.copy()
        committed = plate.image.astype(np.float32)
        self._committed = committed
        self._candidate = committed.copy()
        divisor = max(1, self.config.background.coarse_divisor)
        coarse = (plate.image.shape[0] // divisor, plate.image.shape[1] // divisor)
        self._stable_ticks = np.zeros(coarse, np.uint16)
        self._previous_small = cv2.resize(
            plate.image, (coarse[1], coarse[0]), interpolation=cv2.INTER_AREA
        )
        self._age_s = np.zeros(coarse, np.float32)

        self.photometry.reset()
        self.bump.reset()
        self._corrected = None
        self._corrected_from = None
        self._frames = 0
        self._last_update = None
        self._refresh_frozen_until = 0.0
        self._adopted_at = now
        self._warn_flag.force(False)
        self._bad_flag.force(False)
        self.last_health = Health.unknown("measuring")
        log.info("adopted a new %dx%d plate", plate.image.shape[1], plate.image.shape[0])

    def corrected_plate(self) -> np.ndarray:
        """The plate mapped into the current frame's lighting.

        Cached against the lookup table object, which the tracker only replaces
        when the smoothed parameters have moved past their deadband — roughly
        five times a second rather than thirty.
        """
        if self._plate_u8 is None:
            raise RuntimeError("no plate has been adopted")
        lut = self.photometry.lut
        corrected = self._corrected
        if corrected is None or self._corrected_from is not lut:
            corrected = np.asarray(self.photometry.correct(self._plate_u8))
            self._corrected = corrected
            self._corrected_from = lut
        return corrected

    # ----------------------------------------------------------------- update

    def update(self, frame: np.ndarray, person_mask: np.ndarray | None, now: float) -> Health:
        """Fold one frame into the model. Call this every frame, always."""
        if self._plate is None or self._plate_u8 is None:
            self.last_health = Health.unknown("no background captured yet")
            return self.last_health

        self._plate.require_compatible(frame)

        dt = 0.0 if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now
        self._frames += 1

        cfg = self.config.background
        divisor = max(1, cfg.coarse_divisor)
        height, width = frame.shape[:2]
        small_size = (width // divisor, height // divisor)

        frame_small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
        corrected_small = cv2.resize(
            self.corrected_plate(), small_size, interpolation=cv2.INTER_AREA
        )

        difference = channel_max(cv2.absdiff(frame_small, corrected_small))
        changed_small = difference > cfg.change_exclusion_threshold

        if person_mask is None:
            person_small = np.zeros(changed_small.shape, np.uint8)
        else:
            person_small = cv2.resize(person_mask, small_size, interpolation=cv2.INTER_NEAREST)
            person_small = cv2.dilate(
                person_small,
                ellipse_kernel(max(1, round(cfg.person_exclusion_dilate_px / divisor))),
            )
        person_free_small = person_small == 0

        # Measured over the scene, not over the subject. The person differs from
        # the plate by design — that is the whole point of them — so counting
        # their silhouette here made the "changed" reading a measure of how
        # large you are in frame rather than of how stale the plate is. It feeds
        # the health score and the "plate no longer matches" verdict, so a
        # close-up subject in front of a perfect plate was reported as broken.
        visible = int(np.count_nonzero(person_free_small))
        changed_fraction = (
            float(np.count_nonzero(changed_small & person_free_small)) / visible if visible else 0.0
        )

        erode_kernel = ellipse_kernel(max(1, round(cfg.background_erode_px / divisor)))
        refresh_small = cv2.erode(person_free_small.astype(np.uint8), erode_kernel)
        strict_small = cv2.erode(
            (person_free_small & ~changed_small).astype(np.uint8), erode_kernel
        )

        strict_mask = cv2.resize(strict_small, (width, height), interpolation=cv2.INTER_NEAREST)
        refresh_mask = cv2.resize(refresh_small, (width, height), interpolation=cv2.INTER_NEAREST)

        # No explicit cache invalidation here. `corrected_plate` keys its cache
        # on the lookup table object itself, and the tracker only builds a new
        # one when the smoothed parameters leave their deadband — a few times a
        # second rather than thirty. Invalidating on every successful fit would
        # throw that away and re-map the whole plate every frame.
        # Prefer the strict mask, which keeps a moved chair from dragging the
        # fit, but let the tracker widen to everything-but-the-person when the
        # strict one leaves too little to fit against.
        self.photometry.update(self._plate_u8, frame, strict_mask, now, fallback_mask=refresh_mask)
        self.fit_relaxed = self.photometry.used_fallback
        fit_small = refresh_small if self.fit_relaxed else strict_small

        self._refresh(frame, refresh_mask, now)
        self._update_drift(frame, person_small, now)
        stale_fraction = self._update_staleness(refresh_small, dt)
        self._track_empty_scene(person_mask, now)

        self.last_health = self._assess(
            difference_small=difference,
            frame_small=frame_small,
            corrected_small=corrected_small,
            fit_small=fit_small.astype(bool),
            changed_fraction=changed_fraction,
            stale_fraction=stale_fraction,
        )
        return self.last_health

    # ---------------------------------------------------------------- refresh

    def _refresh(self, frame: np.ndarray, refresh_mask: np.ndarray, now: float) -> None:
        """Slowly relearn the background where the scene is genuinely visible.

        Two speeds. A fast *candidate* tracks what the room looks like now; a
        slow *committed* plate only adopts a pixel once the candidate has stopped
        arguing about it. Someone who holds still for a second cannot burn into
        the background, because their pixels never stabilise long enough — and
        they are excluded by the person mask anyway.
        """
        cfg = self.config.background
        if self._frames % max(1, cfg.refresh_every_n_frames) != 0:
            return
        candidate, committed = self._candidate, self._committed
        previous_small, ticks = self._previous_small, self._stable_ticks
        if candidate is None or committed is None or previous_small is None or ticks is None:
            return

        # Never learn during a lighting transition: averaging a half-adjusted
        # exposure into the plate bakes in a value that was never correct.
        if self.photometry.gain_movement(now) > cfg.refresh_freeze_after_gain_move:
            self._refresh_frozen_until = now + cfg.refresh_freeze_s
        if now < self._refresh_frozen_until or self.photometry.frozen:
            return

        cv2.accumulateWeighted(frame, candidate, cfg.candidate_alpha, mask=refresh_mask)

        ticks_needed = max(
            1,
            int(
                cfg.commit_stable_after_s
                * self.config.camera.fps
                / max(1, cfg.refresh_every_n_frames)
            ),
        )

        # Stability is judged on a quarter-scale uint8 copy. The question is
        # whether a *region* has stopped arguing about its value, which does not
        # need per-pixel precision — and full-resolution float32 differencing
        # measured roughly 3.7x the cost of this for the same answer.
        coarse = (ticks.shape[1], ticks.shape[0])
        candidate_u8 = cv2.convertScaleAbs(candidate)
        candidate_small = cv2.resize(candidate_u8, coarse, interpolation=cv2.INTER_AREA)
        movement = channel_max(cv2.absdiff(candidate_small, previous_small))
        np.copyto(previous_small, candidate_small)

        # (ticks + 1) * settled, vectorised. Boolean fancy indexing allocates
        # two index arrays per call; this does not. The clamp keeps a pixel that
        # has been quiet for hours from wrapping its counter back to zero and
        # briefly un-committing itself.
        settled = (movement < 2).astype(np.uint16)
        ticks += 1
        ticks *= settled
        np.minimum(ticks, ticks_needed, out=ticks)

        ready_small = (ticks >= ticks_needed).astype(np.uint8)
        if np.any(ready_small):
            ready = cv2.resize(
                ready_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            cv2.accumulateWeighted(candidate, committed, cfg.committed_alpha, mask=ready)
            # Safe here, unlike in photometry: an average of uint8 pixels is
            # non-negative, so convertScaleAbs's absolute value never fires.
            self._plate_u8 = cv2.convertScaleAbs(committed)
            self._corrected_from = None

    # ------------------------------------------------------------------ drift

    def _update_drift(self, frame: np.ndarray, person_small: np.ndarray, now: float) -> None:
        cfg = self.config.background
        if now - self._last_drift_check < 1.0 / max(0.1, cfg.drift_check_hz):
            return
        self._last_drift_check = now

        # Phase correlation cannot be masked, so a large moving subject would
        # dominate the estimate. Only measure when the frame is mostly scene.
        if person_small.size and np.count_nonzero(person_small) / person_small.size > 0.25:
            return
        if self._plate_u8 is None:
            return

        divisor = max(1, cfg.drift_scale_divisor)
        size = (frame.shape[1] // divisor, frame.shape[0] // divisor)
        plate_gray = cv2.resize(
            cv2.cvtColor(self._plate_u8, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        frame_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        self.bump.update(plate_gray, frame_gray)

    # -------------------------------------------------------------- staleness

    def _update_staleness(self, refresh_small: np.ndarray, dt: float) -> float:
        """How much of the scene has not been seen for a long time."""
        if self._age_s is None:
            return 0.0
        visible = refresh_small.astype(bool)
        if visible.shape != self._age_s.shape:
            visible = cv2.resize(
                refresh_small,
                (self._age_s.shape[1], self._age_s.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        self._age_s[visible] = 0.0
        self._age_s[~visible] += dt
        return (
            float(np.count_nonzero(self._age_s > self.config.background.stale_after_s))
            / self._age_s.size
        )

    def _track_empty_scene(self, person_mask: np.ndarray | None, now: float) -> None:
        """Remember how long the room has been continuously empty."""
        empty = person_mask is None or not np.any(person_mask)
        if empty:
            if self.empty_scene_since is None:
                self.empty_scene_since = now
        else:
            self.empty_scene_since = None

    def empty_for(self, now: float) -> float:
        if self.empty_scene_since is None:
            return 0.0
        return max(0.0, now - self.empty_scene_since)

    # ----------------------------------------------------------------- health

    def _assess(
        self,
        difference_small: np.ndarray,
        frame_small: np.ndarray,
        corrected_small: np.ndarray,
        fit_small: np.ndarray,
        changed_fraction: float,
        stale_fraction: float,
    ) -> Health:
        cfg = self.config.health
        fit_pixels = int(np.count_nonzero(fit_small))

        gain = tuple(float(g) for g in self.photometry.gain)
        bias = tuple(float(b) for b in self.photometry.bias)

        if fit_pixels < 64:
            # Two very different situations produce the same symptom, and
            # conflating them would either cry wolf or stay silent when it
            # matters. Almost nothing matching the plate *and* almost everything
            # in the frame having changed means the plate describes a room that
            # is no longer there. A person simply filling the frame leaves the
            # measurement impossible but the plate perfectly good.
            if changed_fraction > 0.8:
                return Health(
                    state=HealthState.BAD,
                    score=0.0,
                    reason="the background no longer matches — recapture",
                    changed_fraction=changed_fraction,
                    stale_fraction=stale_fraction,
                    fit_pixels=fit_pixels,
                    gain=gain,  # type: ignore[arg-type]
                    bias=bias,  # type: ignore[arg-type]
                )
            return Health.unknown("not enough visible background to judge")

        residual = float(np.median(difference_small[fit_small].astype(np.float32)))

        frame_lab = cv2.cvtColor(frame_small, cv2.COLOR_BGR2LAB)
        plate_lab = cv2.cvtColor(corrected_small, cv2.COLOR_BGR2LAB)
        chroma_shift = float(
            abs(frame_lab[:, :, 1][fit_small].mean() - plate_lab[:, :, 1][fit_small].mean())
            + abs(frame_lab[:, :, 2][fit_small].mean() - plate_lab[:, :, 2][fit_small].mean())
        )

        shift_px = self.bump.shift_px if self.bump.measurable else 0.0
        score = (
            100.0
            * exp(-((residual / cfg.score_residual_scale) ** 2))
            * exp(-((shift_px / cfg.score_shift_scale) ** 2))
            * max(0.0, 1.0 - changed_fraction)
        )

        warn = self._warn_flag.update(
            residual > cfg.residual_warn or chroma_shift > cfg.chroma_shift_warn,
            self._last_update or 0.0,
        )
        bad = self._bad_flag.update(
            residual > cfg.residual_bad or self.bump.alarm, self._last_update or 0.0
        )

        if bad:
            state = HealthState.BAD
        elif warn:
            state = HealthState.WARN
        else:
            state = HealthState.GOOD

        return Health(
            state=state,
            score=score,
            reason=self._reason(state, residual, chroma_shift, stale_fraction),
            residual=residual,
            chroma_shift=chroma_shift,
            shift_px=shift_px,
            changed_fraction=changed_fraction,
            stale_fraction=stale_fraction,
            fit_pixels=fit_pixels,
            gain=gain,  # type: ignore[arg-type]
            bias=bias,  # type: ignore[arg-type]
            camera_moved=self.bump.alarm,
            frozen=self.photometry.frozen,
        )

    def _reason(
        self, state: HealthState, residual: float, chroma_shift: float, stale_fraction: float
    ) -> str:
        if state is HealthState.GOOD:
            return ""
        if self.bump.alarm:
            return "the camera has moved — press R to recapture"
        if self.photometry.out_of_range:
            return "the scene has changed, not just the light — recapture"
        if chroma_shift > self.config.health.chroma_shift_warn:
            return "the colour of the light has shifted"
        if stale_fraction > 0.5:
            return "most of the background has not been seen for a while"
        if residual > self.config.health.residual_bad:
            return "the background no longer matches — recapture"
        return "the lighting has drifted"

    # ------------------------------------------------------------ change mask

    def change_mask(
        self, frame: np.ndarray, person_binary: np.ndarray, subject_fraction: float
    ) -> np.ndarray:
        """Pixels that differ from the plate *and* belong to the subject.

        This is what removes the cast shadow. It is also the most dangerous
        thing in the file, for one reason: an uncorrected plate makes every
        pixel differ after a single exposure step, and an ungated difference
        would replace the entire frame with wallpaper. Hence the spatial gate,
        which is not optional and never a later refinement.

        The connectivity rule is what keeps a mug you set down on the desk
        visible while your shadow disappears: the shadow touches you, the mug
        does not.
        """
        cfg = self.config.change_mask
        self.change_mask_suppressed = False
        if not cfg.enabled or self._plate_u8 is None:
            return person_binary

        noise_sigma = self._plate.metadata.noise_sigma if self._plate else 0.0
        if noise_sigma <= 0.0:
            noise_sigma = cfg.fallback_noise_sigma
        threshold = max(cfg.min_threshold, noise_sigma * cfg.noise_sigma_multiplier)

        height, width = frame.shape[:2]
        silhouette_area = max(1.0, subject_fraction * height * width)
        roi_radius = cfg.roi_area_coefficient * float(np.sqrt(silhouette_area))
        if subject_fraction < cfg.small_subject_frame_fraction:
            threshold *= cfg.small_subject_threshold_scale
            roi_radius *= cfg.small_subject_roi_scale
        roi_radius = min(max(roi_radius, cfg.roi_min_px), cfg.roi_max_px)

        difference = channel_max(cv2.absdiff(frame, self.corrected_plate()))
        change = difference > threshold
        change &= dilate_scaled(person_binary, roi_radius) > 0

        person_bool = person_binary > 0
        union = (change | person_bool).astype(np.uint8)
        if not cfg.connectivity_gated:
            return union * 255

        count, labels = cv2.connectedComponents(union, connectivity=8)
        keep = np.unique(labels[person_bool])
        keep = keep[keep != 0]
        if keep.size == 0:
            return person_binary
        table = np.zeros(count, np.uint8)
        table[keep] = 255
        result = table[labels]

        # The gate bounds where the removal may reach; this bounds how much of
        # the picture it may take. A shadow roughly doubles the silhouette. A
        # photometric fit that has gone wrong takes everything, and that is the
        # case worth failing safe on — fall back to the bare silhouette, which
        # is merely the old behaviour rather than a screen of wallpaper.
        removed = int(np.count_nonzero(result))
        person_area = int(np.count_nonzero(person_bool))
        overreach = removed - person_area > person_area * cfg.max_expansion_ratio
        if overreach or removed > cfg.max_frame_fraction * height * width:
            self.change_mask_suppressed = True
            log.debug(
                "change mask suppressed: %d px removed against a %d px silhouette",
                removed,
                person_area,
            )
            return person_binary
        return result
