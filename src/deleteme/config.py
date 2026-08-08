"""Tunable parameters, gathered in one place.

Every threshold in this file was calibrated against one webcam in one room. The
ones that depend on sensor noise are *not* hardcoded — they are derived at
capture time from the frames the quality gate collected, and stored in the
plate sidecar. What lives here are the structural constants: kernel sizes, time
constants, and the limits that decide when a measurement is untrustworthy.

Nothing here is loaded from a file. A config file for a webcam toy is a
maintenance burden and one more thing to get out of sync; the handful of things
a user genuinely needs to change are command-line flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30

    warmup_frames: int = 30
    """Frames to discard before locking photometry.

    The driver's auto-exposure and auto-white-balance need time to converge on
    the empty scene. Locking before they settle freezes the wrong values.
    """

    read_timeout_s: float = 5.0
    stall_tolerance_frames: int = 45
    """Consecutive failed reads tolerated before the session is declared dead.

    The previous version aborted on the first failure, which killed the whole
    effect on any transient USB hiccup.
    """

    lock_photometry: bool = True
    exposure_probe_stops: float = 1.0
    """How far to bump exposure when empirically verifying that a lock took.

    Some drivers accept the property, report success, and ignore it. The only
    reliable test is to change it and watch whether the image responds.
    """
    exposure_probe_min_delta_luma: float = 0.8


@dataclass(frozen=True)
class PlateConfig:
    required_good_frames: int = 10
    timeout_s: float = 20.0

    min_mean_luma: float = 60.0
    max_mean_luma: float = 200.0
    max_p99_luma: float = 250.0
    max_highlight_clip_fraction: float = 0.001
    min_focus_variance: float = 40.0

    stability_window: int = 8
    max_luma_spread: float = 1.5
    max_frame_delta: float = 2.0

    idle_recapture_after_s: float = 3.0
    """Scene must be continuously empty this long before an automatic refresh."""


@dataclass(frozen=True)
class PhotometryConfig:
    subsample_stride: int = 5
    """Spatial stride for the fit.

    Measured against a real room photograph, accuracy is flat across strides —
    the worst-case gain error is 0.0017 whether the fit sees 48,000 pixels or
    3,000 — because a real scene has texture everywhere. Cost, however, is
    linear: 6.21 ms at stride 2 against 1.15 ms at stride 5. So take the cheap
    one. (This does not hold for a blank wall, which is what
    ``min_plate_variance`` below exists to catch.)
    """

    min_fit_pixels: int = 1000
    """Below this the fit is refused and the previous parameters are held.

    Counted *after* subsampling. A thousand samples still over-determines two
    parameters by three orders of magnitude; the threshold guards against a
    person filling the frame, not against statistical noise. At stride 5 this
    trips when visible background falls below roughly 8% of the frame.
    """

    min_plate_variance: float = 4.0
    """Below this the plate is effectively flat and gain is unidentifiable.

    A flat plate fits *predictively* well while being *parametrically* nonsense:
    a true gain of 1.1 comes back as gain 1.0002 with a bias of +11.97. The
    pixels are right and the parameters are meaningless, so gain is pinned to 1
    and only bias is fitted.
    """

    gain_min: float = 0.6
    gain_max: float = 1.6
    bias_min: float = -40.0
    bias_max: float = 40.0

    ema_alpha: float = 0.08
    gain_deadband: float = 0.005
    bias_deadband: float = 0.3

    outlier_sigma: float = 3.0


@dataclass(frozen=True)
class BackgroundConfig:
    person_exclusion_dilate_px: int = 32
    change_exclusion_threshold: float = 40.0
    background_erode_px: int = 5

    refresh_every_n_frames: int = 3
    candidate_alpha: float = 0.05
    committed_alpha: float = 0.01
    commit_stable_after_s: float = 2.0
    refresh_freeze_after_gain_move: float = 0.03
    refresh_freeze_s: float = 1.0

    drift_check_hz: float = 8.0
    drift_scale_divisor: int = 2
    """Downscale factor for phase correlation.

    At divisor 4 (160x120) a true 3 px shift was measured back as 0.69 px on one
    axis — a 65% error. At divisor 2 (320x240) the same shift reads 3.24, -1.94.
    """
    drift_trigger_px: float = 2.0
    drift_sustain_samples: int = 15
    drift_min_gradient_energy: float = 12.0

    staleness_scale_divisor: int = 4
    stale_after_s: float = 10.0
    """A region unseen for this long is counted as stale. Enough of the frame
    going stale is the cue that the plate is describing a room that may no
    longer exist."""


@dataclass(frozen=True)
class SegmentationConfig:
    core_threshold: float = 0.65
    shell_threshold: float = 0.35
    mask_ema_alpha: float = 0.35
    """Weight of the *new* mask. ~95 ms time constant at 30 fps.

    The network output is 256x256 upsampled ~2.5x, so a single pixel of network
    jitter becomes 2-3 px of visible edge crawl without this.
    """

    close_radius_px: int = 3
    min_component_frame_fraction: float = 0.005
    """Components smaller than this fraction of the frame are dropped. An area
    rule, not largest-only: largest-only silently reintroduces the bug where a
    second person in shot is never removed."""

    feather_kernel_px: int = 9
    """Gaussian kernel *size* for the soft edge — a size, not a radius, because
    that is what ``cv2.GaussianBlur`` takes."""

    dilate_min_px: int = 4
    dilate_max_px: int = 14
    dilate_area_coefficient: float = 0.036
    """Dilation *radius* scales with sqrt(mask area).

    Over-dilation is nearly free here — the pixels being pasted are the real
    background, so pushing the seam outward lands it on pixels where plate and
    frame show the same scene. Under-dilation leaves a rim of the subject's own
    colour, which is the classic halo.
    """


@dataclass(frozen=True)
class ChangeMaskConfig:
    enabled: bool = True

    roi_min_px: int = 40
    roi_max_px: int = 170
    roi_area_coefficient: float = 0.55
    """The spatial gate's *radius*, scaled by sqrt(silhouette area).

    A gate is not optional: without one, a single auto-exposure step makes every
    pixel differ from the plate and the whole frame turns into wallpaper. But a
    fixed 40 px gate cannot reach a cast shadow, which routinely falls a body's
    width away — measured at 105 px for a seated subject. Since a shadow scales
    with its owner, so does the gate.

    Distance is only a proxy for safety, though. The real guard is
    ``max_expansion_ratio`` below, which bounds the damage directly.
    """

    max_expansion_ratio: float = 2.0
    """How much larger than the silhouette the removal may grow.

    A shadow can plausibly be twice the body's area. A blown photometric fit
    cannot — it takes the whole frame. When the change mask exceeds this, it is
    discarded entirely and the plain silhouette is used, which degrades to the
    previous behaviour rather than to a screen full of wallpaper.
    """
    max_frame_fraction: float = 0.75
    """Absolute ceiling, for when the subject is small enough that a ratio
    against their area is not a meaningful bound."""

    noise_sigma_multiplier: float = 3.0
    fallback_noise_sigma: float = 1.2
    min_threshold: float = 6.0

    small_subject_frame_fraction: float = 0.08
    """Below this the subject is far from the camera, where the segmenter is
    weakest. The change mask compensates by widening its gate and lowering its
    threshold rather than by introducing a second background model that could
    itself absorb a person who stands still."""
    small_subject_threshold_scale: float = 0.75
    small_subject_roi_scale: float = 1.5

    connectivity_gated: bool = True
    """Keep only change components touching the silhouette.

    This is what removes the cast shadow while leaving a mug that was set down
    on the desk visible: the shadow touches the person, the mug does not.
    """


@dataclass(frozen=True)
class GestureConfig:
    num_hands: int = 2
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6
    presence_confidence: float = 0.6

    engage_score: float = 0.70
    release_score: float = 0.60

    engage_hold_s: float = 0.10
    release_hold_s: float = 0.066
    """Deliberately asymmetric. Vanishing should feel considered; coming back
    should feel instant. The old code took ~233 ms in both directions."""

    lost_hand_timeout_s: float = 1.0
    """A missing hand means HOLD, not "open". After this long with no hand at
    all, fail safe back to visible."""


@dataclass(frozen=True)
class CompositeConfig:
    dissolve_s: float = 0.165
    """At 30 fps a hard cut is a single frame, which reads on video as a dropped
    frame rather than as an effect."""


@dataclass(frozen=True)
class HealthConfig:
    residual_good: float = 4.0
    residual_warn: float = 8.0
    residual_bad: float = 15.0
    chroma_shift_warn: float = 3.0
    score_residual_scale: float = 8.0
    score_shift_scale: float = 2.0
    flag_hold_s: float = 0.75


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    plate: PlateConfig = field(default_factory=PlateConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    change_mask: ChangeMaskConfig = field(default_factory=ChangeMaskConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    mirror: bool = True
    """Mirror the preview. Un-mirrored webcam video is disorienting to act in."""

    show_hud: bool = True
    debug_overlay: bool = False
