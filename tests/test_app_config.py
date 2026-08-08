"""Command line and configuration.

These import ``deleteme.cli``, which deliberately pulls in no GUI code at all,
so the argument plumbing can be checked on a machine with no display and no Tk.
That plumbing is exactly where a flag silently failing to reach the config would
otherwise go unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deleteme.camera import PhotometryLock
from deleteme.cli import build_config, build_parser
from deleteme.config import AppConfig


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_defaults_are_sensible():
    config = build_config(parse())

    assert config.camera.index == 0
    assert (config.camera.width, config.camera.height) == (640, 480)
    assert config.camera.lock_photometry is True
    assert config.change_mask.enabled is True
    assert config.mirror is True


def test_camera_flags_reach_the_config():
    config = build_config(
        parse("--camera", "2", "--width", "1280", "--height", "720", "--fps", "60")
    )

    assert config.camera.index == 2
    assert (config.camera.width, config.camera.height) == (1280, 720)
    assert config.camera.fps == 60


def test_no_lock_disables_the_photometry_lock():
    assert build_config(parse("--no-lock")).camera.lock_photometry is False


def test_no_shadow_disables_the_change_mask():
    assert build_config(parse("--no-shadow")).change_mask.enabled is False


def test_no_mirror_and_debug():
    config = build_config(parse("--no-mirror", "--debug"))
    assert config.mirror is False
    assert config.debug_overlay is True


def test_record_and_replay_are_paths():
    args = parse("--record", "out", "--replay", "in")
    assert isinstance(args.record, Path)
    assert isinstance(args.replay, Path)


def test_the_default_config_is_internally_consistent():
    config = AppConfig()

    assert config.segmentation.shell_threshold < config.segmentation.core_threshold
    assert config.photometry.gain_min < 1.0 < config.photometry.gain_max
    assert config.photometry.bias_min < 0.0 < config.photometry.bias_max
    assert config.gesture.release_hold_s < config.gesture.engage_hold_s, (
        "coming back should be quicker than vanishing"
    )
    assert config.change_mask.roi_min_px < config.change_mask.roi_max_px
    assert config.segmentation.dilate_min_px < config.segmentation.dilate_max_px
    assert config.health.residual_good < config.health.residual_warn < config.health.residual_bad
    assert config.background.committed_alpha < config.background.candidate_alpha, (
        "the committed plate must move more slowly than the candidate"
    )


class TestPhotometryLockSerialisation:
    def test_round_trip(self):
        lock = PhotometryLock(
            applied={"exposure": -6.0, "auto_exposure": 0.25},
            failures=["focus"],
            verified=True,
        )
        restored = PhotometryLock.from_dict(lock.as_dict())

        assert restored.applied == lock.applied
        assert restored.failures == lock.failures
        assert restored.verified is True

    def test_missing_data_gives_an_empty_lock(self):
        lock = PhotometryLock.from_dict(None)

        assert lock.applied == {}
        assert lock.failures == []
        assert lock.verified is False

    def test_an_unverified_lock_is_the_default(self):
        """Unverified is a supported mode, not a failure: the software
        correction carries the load when a driver will not pin exposure."""
        assert PhotometryLock().verified is False


@pytest.mark.parametrize("flag", ["--version", "--help"])
def test_informational_flags_exit_cleanly(flag):
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args([flag])
    assert info.value.code == 0
