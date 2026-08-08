"""Camera photometry, against a stand-in for a badly behaved driver.

No real camera is opened. The interesting behaviour here is not "does OpenCV
work" but "what happens when the driver lies", and a fake is the only way to
stage that deterministically.

The fake is modelled on what a real DirectShow webcam actually did during
development: it accepts ``CAP_PROP_AUTO_EXPOSURE``, returns ``True``, reports
``-1`` when read back regardless of what was set, and only genuinely hands over
control for one particular sentinel value.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deleteme.camera import CameraSession, PhotometryLock
from deleteme.config import CameraConfig

AUTO_EXPOSURE_ON = 0.75


class FakeCapture:
    """A driver that reports success far more often than it obeys."""

    def __init__(
        self,
        manual_sentinel: float | None = 0.25,
        *,
        opaque_auto_exposure: bool = True,
        honour_white_balance: bool = True,
        honour_focus: bool = False,
    ) -> None:
        self.manual_sentinel = manual_sentinel
        self.opaque_auto_exposure = opaque_auto_exposure
        self.honour_white_balance = honour_white_balance
        self.honour_focus = honour_focus

        self.manual = False
        self.exposure = -6.0
        self.wb_temperature = 4600.0
        self.focus = 30.0
        self.auto_wb = 1.0
        self.autofocus = 1.0
        self.sets: list[tuple[int, float]] = []

    # -- the OpenCV surface CameraSession uses --------------------------------

    def set(self, prop: int, value: float) -> bool:
        self.sets.append((prop, value))
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            # Always claims success. Only actually switches to manual for the
            # one sentinel this driver happens to want.
            self.manual = self.manual_sentinel is not None and value == self.manual_sentinel
            return True
        if prop == cv2.CAP_PROP_EXPOSURE:
            self.exposure = value
            return True
        if prop == cv2.CAP_PROP_AUTO_WB:
            if not self.honour_white_balance:
                return False
            self.auto_wb = value
            return True
        if prop == cv2.CAP_PROP_WB_TEMPERATURE:
            self.wb_temperature = value
            return True
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            if not self.honour_focus:
                return False
            self.autofocus = value
            return True
        if prop == cv2.CAP_PROP_FOCUS:
            self.focus = value
            return True
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            # The behaviour that caused the bug: an unreadable property.
            return -1.0 if self.opaque_auto_exposure else (0.25 if self.manual else 0.75)
        if prop == cv2.CAP_PROP_EXPOSURE:
            return self.exposure
        if prop == cv2.CAP_PROP_WB_TEMPERATURE:
            return self.wb_temperature
        if prop == cv2.CAP_PROP_FOCUS:
            return self.focus
        if prop == cv2.CAP_PROP_AUTO_WB:
            return self.auto_wb
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            return self.autofocus
        if prop == cv2.CAP_PROP_GAIN:
            return 1.0
        return 0.0

    def read(self):
        """Brightness follows exposure only while genuinely in manual mode."""
        level = 120 + (self.exposure + 6.0) * 25 if self.manual else 120
        image = np.full((48, 64, 3), int(np.clip(level, 0, 255)), np.uint8)
        return True, image

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass


def session_with(capture: FakeCapture) -> CameraSession:
    session = CameraSession(CameraConfig(warmup_frames=2))
    # Injecting the capture directly: `open()` would go looking for real
    # hardware, which is the one thing these tests must not depend on.
    session._cap = capture  # type: ignore[assignment]
    session.backend_name = "fake"
    session._size = (64, 48)
    return session


class TestPhotometryLock:
    def test_the_sentinel_that_worked_is_recorded_not_the_read_back(self):
        """The bug real hardware exposed and no synthetic test would have.

        DSHOW reports -1 for AUTO_EXPOSURE whatever was set. Recording the
        read-back wrote -1 into the plate metadata, so re-applying it on the
        next launch quietly failed to restore manual mode — losing the one
        property that makes a stored plate comparable to today's frames.
        """
        capture = FakeCapture(manual_sentinel=0.25, opaque_auto_exposure=True)
        session = session_with(capture)

        lock = session.lock_photometry()

        assert lock.verified
        assert lock.applied["auto_exposure"] == 0.25
        assert lock.applied["auto_exposure"] != -1.0

    def test_re_applying_a_lock_restores_manual_mode(self):
        capture = FakeCapture()
        session = session_with(capture)
        lock = session.lock_photometry()

        capture.manual = False  # as a fresh session would start
        restored = session.apply_lock(lock)

        assert "auto_exposure" in restored
        assert capture.manual, "the replayed value must actually take effect"

    def test_a_driver_that_never_obeys_is_rolled_back_completely(self):
        """A half-applied lock measured eleven times worse drift than none.

        Forcing EXPOSURE while AUTO_EXPOSURE was silently refused leaves the
        driver fighting itself, so the failure path must hand control back
        rather than leave the camera in a mixed state.
        """
        capture = FakeCapture(manual_sentinel=None)
        session = session_with(capture)

        lock = session.lock_photometry()

        assert not lock.verified
        assert "exposure" in lock.failures
        assert "exposure" not in lock.applied
        assert (cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_ON) in capture.sets, (
            "auto exposure must be handed back, not left half-applied"
        )

    def test_an_unusual_sentinel_is_still_found(self):
        """Documented values are not reliable, so each is tried in turn."""
        capture = FakeCapture(manual_sentinel=1.0)
        session = session_with(capture)

        lock = session.lock_photometry()

        assert lock.verified
        assert lock.applied["auto_exposure"] == 1.0

    def test_a_refused_property_is_reported_rather_than_assumed(self):
        capture = FakeCapture(honour_focus=False)
        session = session_with(capture)

        lock = session.lock_photometry()

        assert "focus" in lock.failures
        assert "auto_focus" not in lock.applied

    def test_locking_can_be_switched_off_entirely(self):
        session = CameraSession(CameraConfig(lock_photometry=False))
        session._cap = FakeCapture()  # type: ignore[assignment]

        lock = session.lock_photometry()

        assert not lock.verified
        assert lock.applied == {}
        assert lock.failures == ["disabled by configuration"]

    def test_white_balance_is_pinned_to_the_value_it_converged_on(self):
        capture = FakeCapture()
        capture.wb_temperature = 5200.0
        session = session_with(capture)

        lock = session.lock_photometry()

        assert lock.applied["auto_white_balance"] == 0.0
        assert lock.applied["wb_temperature"] == pytest.approx(5200.0)


class TestSessionLifetime:
    def test_reading_without_opening_is_a_clear_error(self):
        from deleteme.errors import CameraError

        with pytest.raises(CameraError, match="not open"):
            CameraSession().read()

    def test_transient_read_failures_accumulate_rather_than_abort(self):
        """The previous version broke out of its loop on the first failure,
        killing the whole effect on any USB hiccup."""
        capture = FakeCapture()
        session = session_with(capture)
        capture.read = lambda: (False, None)  # type: ignore[method-assign]

        for _ in range(3):
            assert session.read() is None
        assert session.consecutive_failures == 3
        assert not session.stalled

    def test_a_sustained_outage_is_eventually_called_stalled(self):
        capture = FakeCapture()
        session = session_with(capture)
        capture.read = lambda: (False, None)  # type: ignore[method-assign]

        for _ in range(session.config.stall_tolerance_frames + 1):
            session.read()

        assert session.stalled

    def test_a_successful_read_clears_the_failure_count(self):
        capture = FakeCapture()
        session = session_with(capture)
        capture.read = lambda: (False, None)  # type: ignore[method-assign]
        session.read()
        assert session.consecutive_failures == 1

        capture.read = FakeCapture().read  # type: ignore[method-assign]
        assert session.read() is not None
        assert session.consecutive_failures == 0

    def test_closing_twice_is_harmless(self):
        session = session_with(FakeCapture())
        session.close()
        session.close()


def test_an_empty_lock_serialises_and_restores():
    assert PhotometryLock.from_dict(PhotometryLock().as_dict()).applied == {}
