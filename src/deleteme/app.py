"""The desktop application.

One window. The previous version had two — a Tkinter panel of buttons and a
separate OpenCV HighGUI window — and that split caused three distinct bugs
rather than one: the parent window was withdrawn while the effect ran, so a user
who lost focus had no visible application at all; quitting depended on the
HighGUI window holding keyboard focus, because the only exit was ``waitKey``;
and the Tk window was frozen throughout regardless, because the camera loops
pumped ``update_idletasks``, which flushes redraws but never processes input.

Rendering video into Tk directly removes that whole class of problem rather
than patching each instance, and costs about a millisecond a frame.

The other structural decision is that nothing blocks the event loop. Warming up
the camera and probing its exposure takes seconds, so plate capture runs on a
worker thread and publishes progress; the UI thread only ever draws.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import tkinter as tk
from enum import Enum, auto
from pathlib import Path
from time import perf_counter
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from deleteme import __version__
from deleteme.background import BackgroundModel, HealthState
from deleteme.camera import CameraSession, PhotometryLock
from deleteme.config import AppConfig
from deleteme.errors import CameraError, DeleteMeError
from deleteme.frames import Frame, FrameSource, RecordingSink
from deleteme.gesture import GestureReader, GestureWorker
from deleteme.paths import plate_paths
from deleteme.pipeline import EffectPipeline
from deleteme.plate import Plate, PlateCapture, PlateStore
from deleteme.segment import PersonSegmenter
from deleteme.timers import high_resolution_timer

log = logging.getLogger(__name__)

_STATE_COLOURS = {
    HealthState.GOOD: "#3fb950",
    HealthState.WARN: "#d29922",
    HealthState.BAD: "#f85149",
    HealthState.UNKNOWN: "#6e7681",
}


class Mode(Enum):
    PREVIEW = auto()
    """No plate yet — show the camera and invite a capture."""
    CAPTURING = auto()
    RUNNING = auto()


class _CaptureWorker(threading.Thread):
    """Warms the camera, pins its photometry, and collects a plate.

    On a thread because the exposure probe genuinely takes seconds: it changes
    a camera setting and waits to see whether the picture responds, several
    times over. Doing that on the UI thread would freeze the window during the
    one operation the user is actively watching.
    """

    def __init__(
        self,
        source: FrameSource,
        segmenter: PersonSegmenter,
        config: AppConfig,
        relock: bool,
    ) -> None:
        super().__init__(name="deleteme-capture-plate", daemon=True)
        self._source = source
        self._segmenter = segmenter
        self._config = config
        self._relock = relock

        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._preview: np.ndarray | None = None
        self._progress = 0.0
        self._message = "starting the camera"
        self.plate: Plate | None = None
        self.error: BaseException | None = None
        self.finished = threading.Event()

    # ------------------------------------------------------------- published

    @property
    def preview(self) -> np.ndarray | None:
        with self._lock:
            return self._preview

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    def _publish(self, preview: np.ndarray | None, progress: float, message: str) -> None:
        with self._lock:
            if preview is not None:
                self._preview = preview
            self._progress = progress
            self._message = message

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------- run

    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self.error = exc
            log.exception("background capture failed")
        finally:
            self.finished.set()

    def _run(self) -> None:
        camera = self._source if isinstance(self._source, CameraSession) else None

        if camera is not None:
            self._publish(None, 0.0, "letting the camera settle")
            camera.warm_up()
            if self._cancel.is_set():
                return
            if self._relock:
                self._publish(None, 0.0, "checking what this camera will let us pin down")
                lock = camera.lock_photometry()
                if not lock.verified:
                    log.info("exposure could not be pinned; software correction will carry it")

        capture = PlateCapture(self._config.plate)
        started = perf_counter()

        while not self._cancel.is_set():
            frame = self._source.read()
            if frame is None:
                if perf_counter() - started > self._config.plate.timeout_s:
                    raise CameraError("the camera stopped delivering frames during capture")
                continue

            confidence = self._segmenter.confidence(frame.image)
            scene_empty = bool(confidence.max() < self._config.segmentation.core_threshold)

            quality = capture.offer(frame.image, scene_empty, perf_counter())
            self._publish(
                frame.image,
                capture.progress(),
                "hold still — capturing" if quality.ok else quality.reason,
            )

            if capture.complete:
                lock = camera.lock if camera is not None else PhotometryLock()
                backend = camera.backend_name if camera is not None else "replay"
                index = camera.config.index if camera is not None else 0
                self.plate = capture.build(backend, index, lock)
                return

            if capture.timed_out(perf_counter()):
                raise DeleteMeError(
                    f"Could not get a clean background in "
                    f"{self._config.plate.timeout_s:.0f}s — {capture.last_reason}."
                )


class DeleteMeApp:
    """The window, the loop, and the wiring between them."""

    def __init__(
        self,
        source: FrameSource,
        config: AppConfig | None = None,
        data_dir: Path | None = None,
        recorder: RecordingSink | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.source = source
        self.recorder = recorder

        image_path, metadata_path = plate_paths(data_dir)
        self.store = PlateStore(image_path, metadata_path)

        self.background = BackgroundModel(self.config)
        self.segmenter = PersonSegmenter(
            config=self.config.segmentation, fps=self.config.camera.fps
        )
        gesture_worker = GestureWorker(
            GestureReader(config=self.config.gesture, fps=self.config.camera.fps)
        )
        self.pipeline = EffectPipeline(self.background, self.segmenter, gesture_worker, self.config)

        self.mode = Mode.PREVIEW
        self._worker: _CaptureWorker | None = None
        self._auto_capture: PlateCapture | None = None
        self._flash_until = 0.0
        self._flash_text = ""
        self._debug = self.config.debug_overlay
        self._closing = False
        self._after_id: str | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._frame_times: list[float] = []
        self._stalled_since: float | None = None

        self._build_window()
        self._load_existing_plate()

    # ---------------------------------------------------------------- window

    def _build_window(self) -> None:
        width, height = self.source.size
        self.root = tk.Tk()
        self.root.title(f"DeleteMe {__version__}")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.video = tk.Label(self.root, background="#0d1117", width=width, height=height)
        self.video.pack()

        controls = ttk.Frame(self.root, padding=(10, 8))
        controls.pack(fill="x")

        self.capture_button = ttk.Button(
            controls, text="Capture background", command=self.start_capture
        )
        self.capture_button.pack(side="left")

        self.settings_button = ttk.Button(
            controls, text="Camera settings…", command=self._open_driver_settings
        )
        self.settings_button.pack(side="left", padx=(8, 0))
        if not isinstance(self.source, CameraSession):
            self.settings_button.state(["disabled"])

        self.health_label = tk.Label(controls, text="BG --", font=("Segoe UI", 10, "bold"))
        self.health_label.pack(side="right")
        self.health_dot = tk.Canvas(controls, width=12, height=12, highlightthickness=0)
        self._dot = self.health_dot.create_oval(
            2, 2, 11, 11, fill=_STATE_COLOURS[HealthState.UNKNOWN], outline=""
        )
        self.health_dot.pack(side="right", padx=(0, 6))

        self.status = tk.StringVar(value="Starting…")
        ttk.Label(self.root, textvariable=self.status, padding=(10, 0, 10, 8)).pack(
            fill="x", anchor="w"
        )

        for sequence, handler in (
            ("<Escape>", lambda _e: self.quit()),
            ("<q>", lambda _e: self.quit()),
            ("<r>", lambda _e: self.start_capture()),
            ("<d>", lambda _e: self._toggle_debug()),
            ("<f>", lambda _e: self._toggle_force()),
        ):
            self.root.bind(sequence, handler)

    def _load_existing_plate(self) -> None:
        """Adopt a stored plate so the app is usable the moment it opens.

        The old version kept its plate only in memory, so every launch demanded
        the five-second capture ritual again even though bg.jpg was sitting
        right there on disk.
        """
        plate = self.store.load()
        if plate is None:
            self.status.set("Clear the scene, then capture the background.")
            return

        width, height = self.source.size
        if (plate.metadata.width, plate.metadata.height) != (width, height):
            self.status.set(
                f"The saved background is {plate.metadata.width}x{plate.metadata.height} "
                f"but the camera is {width}x{height}. Capture a new one."
            )
            return

        if isinstance(self.source, CameraSession):
            if not plate.matches_camera(self.source.backend_name, self.source.config.index):
                self.status.set("The saved background came from another camera. Capture a new one.")
                return
            # Restoring the settings the plate was shot under is what keeps it
            # comparable to today's frames instead of letting the driver
            # reconverge somewhere slightly different.
            self.source.apply_lock(plate.metadata.lock)

        self.background.adopt(plate, now=perf_counter())
        self.mode = Mode.RUNNING
        self.status.set("Ready. Make a fist to disappear.")

    # --------------------------------------------------------------- capture

    def start_capture(self) -> None:
        if self.mode is Mode.CAPTURING:
            return
        self.mode = Mode.CAPTURING
        self.capture_button.state(["disabled"])
        self.pipeline.reset()
        if isinstance(self.source, CameraSession):
            self.source.stop_stream()

        self._worker = _CaptureWorker(
            self.source,
            self.segmenter,
            self.config,
            relock=isinstance(self.source, CameraSession),
        )
        self._worker.start()

    def _finish_capture(self, worker: _CaptureWorker) -> None:
        self._worker = None
        self.capture_button.state(["!disabled"])

        if worker.error is not None:
            messagebox.showerror("Could not capture the background", str(worker.error))
            self.status.set("Background capture failed.")
            self.mode = Mode.RUNNING if self.background.has_plate else Mode.PREVIEW
        elif worker.plate is not None:
            self.background.adopt(worker.plate, now=perf_counter())
            try:
                self.store.save(worker.plate)
            except DeleteMeError as exc:
                log.warning("could not save the plate: %s", exc)
            self.mode = Mode.RUNNING
            self.status.set("Ready. Make a fist to disappear.")
        else:
            self.mode = Mode.RUNNING if self.background.has_plate else Mode.PREVIEW

        if isinstance(self.source, CameraSession):
            self.source.start_stream()

    def _maybe_auto_recapture(self, frame: Frame, now: float) -> None:
        """Quietly rebuild the plate when the room has been empty for a while.

        Only when nobody is in shot, nothing is being deleted, and the model
        already believes the plate has drifted out of date. It is announced when
        it happens — a background that silently changes under the user is how a
        feature earns a reputation for being haunted.
        """
        health = self.background.last_health
        if self.pipeline.dissolve.value(now) > 0 or self.background.empty_for(now) < (
            self.config.plate.idle_recapture_after_s
        ):
            self._auto_capture = None
            return
        drifted = health.stale_fraction > 0.5 or health.changed_fraction > 0.15
        if not drifted and self._auto_capture is None:
            return

        capture = self._auto_capture or PlateCapture(self.config.plate)
        self._auto_capture = capture

        capture.offer(frame.image, True, now)
        if capture.complete:
            camera = self.source if isinstance(self.source, CameraSession) else None
            plate = capture.build(
                camera.backend_name if camera else "replay",
                camera.config.index if camera else 0,
                camera.lock if camera else PhotometryLock(),
            )
            self.background.adopt(plate, now=now)
            try:
                self.store.save(plate)
            except DeleteMeError as exc:
                log.warning("could not save the refreshed plate: %s", exc)
            self._auto_capture = None
            self._flash("Background refreshed", now, seconds=1.5)
        elif capture.timed_out(now):
            self._auto_capture = None

    # ------------------------------------------------------------------ loop

    def _tick(self) -> None:
        if self._closing:
            return
        started = perf_counter()

        worker = self._worker
        if worker is not None and worker.finished.is_set():
            self._finish_capture(worker)
            worker = None

        if worker is not None:
            preview = worker.preview
            if preview is not None:
                self._render(preview)
            self.status.set(f"{worker.message}  ({worker.progress:.0%})")
        else:
            self._tick_frame()

        elapsed = perf_counter() - started
        delay = max(1, int((1.0 / self.config.camera.fps - elapsed) * 1000))
        self._after_id = self.root.after(delay, self._tick)

    def _tick_frame(self) -> None:
        frame = self.source.read()
        now = perf_counter()

        if frame is None:
            self._handle_stall(now)
            return
        self._stalled_since = None

        if self.recorder is not None:
            self.recorder.write(frame)

        if self.mode is not Mode.RUNNING or not self.background.has_plate:
            self._render(frame.image)
            return

        # Give the pipeline monotonic wall-clock time. A recorded source's own
        # timestamps restart at zero on replay, which would rewind every
        # debounce and dissolve in the system.
        result = self.pipeline.process(Frame(frame.image, frame.index, now))

        self._sample_fps(now)
        self._maybe_auto_recapture(frame, now)
        self._render(result.image)
        self._update_status(now)

    def _handle_stall(self, now: float) -> None:
        if self._stalled_since is None:
            self._stalled_since = now
        waited = now - self._stalled_since
        if waited > 2.0:
            self.status.set(
                "The camera has stopped sending frames. Another application may have taken it."
            )

    # ---------------------------------------------------------------- render

    def _render(self, image: np.ndarray) -> None:
        if self.config.mirror:
            image = cv2.flip(image, 1)
        if self._debug:
            image = self._draw_debug(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if self._photo is None or (self._photo.width(), self._photo.height()) != pil.size:
            self._photo = ImageTk.PhotoImage(pil)
            self.video.configure(image=self._photo)
        else:
            self._photo.paste(pil)

    def _draw_debug(self, image: np.ndarray) -> np.ndarray:
        health = self.background.last_health
        image = image.copy()
        lines = [
            f"gain {health.gain[0]:.3f} {health.gain[1]:.3f} {health.gain[2]:.3f}",
            f"bias {health.bias[0]:+.1f} {health.bias[1]:+.1f} {health.bias[2]:+.1f}",
            f"residual {health.residual:.1f}  chroma {health.chroma_shift:.1f}",
            f"shift {health.shift_px:.2f}px  changed {health.changed_fraction:.1%}",
            f"stale {health.stale_fraction:.1%}  fit px {health.fit_pixels}",
            f"fps {self._fps():.1f}",
        ]
        if self.background.change_mask_suppressed:
            lines.append("change mask suppressed")
        for i, line in enumerate(lines):
            cv2.putText(image, line, (10, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(
                image, line, (10, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1
            )
        return image

    def _sample_fps(self, now: float) -> None:
        self._frame_times.append(now)
        del self._frame_times[:-30]

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def _update_status(self, now: float) -> None:
        health = self.background.last_health
        self.health_label.configure(text=health.label)
        self.health_dot.itemconfigure(self._dot, fill=_STATE_COLOURS[health.state])

        if now < self._flash_until:
            self.status.set(self._flash_text)
            return

        forced = self.pipeline.force
        if health.reason:
            self.status.set(health.reason)
        elif forced is not None:
            self.status.set(
                f"Effect forced {'on' if forced else 'off'} — press F again to return to gestures."
            )
        else:
            self.status.set("Ready. Make a fist to disappear.")

    def _flash(self, text: str, now: float, seconds: float = 1.5) -> None:
        self._flash_text = text
        self._flash_until = now + seconds

    # --------------------------------------------------------------- actions

    def _open_driver_settings(self) -> None:
        if isinstance(self.source, CameraSession) and not self.source.open_driver_settings():
            messagebox.showinfo(
                "Camera settings",
                "This camera's driver does not offer a settings panel through OpenCV.",
            )

    def _toggle_debug(self) -> None:
        self._debug = not self._debug

    def _toggle_force(self) -> None:
        """Cycle the manual override: gestures -> forced on -> forced off -> gestures."""
        if self.pipeline.force is None:
            self.pipeline.force = True
        elif self.pipeline.force:
            self.pipeline.force = False
        else:
            self.pipeline.force = None

    # ------------------------------------------------------------- lifecycle

    def run(self) -> None:
        if isinstance(self.source, CameraSession):
            self.source.start_stream()
        # A finer system timer, so `after(1)` means roughly a millisecond rather
        # than the default ~15.6 ms scheduling tick.
        with high_resolution_timer(1):
            self._after_id = self.root.after(0, self._tick)
            self.root.mainloop()

    def quit(self) -> None:
        """Shut down in the one order that is safe.

        Workers are joined before any native task is released. Closing a
        MediaPipe task while a call is in flight is an access violation in
        native code, not a Python exception, so there is nothing to catch.
        """
        if self._closing:
            return
        self._closing = True

        if self._after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.root.after_cancel(self._after_id)

        worker = self._worker
        if worker is not None:
            worker.cancel()
            worker.join(timeout=3.0)

        if isinstance(self.source, CameraSession):
            self.source.stop_stream()

        self.pipeline.close()

        self.source.close()
        if self.recorder is not None:
            self.recorder.close()

        with contextlib.suppress(tk.TclError):
            self.root.destroy()


def report_fatal(title: str, message: str) -> None:
    """Report a startup failure through a dialog as well as the log.

    A GUI launcher has no console to print to; the old code raised inside a
    Tkinter callback, where Tkinter swallowed the traceback into a stderr
    nobody was reading.
    """
    log.error("%s: %s", title, message)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except tk.TclError:  # pragma: no cover - headless
        print(f"{title}: {message}", file=sys.stderr)


__all__ = ["DeleteMeApp", "Mode", "report_fatal"]
