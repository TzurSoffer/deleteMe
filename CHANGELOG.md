# Changelog

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-08-08

A rewrite around one idea: the background plate is a live model, not a saved
photograph.

### Added

- **Adaptive background and lighting model.** A per-channel gain and offset is
  fitted from the plate to each frame over pixels known to still be background,
  and applied through a clipped lookup table. Against a known change of gain 1.2
  and bias -10 it recovers (1.200, -10.02).
- **A single camera session** held for the process lifetime, with a pinned
  backend and resolution, and photometry locked only when the lock can be
  confirmed by experiment.
- **Plate metadata** (`plate.json`) recording resolution, camera identity, the
  settings that were successfully pinned, and the measured sensor noise floor —
  which calibrates change detection to your camera rather than to a constant.
- **A plate quality gate.** Capture completes when the scene is empty, exposed,
  focused and settled, and reports which of those it is waiting on. The plate is
  the median of ten accepted frames.
- **Continuous plate refresh** with two-speed anti-burn-in, and a quiet
  automatic recapture when the room has been empty and the plate has drifted.
- **Camera-bump detection** by phase correlation, which prompts for a recapture
  rather than trying to warp the plate back into place.
- **A background health readout** — one score and one reason — that reports
  `BG --` when the measurement itself is untrustworthy.
- **Cast-shadow removal** via a change mask restricted to components connected
  to the silhouette, so a shadow goes but an object set down nearby stays.
  Disable with `--no-shadow`.
- **A dissolve transition** of about 165 ms, eased, rather than a hard cut.
- `--record` and `--replay`, which turn a live session into an offline fixture.
- A test suite of around 130 tests needing neither camera nor display, a CI
  matrix over Ubuntu and Windows on Python 3.11–3.13, and a benchmark tool.

### Changed

- **Segmentation replaces object detection.** `selfie_segmenter` (9.9 ms,
  244 KB) supersedes `efficientdet_lite0` (37.1 ms, 13.8 MB), giving a
  person-shaped removal instead of a rectangle.
- **Gesture recognition replaces finger counting.** The previous heuristic
  classified pointing, a peace sign and a thumbs-up as a closed fist.
- **Debounce is measured in seconds, not frames**, starts fully charged, and is
  asymmetric: about 100 ms to vanish, 66 ms to return.
- **One window.** `cv2.imshow` is gone; video renders directly into Tk.
- Packaged as `deleteme` with a `pyproject.toml` and a `src/` layout. The old
  `main.py`, `deleteMe.py` and `utils.py` are replaced by the `deleteme` package
  and the `deleteme` command.
- The plate is stored in the per-user data directory, as PNG rather than JPEG.

### Fixed

- **`requirements.txt` was uninstallable.** It contained a stray `<` after
  `mediapipe`, so `pip install -r requirements.txt` — the command in the README —
  aborted on a fresh clone.
- **A green outline survived around the deleted region.** The debug rectangle
  was drawn before the background was pasted, so its outer edge fell outside the
  pasted area: 1291 surviving pixels.
- **The camera was released and reopened between capture and effect**, which
  made the plate and the live feed photometrically mismatched by construction.
- **A plate smaller than the frame pasted the wrong crop of the room** with no
  error at all. Shapes are now checked in both directions.
- **A camera that opened but delivered no frames span at 100% CPU** with no
  preview and no way to cancel, because the retry path skipped the event pump.
- **A stored background was never reloaded**, so every launch demanded a fresh
  capture even though the file was on disk.
- **A missing hand detection was read as "hand open"**, so any dropped frame
  argued for cancelling the effect and the user had to hold a fist perfectly.
- **A fresh debouncer flipped on its first disagreeing observation**, and its
  `fps=1` default reduced the hold to zero frames.
- **Only the largest person was removed**, so a second person stayed visible.
- **The removal region was stretched to the bottom of the frame**, taking the
  desk with it.
- Models and the plate resolved against the working directory, so the app only
  ran from the repository root.
- Detectors were reconstructed on every start and never closed, reloading 21 MB
  and leaking native resources each time.
- The `LICENSE` copyright line still contained its `<>` placeholders.

## [0.1.0]

Initial release: fist-triggered removal of a detected bounding box.
