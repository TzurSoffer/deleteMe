"""Measure the per-frame cost of each pipeline stage.

Run with ``python -m deleteme.tools.benchmark``.

Reports the **minimum** over many repetitions as well as the median. The minimum
is the honest estimate of what the code costs: it is the run that happened to
get a clean slice of CPU. Means and percentiles on a developer machine mostly
measure whatever else is running — the same code measured 7.5 ms and 40.8 ms an
hour apart on this machine purely because a browser and a container runtime woke
up in between. The median is reported alongside so the gap between them tells
you how contended the machine was while you measured.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from deleteme.background import BackgroundModel
from deleteme.composite import Compositor
from deleteme.config import AppConfig
from deleteme.frames import textured_frame
from deleteme.photometry import fit_gain_bias
from deleteme.plate import Plate, PlateMetadata
from deleteme.segment import MaskStabilizer, PersonSegmenter

FRAME_BUDGET_MS = 1000.0 / 30.0


def raise_priority() -> str:
    """Ask the OS for a better slice, so the numbers describe the code.

    Without this the measurement is mostly a report on whatever else the
    machine felt like doing. Best effort — a refusal is not worth failing over,
    it just means the numbers are noisier.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            HIGH_PRIORITY_CLASS = 0x00000080
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS):
                return "high"
        except (AttributeError, OSError):
            pass
        return "default (elevation refused)"

    try:
        import os

        os.nice(-5)
        return "nice -5"
    except (OSError, AttributeError):
        return "default (needs privileges to raise)"


def measure(label: str, fn: Callable[[int], object], repeats: int = 40, warmup: int = 8) -> dict:
    for i in range(warmup):
        fn(i)
    samples = []
    for i in range(repeats):
        start = time.perf_counter()
        fn(i)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "label": label,
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
    }


def report(rows: list[dict]) -> None:
    width = max(len(r["label"]) for r in rows)
    print(f"\n{'stage'.ljust(width)}   {'min':>7} {'median':>7} {'max':>7}   (ms)")
    print("-" * (width + 32))
    for row in rows:
        label = row["label"].ljust(width)
        print(f"{label}   {row['min']:7.2f} {row['median']:7.2f} {row['max']:7.2f}")
    contention = statistics.median(r["median"] / max(r["min"], 1e-6) for r in rows)
    print(f"\nmedian/min ratio across stages: {contention:.2f}x", end="  ")
    if contention < 1.3:
        print("(machine was quiet — these numbers are trustworthy)")
    else:
        print("(machine was busy — trust the min column, and rerun when it is idle)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plate", type=Path, help="a real room photo to benchmark against")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--skip-models", action="store_true", help="skip real inference")
    args = parser.parse_args(argv)

    plate_image: np.ndarray | None = None
    if args.plate and args.plate.is_file():
        plate_image = cv2.imread(str(args.plate))
    if plate_image is None:
        plate_image = textured_frame(640, 480, seed=7)
    height, width = plate_image.shape[:2]
    print(f"Frame size {width}x{height}, budget {FRAME_BUDGET_MS:.1f} ms at 30 fps")
    print(f"Process priority: {raise_priority()}")

    rng = np.random.default_rng(0)
    frames = [
        np.clip(
            plate_image.astype(np.float32) * (1 + 0.12 * np.sin(i / 9))
            + rng.normal(0, 0.6, plate_image.shape),
            0,
            255,
        ).astype(np.uint8)
        for i in range(24)
    ]
    person = np.zeros((height, width), np.uint8)
    cv2.ellipse(person, (width // 2, height * 5 // 8), (70, 170), 0, 0, 360, 255, -1)
    occupied = frames[0].copy()
    occupied[person > 0] = (38, 34, 30)

    plate = Plate(plate_image, PlateMetadata(width=width, height=height, noise_sigma=0.5))
    config = AppConfig()

    rows: list[dict] = []

    rows.append(
        measure(
            "photometry fit",
            lambda i: fit_gain_bias(plate_image, frames[i % 24], None, config.photometry),
            args.repeats,
        )
    )

    idle = BackgroundModel(config)
    idle.adopt(plate)
    rows.append(
        measure(
            "background.update (idle)",
            lambda i: idle.update(frames[i % 24], None, now=i / 30),
            args.repeats,
        )
    )

    busy = BackgroundModel(config)
    busy.adopt(plate)
    rows.append(
        measure(
            "background.update (person)",
            lambda i: busy.update(occupied, person, now=i / 30),
            args.repeats,
        )
    )
    rows.append(
        measure(
            "change_mask",
            lambda i: busy.change_mask(occupied, person, float((person > 0).mean())),
            args.repeats,
        )
    )

    compositor = Compositor(config.composite)
    alpha = cv2.GaussianBlur(person, (9, 9), 0).astype(np.float32) / 255.0
    rows.append(
        measure(
            "composite",
            lambda i: compositor.compose(occupied, busy.corrected_plate(), alpha, 1.0),
            args.repeats,
        )
    )

    stabilizer = MaskStabilizer(config.segmentation)
    confidence = (person > 0).astype(np.float32)
    rows.append(measure("mask stabiliser", lambda i: stabilizer.update(confidence), args.repeats))

    if not args.skip_models:
        segmenter = PersonSegmenter(config=config.segmentation, fps=config.camera.fps)
        rows.append(
            measure(
                "segmenter (inference)",
                lambda i: segmenter.confidence(frames[i % 24]),
                args.repeats,
            )
        )
        segmenter.close()

    report(rows)

    engaged = sum(
        r["min"]
        for r in rows
        if r["label"]
        in {
            "background.update (person)",
            "change_mask",
            "composite",
            "mask stabiliser",
            "segmenter (inference)",
        }
    )
    print(
        f"\nEffect engaged, main thread: {engaged:.2f} ms of a {FRAME_BUDGET_MS:.1f} ms budget "
        f"({engaged / FRAME_BUDGET_MS:.0%}). Gesture recognition runs on its own thread."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
