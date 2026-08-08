"""The pipeline end to end, with the models stubbed out.

This file exists because of a bug that every other test missed. The
``GestureWorker`` was constructed and handed to the pipeline but never started,
so its recognition thread never ran, ``reading`` returned its initial UNKNOWN
forever, and the fist gesture — the entire premise of the application — could
never engage the effect. It failed in complete silence: ``submit()`` happily
overwrote a pending frame, ``close()`` tolerated a thread that did not exist,
and ``error`` stayed ``None`` because no thread ever ran to set it.

Every test below the first drives ``EffectPipeline.process`` the way the app
does, rather than testing the gate in isolation, because testing the gate in
isolation is exactly what let that through.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from deleteme.background import BackgroundModel
from deleteme.config import AppConfig
from deleteme.frames import Frame, textured_frame
from deleteme.gesture import GestureReading, GestureState, GestureWorker
from deleteme.pipeline import EffectPipeline
from deleteme.plate import Plate, PlateMetadata

WIDTH, HEIGHT = 320, 240


class StubSegmenter:
    """Returns a fixed confidence map. No model, no inference."""

    def __init__(self, confidence: np.ndarray | None = None) -> None:
        self.confidence_map = (
            confidence if confidence is not None else np.zeros((HEIGHT, WIDTH), np.float32)
        )
        self.closed = False
        self.calls = 0

    def confidence(self, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.confidence_map

    def close(self) -> None:
        self.closed = True


class StubReader:
    """A gesture reader that always sees the same thing."""

    def __init__(self, reading: GestureReading) -> None:
        self.reading = reading
        self.calls = 0
        self.closed = False

    def read(self, frame: np.ndarray) -> GestureReading:
        self.calls += 1
        return self.reading

    def close(self) -> None:
        self.closed = True


def make_pipeline(reading: GestureReading | None = None, confidence=None):
    room = textured_frame(WIDTH, HEIGHT, seed=41)
    background = BackgroundModel(AppConfig())
    background.adopt(Plate(room, PlateMetadata(width=WIDTH, height=HEIGHT, noise_sigma=0.5)))
    segmenter = StubSegmenter(confidence)
    worker = None
    reader = None
    if reading is not None:
        reader = StubReader(reading)
        worker = GestureWorker(reader)  # type: ignore[arg-type]
    pipeline = EffectPipeline(background, segmenter, worker, AppConfig())  # type: ignore[arg-type]
    return pipeline, room, segmenter, reader


def drive(pipeline, image, seconds: float, start: float = 0.0, fps: int = 30):
    """Feed frames for a span of simulated time, letting the worker keep up."""
    result = None
    now = start
    for i in range(int(seconds * fps)):
        now = start + i / fps
        result = pipeline.process(Frame(image, i, now))
        time.sleep(0.002)  # give the real worker thread a slice
    return result, now


def person_confidence(coverage: float = 0.2) -> np.ndarray:
    field = np.zeros((HEIGHT, WIDTH), np.float32)
    height = int(HEIGHT * coverage * 2)
    field[HEIGHT - height :, WIDTH // 3 : 2 * WIDTH // 3] = 1.0
    return field


class TestGestureWorkerIsActuallyRunning:
    def test_a_held_fist_engages_the_effect(self):
        """The regression. Fails outright if the worker is never started."""
        pipeline, room, _seg, reader = make_pipeline(GestureReading(GestureState.CLOSED, 0.95))
        try:
            result, _ = drive(pipeline, room, seconds=1.0)
            assert reader is not None
            assert reader.calls > 0, "the recognition thread never ran"
            assert result is not None and result.engaged, "a held fist must engage the effect"
        finally:
            pipeline.close()

    def test_the_worker_thread_is_alive_after_construction(self):
        pipeline, _room, _seg, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        try:
            worker = pipeline.gesture_worker
            assert worker is not None
            assert worker._thread is not None and worker._thread.is_alive()
        finally:
            pipeline.close()

    def test_an_open_palm_leaves_the_effect_off(self):
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.95))
        try:
            result, _ = drive(pipeline, room, seconds=1.0)
            assert result is not None and not result.engaged
        finally:
            pipeline.close()

    def test_seeing_no_hand_never_engages_it_either(self):
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.UNKNOWN, 0.0))
        try:
            result, _ = drive(pipeline, room, seconds=1.0)
            assert result is not None and not result.engaged
        finally:
            pipeline.close()

    def test_close_stops_the_thread_and_releases_both_tasks(self):
        pipeline, _room, segmenter, reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        worker = pipeline.gesture_worker
        assert worker is not None

        pipeline.close()

        assert worker._thread is None
        assert reader is not None and reader.closed
        assert segmenter.closed

    def test_closing_twice_is_harmless(self):
        pipeline, _room, _seg, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        pipeline.close()
        pipeline.close()


class TestManualOverride:
    def test_force_engages_without_any_gesture(self):
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.UNKNOWN, 0.0))
        try:
            pipeline.force = True
            result, _ = drive(pipeline, room, seconds=0.5)
            assert result is not None and result.engaged
        finally:
            pipeline.close()

    def test_releasing_force_hands_control_back(self):
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.UNKNOWN, 0.0))
        try:
            pipeline.force = True
            drive(pipeline, room, seconds=0.5)
            pipeline.force = None
            result, _ = drive(pipeline, room, seconds=1.0, start=1.0)
            assert result is not None and not result.engaged
        finally:
            pipeline.close()


class TestMaskSmoothingOnTheIdlePath:
    def test_the_mask_ema_survives_idle_frames(self):
        """The idle path is where the background model learns best.

        Resetting the stabiliser on every idle frame — rather than once when the
        effect switches off — left ``mask_ema_alpha`` with no effect at all
        there, so a single frame of segmenter noise was taken at face value.
        """
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        try:
            drive(pipeline, room, seconds=0.3)
            assert pipeline.stabilizer._ema is not None, "smoothing history was thrown away"
        finally:
            pipeline.close()

    def test_a_single_noisy_frame_does_not_read_as_a_person(self):
        segmenter = StubSegmenter(np.zeros((HEIGHT, WIDTH), np.float32))
        room = textured_frame(WIDTH, HEIGHT, seed=42)
        background = BackgroundModel(AppConfig())
        background.adopt(Plate(room, PlateMetadata(width=WIDTH, height=HEIGHT, noise_sigma=0.5)))
        pipeline = EffectPipeline(background, segmenter, None, AppConfig())  # type: ignore[arg-type]
        try:
            for i in range(20):
                pipeline.process(Frame(room, i, i / 30))

            segmenter.confidence_map = person_confidence(0.3)  # one bad frame
            result = pipeline.process(Frame(room, 20, 20 / 30))

            assert not result.mask.present, "one frame of noise was accepted as a person"
        finally:
            pipeline.close()

    def test_the_stabiliser_is_cleared_once_on_switch_off_not_every_idle_frame(self):
        """Once, on the transition — so no ghost of the last silhouette fades
        into the next activation — and not repeatedly, which would disable the
        smoothing entirely on the idle path.

        Counting the resets is the point. Asserting that the history is empty
        afterwards would pass under the bug too, since the very next frame
        repopulates it.
        """
        segmenter = StubSegmenter(person_confidence(0.3))
        room = textured_frame(WIDTH, HEIGHT, seed=43)
        background = BackgroundModel(AppConfig())
        background.adopt(Plate(room, PlateMetadata(width=WIDTH, height=HEIGHT, noise_sigma=0.5)))
        pipeline = EffectPipeline(background, segmenter, None, AppConfig())  # type: ignore[arg-type]

        resets = 0
        original = pipeline.stabilizer.reset

        def counting_reset() -> None:
            nonlocal resets
            resets += 1
            original()

        pipeline.stabilizer.reset = counting_reset  # type: ignore[method-assign]

        try:
            pipeline.force = True
            for i in range(30):
                pipeline.process(Frame(room, i, i / 30))
            assert resets == 0, "nothing should be cleared while the effect is on"

            pipeline.force = False
            for i in range(30, 90):
                pipeline.process(Frame(room, i, i / 30))

            assert resets == 1, f"cleared {resets} times across 60 idle frames, expected once"
        finally:
            pipeline.close()


class TestPipelineWiring:
    def test_segmentation_runs_on_every_frame_even_while_idle(self):
        """The background model needs to know where the person is in order to
        keep them out of the plate refresh, whether or not anything is being
        deleted."""
        pipeline, room, segmenter, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        try:
            drive(pipeline, room, seconds=0.5)
            assert segmenter.calls >= 15
        finally:
            pipeline.close()

    def test_the_background_model_updates_while_idle(self):
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        try:
            brighter = np.clip(room.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
            for i in range(90):
                pipeline.process(Frame(brighter, i, i / 30))

            assert pipeline.background.photometry.gain.mean() == pytest.approx(1.2, abs=0.06)
        finally:
            pipeline.close()

    def test_an_idle_frame_is_returned_untouched(self):
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        try:
            result = pipeline.process(Frame(room, 0, 0.0))
            assert result.image is room
            assert result.strength == 0.0
        finally:
            pipeline.close()

    def test_timings_are_reported_for_each_stage(self):
        pipeline, room, _seg, _reader = make_pipeline(GestureReading(GestureState.OPEN, 0.9))
        try:
            result = pipeline.process(Frame(room, 0, 0.0))
            assert set(result.timings) >= {"segment", "background", "composite"}
        finally:
            pipeline.close()
