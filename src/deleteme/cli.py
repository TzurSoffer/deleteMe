"""Command line entry point.

Deliberately free of any GUI import. Argument parsing has nothing to do with
the window, and keeping them apart means the CLI can be tested on a machine
with no display and no Tk at all — which is every CI runner. The window is
imported inside :func:`main`, after the arguments have been understood.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from deleteme import __version__
from deleteme.config import AppConfig
from deleteme.errors import CameraError, ModelError
from deleteme.paths import plate_paths
from deleteme.plate import PlateStore

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deleteme",
        description="Make yourself disappear from your own webcam feed.",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="do not try to pin the camera's exposure and white balance",
    )
    parser.add_argument(
        "--no-shadow",
        action="store_true",
        help="remove only the body, leaving the cast shadow in place",
    )
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    parser.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    parser.add_argument("--record", type=Path, help="write raw camera frames to a directory")
    parser.add_argument(
        "--replay", type=Path, help="replay a recorded directory instead of a camera"
    )
    parser.add_argument("--reset-plate", action="store_true", help="discard the stored background")
    parser.add_argument("--data-dir", type=Path, help="override where the plate is stored")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--version", action="version", version=f"deleteme {__version__}")
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    return dataclasses.replace(
        base,
        camera=dataclasses.replace(
            base.camera,
            index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            lock_photometry=not args.no_lock,
        ),
        change_mask=dataclasses.replace(base.change_mask, enabled=not args.no_shadow),
        mirror=not args.no_mirror,
        debug_overlay=args.debug,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(args)

    # Imported here rather than at module scope: this pulls in Tk and Pillow's
    # Tk bridge, which a headless machine may not have.
    from deleteme.app import DeleteMeApp, report_fatal
    from deleteme.camera import CameraSession
    from deleteme.frames import FrameSource, RecordingSink, ReplayFrameSource

    if args.reset_plate:
        PlateStore(*plate_paths(args.data_dir)).clear()
        log.info("stored background discarded")

    source: FrameSource
    if args.replay is not None:
        source = ReplayFrameSource(args.replay)
    else:
        camera = CameraSession(config.camera)
        try:
            camera.open()
        except CameraError as exc:
            report_fatal("Camera unavailable", str(exc))
            return 2
        source = camera

    recorder = RecordingSink(args.record) if args.record else None

    try:
        app = DeleteMeApp(source, config, data_dir=args.data_dir, recorder=recorder)
    except ModelError as exc:
        source.close()
        report_fatal("Models missing", str(exc))
        return 3

    try:
        app.run()
    finally:
        app.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
