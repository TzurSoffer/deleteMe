"""Filesystem locations for model assets and per-user state.

Two rules here, both learned from the previous version:

* Model paths resolve against this file, never the working directory. The old
  code used bare relative paths, so the app only ran when the shell happened to
  be sitting in the repository root.
* Nothing is written into the installation directory. The captured plate is
  user state and belongs in the platform's data directory, which also means an
  installed copy works without write access to site-packages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from deleteme.errors import ModelError

APP_NAME = "DeleteMe"

SEGMENTER_MODEL = "selfie_segmenter.tflite"
GESTURE_MODEL = "gesture_recognizer.task"


def package_root() -> Path:
    """Directory containing this package."""
    return Path(__file__).resolve().parent


def model_path(name: str) -> Path:
    """Absolute path to a bundled model asset.

    MediaPipe loads models through native code that does not understand MSYS or
    other shell-mangled paths, so this always returns a fully resolved path.
    """
    path = package_root() / "models" / name
    if not path.is_file():
        raise ModelError(
            f"Model asset {name!r} is missing from {path.parent}. "
            "Reinstall the package, or run `python -m deleteme.tools.fetch_models` "
            "to download it."
        )
    return path


def user_data_dir(app_name: str = APP_NAME) -> Path:
    """Per-user data directory, following each platform's convention."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / app_name


def plate_paths(data_dir: Path | None = None) -> tuple[Path, Path]:
    """Return ``(image_path, metadata_path)`` for the stored background plate.

    The plate is PNG, not JPEG. The composite seam puts plate pixels directly
    against live camera pixels, and JPEG ringing along that boundary is visible.
    """
    root = data_dir if data_dir is not None else user_data_dir()
    return root / "plate.png", root / "plate.json"
