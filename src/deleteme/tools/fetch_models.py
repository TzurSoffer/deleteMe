"""Re-download the bundled model assets.

The models ship inside the package, so this is a repair tool rather than part
of startup. Downloading them on first launch was considered and rejected: it
adds a network failure mode to an application that otherwise has none, and an
app whose whole premise is removing you from a video calling out to a CDN the
moment it opens invites a question that shipping the files simply avoids.

Run with ``python -m deleteme.tools.fetch_models``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from deleteme.paths import GESTURE_MODEL, SEGMENTER_MODEL, package_root

BASE = "https://storage.googleapis.com/mediapipe-models"

#: Pinned by digest. A model silently changing underneath the thresholds in
#: config.py would be a very confusing bug to chase.
MODELS: dict[str, tuple[str, str]] = {
    SEGMENTER_MODEL: (
        f"{BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
    ),
    GESTURE_MODEL: (
        f"{BASE}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
        "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, url: str, expected: str, target_dir: Path, force: bool) -> bool:
    target = target_dir / name
    if target.is_file() and not force:
        actual = sha256(target)
        if actual == expected:
            print(f"  {name}: present and verified")
            return True
        print(f"  {name}: digest mismatch ({actual[:12]}...), re-downloading")

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: downloading from {url}")
    temporary = target.with_suffix(target.suffix + ".partial")
    urllib.request.urlretrieve(url, temporary)

    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        print(f"  {name}: FAILED — expected {expected[:12]}..., got {actual[:12]}...")
        return False

    temporary.replace(target)
    print(f"  {name}: downloaded and verified ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--dir", type=Path, default=None, help="target directory")
    args = parser.parse_args(argv)

    target_dir = args.dir or (package_root() / "models")
    print(f"Model directory: {target_dir}")
    ok = all(
        fetch(name, url, digest, target_dir, args.force) for name, (url, digest) in MODELS.items()
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
