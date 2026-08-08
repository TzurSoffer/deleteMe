# Third-party notices

DeleteMe is MIT licensed. It redistributes the model assets below, which are
not, and which carry their own terms.

## MediaPipe model assets

Both files are redistributed unmodified from Google's MediaPipe model
repository, under the Apache License 2.0. A copy of that licence is included at
[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

Digests are recorded so it can be shown that these are the upstream files and
have not been altered. `python -m deleteme.tools.fetch_models --verify-only`
checks them without touching the network.

### selfie_segmenter.tflite

- **Purpose:** per-pixel person segmentation
- **Size:** 249,537 bytes
- **SHA-256:** `191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b`
- **Source:** <https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite>
- **Model card:** <https://developers.google.com/mediapipe/solutions/vision/image_segmenter>
- **Licence:** Apache License 2.0

### gesture_recognizer.task

- **Purpose:** hand detection, landmarks, and canned gesture classification
- **Size:** 8,373,440 bytes
- **SHA-256:** `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`
- **Source:** <https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task>
- **Model card:** <https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer>
- **Licence:** Apache License 2.0

## Runtime dependencies

Installed by pip rather than redistributed here, and listed for completeness:

| Package | Licence |
|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 |
| [OpenCV](https://github.com/opencv/opencv-python) (`opencv-contrib-python`) | Apache-2.0 |
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [Pillow](https://python-pillow.org/) | MIT-CMU |

## Models no longer bundled

Earlier versions shipped `efficientdet_lite0.tflite` and
`hand_landmarker.task`, also Apache-2.0. Both were removed in v0.2.0 when the
pipeline moved to segmentation and gesture recognition. They remain in the git
history; the history has deliberately not been rewritten, because doing so would
break every existing clone and fork to reclaim about 18 MB.
