# DeleteMe

Make a fist and vanish from your own webcam feed. Open your hand and come back.

[![Preview](https://img.youtube.com/vi/-lTpwSLZpeo/maxresdefault.jpg)](https://www.youtube.com/shorts/-lTpwSLZpeo)

*(The linked short shows v0.1, which pasted a rectangle of background over a
detected bounding box. v0.2 removes a person-shaped region and their cast
shadow.)*

## The interesting problem

The trick is simple: photograph the empty room, then paste that photograph over
the person. Everything difficult about it comes from that photograph — the
*plate* — going out of date.

Rooms do not hold still. A cloud passes, a monitor changes what it is showing,
the camera's auto-exposure hunts, someone nudges the desk. A plate captured
thirty seconds ago no longer matches the pixels arriving now, and the composite
shows a visible patch where *then* has been pasted into *now*.

So DeleteMe treats the plate as a live model rather than a saved image:

- **One camera session for the whole process.** Releasing the camera and
  reopening it — which is what v0.1 did between capturing the plate and running
  the effect — makes the driver re-run auto-exposure and auto-white-balance from
  scratch, the second time with a person in shot. That alone left the plate and
  the live feed mismatched before any light in the room had changed.
- **Camera settings pinned, and pinned honestly.** Exposure, white balance and
  focus are frozen after the driver has settled, and each lock is confirmed by
  experiment rather than by trusting the return value. Drivers routinely accept
  a setting, report success, and ignore it; a naive lock measured *eleven times
  worse* drift than leaving the camera alone. A lock that cannot be confirmed is
  rolled back completely rather than left half-applied — the software correction
  is designed to carry the load on its own.
- **A per-frame photometric fit.** The plate is mapped into the current light by
  a per-channel gain and offset, fitted over pixels known to still be background.
  Against a known change of gain 1.2 and bias -10 it recovers (1.200, -10.02),
  taking the median error from 8.0 to 0.00.
- **Continuous relearning, with two speeds.** A fast candidate plate tracks what
  the room looks like now; a slow committed plate adopts a pixel only once the
  candidate has stopped arguing about it. Somebody who holds still cannot burn
  into the background.
- **It says when it has lost confidence.** One score, one reason. When the
  measurement itself is untrustworthy it shows `BG --` rather than inventing a
  number, and when the camera has been bumped it says so instead of silently
  smearing a misaligned plate over the frame.

All of that runs whether or not the effect is switched on, because an empty room
with nobody gesturing is exactly when the model can see the scene clearly.

## Install

Python 3.11 or newer, and a webcam.

```bash
git clone https://github.com/TzurSoffer/deleteMe.git
cd deleteMe
pip install -e .
```

The two MediaPipe models ship with the package, so there is nothing to download
and the app never contacts the network.

## Run

```bash
deleteme
```

or, equivalently, `python -m deleteme`.

1. Step out of shot and press **Capture background**. The app waits until the
   scene is genuinely empty, correctly exposed, in focus, and has stopped
   moving — then stacks ten frames. It tells you which of those it is still
   waiting on rather than counting down and taking whatever it gets.
2. Step back in and make a fist.

The plate is saved to your user data directory along with the camera settings it
was shot under, so the next launch is ready immediately.

### Controls

| Key | |
|---|---|
| `R` | recapture the background |
| `F` | force the effect on / off / back to gestures |
| `D` | show the diagnostic overlay |
| `Q` or `Esc` | quit |

### Useful flags

```bash
deleteme --no-shadow          # remove only the body, leave the cast shadow
deleteme --no-lock            # do not touch the camera's exposure settings
deleteme --camera 1           # a different camera
deleteme --record session/    # save raw frames for later analysis
deleteme --replay session/    # run the pipeline over a recording, no camera needed
deleteme --reset-plate        # discard the stored background
```

`--record` and `--replay` are the most useful pair for development: they turn
"it looked wrong when I waved" into a fixture you can iterate against offline.

## How it works

Each frame, in order:

1. **Capture** — a dedicated thread drains the camera into a one-deep slot, so
   the pipeline always gets the newest frame rather than the oldest queued one.
2. **Gesture** — MediaPipe's gesture recogniser, on its own thread. A fist
   engages, an open palm releases, and *seeing nothing holds the current state*
   rather than counting as "hand open".
3. **Segmentation** — a per-pixel person mask, smoothed over time before it is
   thresholded, then dilated by an amount that scales with how large the subject
   is in frame.
4. **Background model** — the plate is fitted to the current light, relearned
   where the scene is visible, and checked for drift and camera movement.
5. **Change mask** — pixels differing from the corrected plate, restricted to
   components connected to the silhouette. This is what removes your cast
   shadow while leaving a mug you just set down visible.
6. **Composite** — a feathered alpha blend inside the silhouette's bounding box,
   eased over about 165 ms so the transition reads as an effect rather than a
   dropped frame.

## Limitations

Stated plainly, because they are inherent rather than bugs:

- **No photometric model fixes a change in light *direction*.** Gain and offset
  can follow a room getting brighter; they cannot follow a lamp moving. When the
  error stays high after correction the app asks you to recapture, which is the
  honest answer rather than a more elaborate correction.
- **A blank wall defeats parts of the model.** Camera-shift detection needs
  visible detail to correlate against, and on a featureless plate the gain term
  is mathematically unidentifiable. Both are detected and reported rather than
  guessed at.
- **The segmenter is trained on selfie framing.** Mask quality drops at
  full-body distance. The change mask compensates, and the effect degrades
  gracefully, but close range looks better.
- **Locking exposure is a trade.** A setting that suits an empty room becomes
  wrong when you lean towards a window, and in a dim room a long fixed exposure
  adds motion blur. Use `--no-lock` if your camera behaves better on automatic.
- **Two people are both removed.** If that is not what you want, it is not
  currently configurable.

## Development

```bash
pip install -e ".[dev]"
pytest                                  # ~130 tests, no camera, no inference
ruff check src tests && ruff format --check src tests
python -m deleteme.tools.benchmark --plate your_room.png
```

The test suite deliberately needs neither a camera nor a display. Two seams make
that possible: `FrameSource`, which the camera, a recording and a synthetic numpy
sequence all satisfy, and `Clock`, which lets the debounce tests assert on time
instead of on frame counts.

### Performance

Measured on a 20-core Raptor Lake laptop at 640×480, minimum of 80 runs:

| stage | ms |
|---|---|
| segmentation (inference) | 4.4 |
| change mask | 3.9 |
| mask stabilisation | 3.5 |
| background model | 2.5 |
| composite | 1.1 |
| photometric fit | 1.0 |
| **total, effect engaged** | **15.4 of a 33.3 ms budget** |

Gesture recognition adds nothing to that: it runs on its own thread.

Read the `min` column, and measure on an otherwise idle machine. This is not
pedantry — the same unchanged code measured 7.5 ms and 40.8 ms an hour apart
during development, purely because a browser and a container runtime woke up in
between. The tool prints a median/min ratio so you can see whether to believe
it.

## Attribution

Bundles two Apache-2.0 licensed models from Google's MediaPipe project. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT — see [LICENSE](LICENSE).
