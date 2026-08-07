# DeleteMe

DeleteMe is a small OpenCV-based webcam app that removes a person from the frame when they close their fist

## Requirements

- Python 3.10+ recommended
- Webcam
- Windows, macOS, or Linux with camera access

Python packages are listed in [requirements.txt](requirements.txt).

## Install

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

If you prefer to install manually, the project currently depends on:

- opencv-python
- mediapipe

## Run

Start the GUI from the project root:

```bash
python main.py
```

## How to use

1. Launch the app.
2. Make sure the camera shows an empty background.
3. Click Capture Background.
4. Watch the live preview window and move out of frame before the countdown ends.
5. Click Start App to begin the main effect loop.

### Controls

- In the background preview window, press `q` or `Esc` to cancel capture.
- In the main OpenCV window, press `q` to exit the loop.

## Files

- [main.py](main.py) contains the Tkinter GUI and camera flow.
- [deleteMe.py](deleteMe.py) contains the main effect logic.
- [utils.py](utils.py) contains the person and hand helpers.
- [bg.jpg](bg.jpg) is the saved background image created by the app.

## Notes

- The app saves the captured background to `bg.jpg`.
- If the camera fails to open, check that no other app is using it.
- If the background capture looks wrong, clear the scene again and recapture before starting.

LICENSE
This project is hosted under the MIT LICENSE, see license file for more info