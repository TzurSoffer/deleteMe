from typing import Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class KeepState:
    def __init__(self, keepfor, fps=1, initialState=False):
        self.keepfor = int(keepfor*fps)
        self.i = self.keepfor  #< start saturated so the first flip debounces like the rest
        self.state = initialState

    def update(self, state):
        if state == self.state:
            self.i += 1
            if self.i > self.keepfor:
                self.i = self.keepfor
            return self.state
        self.i -= 1
        if self.i < 0:
            self.i = self.keepfor
            self.state = state
        return self.state

class PersonDetector:
    def __init__(
        self, 
        modelPath: str = "efficientdet_lite0.tflite", 
        scoreThreshold: float = 0.5
    ) -> None:
        options = vision.ObjectDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=modelPath),
            score_threshold=scoreThreshold,
            running_mode=vision.RunningMode.IMAGE,
            category_allowlist=["person"]  #< Human detection
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def detectBbox(self, frame, padding=10):
        """Return the persons bbox as (x, y, w, h). x y are topleft"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        res = self._detector.detect(frame)
        if not res.detections:
            return None

        area = 0
        bbox = None
        for detection in res.detections:
            b = detection.bounding_box
            if b.width * b.height > area:
                area = b.width * b.height
                x = max(0, int(b.origin_x)-padding)
                y = max(0, int(b.origin_y)-padding)
                w = min(width-x, int(b.width)+padding*2)
                h = height-y #< just go all the way for height (cuts out legs if using bbox)
                bbox = (x, y, w, h)

        return bbox

    def drawBox(self, frame, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame

class HandStateClassifier:
    def __init__(
        self, 
        modelPath: str = "hand_landmarker.task", 
        detectionConfidence: float = 0.5, 
        trackingConfidence: float = 0.5
        
    ) -> None:
        base_options = python.BaseOptions(model_asset_path=modelPath)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=detectionConfidence,
            min_tracking_confidence=trackingConfidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def _getExtendedFingers(self, handLandmarks) -> int:
        # Landmarks are NormalizedLandmark objects with x, y, z attributes
        wrist = handLandmarks[0]

        fingerPairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        extendedFingers = sum(
            self._distance(handLandmarks[tip], wrist) > self._distance(handLandmarks[pip], wrist)
            for tip, pip in fingerPairs
        )
        return extendedFingers

    def isHandClosed(self, frame):
        """Detect the first hand in a frame and return True if closed, False if open, None if no hand detected."""
        # Convert BGR OpenCV image to MediaPipe's Image object
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

        # Detect hand landmarks
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # result.hand_landmarks is a list of hands, each containing a list of landmarks
        extendedFingers = self._getExtendedFingers(result.hand_landmarks[0])
        if extendedFingers >= 3:
            return False
        return True

    @staticmethod
    def _distance(pointA, pointB) -> float:
        dx = pointA.x - pointB.x
        dy = pointA.y - pointB.y
        return (dx * dx + dy * dy) ** 0.5

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pd = PersonDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = pd.detectBbox(frame)
        if bbox is not None:
            frame = pd.drawBox(frame, bbox)
        cv2.imshow("Delete Me", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()