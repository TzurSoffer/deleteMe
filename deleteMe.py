import cv2
from utils import PersonDetector, HandStateClassifier, KeepState

class DeleteMe():
    def __init__(self, bgImage=None, registerFor=0.2, fps=30):
        self.personDetector = PersonDetector()
        self.handStateClassifier = HandStateClassifier()
        self.bgImage = None
        self.state = KeepState(registerFor, fps, initialState=False)
        if bgImage is not None:
            self.loadBackgroundImage(bgImage)

    def loadBackgroundImage(self, bgImage):
        self.bgImage = bgImage

    def deleteMe(self, frame):
        if self.bgImage is None:
            raise ValueError("Background image not set. Use loadBackgroundImage() to set it.")
        handClosed = self.handStateClassifier.isHandClosed(frame)
        if handClosed == None:
            # hand not found should be treated the same as hand open
            handClosed = False
        handClosed = self.state.update(handClosed)
        if handClosed == False:  #< hand open
            return frame
        bbox = self.personDetector.detectBbox(frame)
        if bbox == None:        #< person not found
            return frame
        frame = self.personDetector.drawBox(frame, bbox)

        x, y, w, h = bbox
        frame[y:y+h, x:x+w] = self.bgImage[y:y+h, x:x+w]
        return frame

if __name__ == "__main__":
    import time
    fps = 30
    cap = cv2.VideoCapture(0)
    deleteMe = DeleteMe(fps=fps)
    # bgImage = cv2.imread("bg.jpg")
    ret = False
    while ret == False:
        print("couldn't capture with camera! trying again in 100ms")
        time.sleep(0.1)
        ret, _ = cap.read()
    for _ in range(60):  #< flush camera for auto white
        ret, bgImage = cap.read()
    cv2.imwrite("bg.jpg", bgImage)
    deleteMe.loadBackgroundImage(bgImage)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputFrame = deleteMe.deleteMe(frame)
        cv2.imshow("Delete Me", outputFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1/fps)

    cap.release()
    cv2.destroyAllWindows()