import tkinter as tk
from tkinter import ttk, messagebox
import time
import cv2
from deleteMe import DeleteMe


backgroundFile = "bg.jpg"
captureCountdownSeconds = 5


class DeleteMeApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DeleteMe")
        self.root.geometry("520x260")
        self.root.resizable(False, False)

        self.statusVar = tk.StringVar(
            value="Clear the scene, then capture a background image from the live camera preview."
        )
        self.backgroundImage = None

        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)

        title = ttk.Label(container, text="DeleteMe", font=("Segoe UI", 20, "bold"))
        title.pack(anchor="w")

        instructions = ttk.Label(
            container,
            text=(
                "Step 1: clear the scene and make sure nobody is visible. "
                "Step 2: click Capture Background, watch the live preview, and use the countdown to move out of frame. "
                "Step 3: once the background is captured, start the effect."
            ),
            wraplength=470,
        )
        instructions.pack(anchor="w", pady=(8, 12))

        buttonRow = ttk.Frame(container)
        buttonRow.pack(anchor="w", pady=(0, 12))

        self.captureButton = ttk.Button(
            buttonRow,
            text="Capture Background",
            command=self.captureBackground,
        )
        self.captureButton.pack(side="left")

        self.startButton = ttk.Button(
            buttonRow,
            text="Start App",
            command=self.startApp,
            state="disabled",
        )
        self.startButton.pack(side="left", padx=(10, 0))

        status = ttk.Label(container, textvariable=self.statusVar, wraplength=470)
        status.pack(anchor="w")

    def captureBackground(self) -> None:
        self.statusVar.set(
            f"Opening camera preview. Move away before the {captureCountdownSeconds}-second countdown ends."
        )
        self.root.update_idletasks()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the camera.")
            self.statusVar.set("Camera not available.")
            return

        frame = None
        startTime = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                elapsed = time.time() - startTime
                remaining = max(0, int(captureCountdownSeconds - elapsed + 0.999))

                previewFrame = frame.copy()
                cv2.putText(
                    previewFrame,
                    "Live background preview",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    previewFrame,
                    "Move out of frame before capture",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    previewFrame,
                    f"Capturing in {remaining}s",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

                cv2.imshow("Background Preview", previewFrame)
                self.statusVar.set(
                    f"Preview running. Capture starts in {remaining} seconds."
                )
                self.root.update_idletasks()

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.statusVar.set("Background capture cancelled.")
                    return

                if elapsed >= captureCountdownSeconds:
                    break

            if frame is None:
                messagebox.showerror(
                    "Capture error", "Could not capture a background image."
                )
                self.statusVar.set("Capture failed.")
                return
        finally:
            cap.release()
            cv2.destroyWindow("Background Preview")

        if frame is None:
            messagebox.showerror("Capture error", "Could not capture a background image.")
            self.statusVar.set("Capture failed.")
            return

        self.backgroundImage = frame.copy()
        cv2.imwrite(backgroundFile, self.backgroundImage)
        loadedBackground = cv2.imread(backgroundFile)
        if loadedBackground is not None:
            self.backgroundImage = loadedBackground

        self.statusVar.set(
            f"Background captured from the live preview and saved to {backgroundFile}."
        )
        self.startButton.config(state="normal")

    def startApp(self) -> None:
        if self.backgroundImage is None:
            messagebox.showinfo("Missing background", "Capture a background image first.")
            return

        self.root.withdraw()
        try:
            self.runCameraLoop()
        finally:
            self.root.deiconify()

    def runCameraLoop(self) -> None:
        detector = DeleteMe(bgImage=self.backgroundImage, fps=30)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the camera.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                outputFrame = detector.deleteMe(frame)
                cv2.imshow("Delete Me", outputFrame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    DeleteMeApp().run()