# src/modules/collect/camera_module.py
import time

import cv2

from src.modules.base_module import BaseModule


class CameraModule(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        self._video_path: str = "data/test_videos/1.mp4"
        self._capture: cv2.VideoCapture | None = None
        self.frame: cv2.Mat | None = None
        self._last_time: float = 0.0
        self._fps: float = 0.0

    def __init_module__(self) -> None:
        self._capture = cv2.VideoCapture(self._video_path)
        self.frame = None
        if not self._capture.isOpened():
            self.logger.error(f"Failed to open video: {self._video_path}")
            self._capture = None
        else:
            self.logger.info(f"Video opened: {self._video_path}")
        self._last_time = time.time()

    def __ready__(self) -> None:
        pass

    def __mount__(self) -> None:
        self.logger.info("CameraModule is mounted.")

    def __unmount__(self) -> None:
        if self._capture:
            self._capture.release()
        cv2.destroyAllWindows()
        self.logger.warning("CameraModule is unmounted.")

    def update(self) -> None:
        pass

    def fixed_update(self) -> None:
        if not self._capture:
            self.logger.error("Video capture is not initialized.")
            return

        ret: bool
        ret, self.frame = self._capture.read()
        if not ret or self.frame is None:
            self.logger.warning("No more frames to read or an error occurred.")
            return

        # Calculate FPS
        current_time: float = time.time()
        elapsed_time: float = current_time - self._last_time
        self._last_time = current_time
        if elapsed_time > 0:
            self._fps = 1.0 / elapsed_time

        # Overlay FPS on the frame
        cv2.putText(
            self.frame,
            f"FPS: {self._fps:.2f}",
            (10, 30),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,  # Font size
            (0, 255, 0),  # Font color
            2,  # Thickness
        )

        # Display the frame
        cv2.imshow("CameraModule", self.frame)
        cv2.waitKey(1)
