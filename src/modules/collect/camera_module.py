# src/modules/collect/camera_module.py
import time

import cv2
import rerun as rr

from src.modules.base_module import BaseModule
from src.view_controller import Providers, on_view_update


class CameraModule(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        self._video_path: str = "data/test_videos/1.mp4"
        self._capture: cv2.VideoCapture | None = None
        self.frame: cv2.Mat | None = None

    def __mount__(self) -> None:
        self._capture = cv2.VideoCapture(self._video_path)
        self.frame = None
        if not self._capture.isOpened():
            self.logger.error(f"Failed to open video: {self._video_path}")
            self._capture = None
        else:
            self.logger.debug(f"Video opened: {self._video_path}")

    def __sysready__(self) -> None:
        pass

    def __unmount__(self) -> None:
        if self._capture:
            self._capture.release()
        cv2.destroyAllWindows()
        self.logger.warning("CameraModule is unmounted.")

    def update(self) -> None:
        pass

    def rare_update(self) -> None:
        if not self._capture:
            self.logger.error("Video capture is not initialized.")
            return

        ret: bool
        ret, self.frame = self._capture.read()
        if not ret or self.frame is None:
            self.logger.warning("No more frames to read or an error occurred.")
            return

    @on_view_update(interval=1 / 10)
    def display_frame(self, providers: Providers) -> None:
        if self.frame is not None:
            providers.rerun.set_time_seconds("time", time.time())
            providers.rerun.log("camera/image/rgb", rr.Image(self.frame).compress(jpeg_quality=50))
        else:
            self.logger.warning("No frame to display.")
