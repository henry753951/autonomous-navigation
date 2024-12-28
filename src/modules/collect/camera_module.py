# src/modules/collect/camera_module.py
import sys
import time

import cv2
import numpy as np
import pygetwindow as gw
import rerun as rr
from mss import mss

from src.modules.base_module import BaseModule
from src.view_controller import Providers, on_view_update


class CameraModule(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        self.window_name: str = "Grand Theft Auto V"
        self.frame: cv2.Mat | None = None
        self.monitor: dict[str, int] = {}

    def __mount__(self) -> None:
        self.frame = None
        windows = gw.getWindowsWithTitle(self.window_name)
        if not windows:
            self.logging.error(f"找不到名為 '{self.game_window_name}' 的視窗。請確認遊戲是否正在運行且名稱正確。")
            sys.exit(1)
        game_window = windows[0]
        tool_bar_height = 27
        self.monitor = {
            "top": game_window.top + tool_bar_height,
            "left": game_window.left + 2,
            "width": 1600,
            "height": 900,
        }
        self.logger.debug(f"Monitor: {self.monitor}")

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
        with mss() as sct:
            screen = np.array(sct.grab(self.monitor))
            self.frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

    @on_view_update(interval=1 / 10)
    def display_frame(self, providers: Providers) -> None:
        if self.frame is not None:
            providers.rerun.set_time_seconds("time", time.time())
            providers.rerun.log("camera/image/rgb", rr.Image(self.frame, opacity=0.1).compress(jpeg_quality=15))
        else:
            self.logger.warning("No frame to display.")
