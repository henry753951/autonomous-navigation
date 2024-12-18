import threading
import time

from src.module_controller import ModuleController
from src.modules.collect.camera_module import CameraModule
from src.utils.logger import setup_logger


class App:
    def __init__(self) -> None:
        self.controller: ModuleController = ModuleController()
        self.running: threading.Event = threading.Event()
        self.logger = setup_logger("Main", None)
        self.max_tick = {
            "update": 120,
            "fixed_update": 30,
        }

    def update_loop(self) -> None:
        """執行 update 的迴圈"""
        while self.running.is_set():
            start_time: float = time.time()
            self.controller.update()
            elapsed: float = time.time() - start_time
            time.sleep(max(0, 1 / self.max_tick["update"] - elapsed))

    def fixed_update_loop(self) -> None:
        """執行 fixed_update 的迴圈"""

        while self.running.is_set():
            start_time: float = time.time()
            self.controller.fixed_update()
            elapsed: float = time.time() - start_time
            time.sleep(max(0, 1 / self.max_tick["fixed_update"] - elapsed))

    def stop(self) -> None:
        """停止執行"""
        self.running.clear()

    def initialize_modules(self) -> None:
        """初始化模組"""
        # 註冊模組
        self.controller.register_component(CameraModule())
        # 初始化所有模組
        self.controller.initialize_modules()

    def run(self) -> None:
        """啟動應用程式"""
        # 初始化模組
        self.initialize_modules()

        # 啟動執行緒
        self.running.set()
        update_thread: threading.Thread = threading.Thread(
            target=self.update_loop,
            daemon=False,
        )
        fixed_update_thread: threading.Thread = threading.Thread(
            target=self.fixed_update_loop,
            daemon=False,
        )
        update_thread.start()
        fixed_update_thread.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.warning("Detected keyboard interrupt. Stopping...")
            self.stop()
            update_thread.join()
            fixed_update_thread.join()
            self.logger.info("App stopped.")


if __name__ == "__main__":
    app: App = App()
    app.run()
