import threading
import time

from src.module_controller import ModuleController
from src.modules.base_module import BaseModule
from src.modules.collect.camera_module import CameraModule
from src.modules.process.depth_module import DepthModule
from src.utils.logger import setup_logger
from src.view_controller import ViewController


class App:
    def __init__(self, modules: list[BaseModule] = []) -> None:
        self.view_controller: ViewController = ViewController()
        self.controller: ModuleController = ModuleController(
            view_controller=self.view_controller,
        )
        self.running: threading.Event = threading.Event()
        self.logger = setup_logger("Main", None)
        self.max_tick = {
            "update": 120,
            "rare_update": 30,
        }
        self.modules = modules
        self.initialize_modules()

    def update_loop(self) -> None:
        """執行 update 的迴圈"""
        while self.running.is_set():
            start_time: float = time.time()
            self.controller.update()
            elapsed: float = time.time() - start_time
            time.sleep(max(0, 1 / self.max_tick["update"] - elapsed))

    def rare_update_loop(self) -> None:
        """執行 rare_update 的迴圈"""
        while self.running.is_set():
            start_time: float = time.time()
            self.controller.rare_update()
            elapsed: float = time.time() - start_time
            time.sleep(max(0, 1 / self.max_tick["rare_update"] - elapsed))

    def stop(self) -> None:
        """停止執行"""
        self.view_controller.stop()
        self.running.clear()

    def initialize_modules(self) -> None:
        """初始化模組"""
        for module in self.modules:
            self.controller.register_component(module)
        # 初始化所有模組
        self.controller.initialize_modules()

    def run(self) -> None:
        """啟動應用程式.

        dsadsa
        """
        # 啟動 view controller
        self.view_controller.start()
        # 啟動執行緒
        self.running.set()
        update_thread: threading.Thread = threading.Thread(
            target=self.update_loop,
            daemon=False,
        )
        rare_update_thread: threading.Thread = threading.Thread(
            target=self.rare_update_loop,
            daemon=False,
        )
        update_thread.start()
        rare_update_thread.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.warning("Detected keyboard interrupt. Stopping...")
            self.stop()
            update_thread.join()
            rare_update_thread.join()
            self.logger.info("App stopped.")


if __name__ == "__main__":
    app: App = App(
        modules=[
            # 模組的初始化參數可以在這裡設定
            # set_key() 可設定模組的 key，沒呼叫 set_key 的話，預設為 Default
            # (當有同名模組時可以用 key 區分)
            CameraModule(),
            DepthModule(),
        ],
    )
    app.run()
