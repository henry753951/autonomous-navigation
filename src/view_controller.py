# src/view_controller.py
import inspect
import threading
import time
from collections.abc import Callable

import rerun as rr

from src.utils.logger import setup_logger


class Providers:
    def __init__(self) -> None:
        self.rerun = rr
        rr.init("view_controller")
        rr.spawn(memory_limit="1500MB")


class ViewController:
    def __init__(self) -> None:
        self.logger = setup_logger(self.__class__.__name__, None)
        self.providers = Providers()
        self._running = threading.Event()
        self._tasks: list[dict] = []

    def start(self) -> None:
        """Start the view controller."""
        self._running.set()
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        self.logger.info("View controller started.")

    def stop(self) -> None:
        """Stop the view controller."""
        self._running.clear()
        self.logger.info("View controller stopped.")

    def _run(self) -> None:
        while self._running.is_set():
            current_time = time.time()
            for task_info in self._tasks:
                if current_time - task_info["last_run"] >= task_info["interval"]:
                    self._execute_task(
                        task=task_info["task"],
                        providers=self.providers,
                    )
                    task_info["last_run"] = current_time
            elapsed = time.time() - current_time
            time.sleep(max(0.01 - elapsed, 0))

    def _execute_task(self, task: Callable, **available_args) -> None:
        """Execute a task with available arguments."""
        sig = inspect.signature(task)
        kwargs = {
            name: available_args[name]
            for name, param in sig.parameters.items()
            if name in available_args
        }
        task(**kwargs)

    def register_task(self, task: Callable, interval: float) -> None:
        """Register a task to be executed every interval seconds."""
        self._tasks.append({"task": task, "interval": interval, "last_run": 0.0})


def on_view_update(interval: float) -> Callable:
    def decorator(func: Callable) -> Callable:
        func._view_update = {  # noqa: SLF001
            "interval": interval,
        }
        return func

    return decorator
