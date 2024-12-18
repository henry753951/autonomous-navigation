# src/base_module.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, TypeVar

from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.module_controller import ModuleController

T = TypeVar("T", bound="BaseModule")


class ModuleInfo:
    def __init__(self, name: str, key: str) -> None:
        self.name = name
        self.key = key


class BaseModule(ABC):
    controller: ModuleController | None = None
    is_active = True

    @abstractmethod
    def __init__(self) -> None:
        """
        Called when the class is created.

        No Controller available at this point.
        """

    def __new__(cls) -> Self:
        instance = super().__new__(cls)
        instance.module_info = ModuleInfo(instance.__class__.__name__, key="default")
        instance.logger = setup_logger(
            instance.module_info.name,
            instance.module_info.key,
        )
        return instance

    # Lifecycle hooks
    @abstractmethod
    def __mount__(self) -> None:
        """
        Called when the module is initialized.

        This method is not called during the class creation.
        If you want to, you can use the __init__ instead.
        """

    @abstractmethod
    def __sysready__(self) -> None:
        """
        Called after all modules are initialized. (Only called once)
        """

    @abstractmethod
    def __unmount__(self) -> None:
        """
        Called before the module is unmounted.
        """

    # Update methods
    @abstractmethod
    def update(self) -> None:
        """
        Called every frame.
        """

    @abstractmethod
    def rare_update(self) -> None:
        """
        Called every fixed interval.
        """

    # Utility methods
    def set_controller(self, controller: ModuleController) -> Self:
        self.controller = controller
        return self

    def set_key(self, key: str) -> BaseModule:
        """
        Set the key of the module.
        """
        self.module_info.key = key
        del self.logger
        self.logger = setup_logger(
            self.module_info.name,
            self.module_info.key,
        )
        return self

    def get_component(
        self,
        component_cls: type[T],
        key: str = "default",
    ) -> T | None:
        return self.controller.get_component(component_cls, key)

    def register_view_updates(self) -> None:
        """自動將標記的函數註冊到 ViewController"""
        if not self.controller or not self.controller.view_controller:
            self.logger.warning("Controller or view controller is not available.")

        view_controller = self.controller.view_controller

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_view_update"):
                interval = attr._view_update["interval"]  # noqa: SLF001
                view_controller.register_task(attr, interval)
