# src/base_module.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

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
    def __init__(self, key: str = "default") -> None:
        """
        Called when the class is created.

        No Controller available at this point.
        Remember to call the super().__init__() if you override this method.
        """
        self.module_info = ModuleInfo(self.__class__.__name__, key)
        self.logger = setup_logger(self.module_info.name, self.module_info.key)

    # Lifecycle hooks
    @abstractmethod
    def __init_module__(self) -> None:
        """
        Called when the module is initialized.

        This method is not called during the class creation.
        If you want to, you can use the __init__ instead.
        """

    @abstractmethod
    def __ready__(self) -> None:
        """
        Called after all modules are initialized. (Only called once)
        """

    @abstractmethod
    def __mount__(self) -> None:
        """
        Called after the module is mounted.
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
    def fixed_update(self) -> None:
        """
        Called every fixed interval.
        """

    # Utility methods
    def set_controller(self, controller: ModuleController) -> None:
        self.controller = controller

    def get_component(
        self,
        component_cls: type[T],
        key: str | None = "default",
    ) -> T | None:
        return self.controller.get_component(component_cls, key)
