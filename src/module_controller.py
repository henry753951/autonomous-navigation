# src/module_controller.py
from __future__ import annotations

from typing import TypeVar

from src.modules.base_module import BaseModule
from src.utils.logger import setup_logger
from src.view_controller import ViewController

T = TypeVar("T", bound=BaseModule)


class ModuleController:
    def __init__(self, view_controller: ViewController = None) -> None:
        self.components: dict[str, BaseModule] = {}
        self.logger = setup_logger("ModuleController", None)
        self.view_controller = view_controller

    def register_component(self, component: BaseModule) -> None:
        key = f"{component.__class__.__name__}#{component.module_info.key}"
        if key in self.components:
            error_message = (
                f'Component "{component.__class__.__name__}" with key '
                f'"{component.module_info.key}" is already registered.'
                "\nKey list:\n" + "\n".join(f"- {key}" for key in self.components)
            )
            raise ValueError(error_message)

        component.set_controller(self)
        self.components[key] = component

    def get_component(
        self,
        component_cls: type[T],
        key: str = "default",
    ) -> T | None:
        component = self.components.get(f"{component_cls.__name__}#{key}")
        if isinstance(component, component_cls):
            return component
        return None

    def initialize_modules(self) -> None:
        for component in self.components.values():
            component.logger.info("🐌 Mounting module...")
            component.__mount__()
            component.logger.info("😊 Module is mounted.")
        for component in self.components.values():
            component.__sysready__()
        component.register_view_updates()
        self.logger.info("🔥 All modules are ready.")

    def update(self) -> None:
        for component in self.components.values():
            if component.is_active:
                component.update()

    def rare_update(self) -> None:
        for component in self.components.values():
            if component.is_active:
                component.rare_update()
