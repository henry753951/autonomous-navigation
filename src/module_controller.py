# src/module_controller.py
from __future__ import annotations

from typing import TypeVar

from src.modules.base_module import BaseModule

T = TypeVar("T", bound=BaseModule)


class ModuleController:
    def __init__(self) -> None:
        self.components: dict[str, BaseModule] = {}

    def register_component(self, component: BaseModule) -> None:
        key = f"{component.__class__.__name__}#{component.module_info.key}"
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
            component.set_controller(self)
            component.__init_module__()
        for component in self.components.values():
            component.__ready__()
            component.__mount__()

    def update(self) -> None:
        for component in self.components.values():
            if component.is_active:
                component.update()

    def fixed_update(self) -> None:
        for component in self.components.values():
            if component.is_active:
                component.fixed_update()
