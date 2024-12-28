# src/modules/process/depth_module.py


import numpy as np
import rerun as rr

from src.modules.base_module import BaseModule
from src.view_controller import Providers, on_view_update


class ObjectModule(BaseModule):
    def __mount__(self) -> None:
        pass

    def __unmount__(self) -> None:
        pass

    def update(self) -> None:
        pass

    def rare_update(self) -> None:
        pass

    @on_view_update(interval=1 / 10)
    def display_frame(self, providers: Providers) -> None:
        # 定義車輛的 3D 方塊位置
        # 方塊的中心在 (0, 0, 0)，邊長為 2
        cube_center = np.array([0.0, 0.0, 0.0])
        cube_size = 2.0

        # 計算方塊的頂點
        half_size = cube_size / 2
        vertices = np.array(
            [
                [cube_center[0] - half_size, cube_center[1] - half_size, cube_center[2] - half_size],  # 左下後
                [cube_center[0] + half_size, cube_center[1] - half_size, cube_center[2] - half_size],  # 右下後
                [cube_center[0] + half_size, cube_center[1] + half_size, cube_center[2] - half_size],  # 右上後
                [cube_center[0] - half_size, cube_center[1] + half_size, cube_center[2] - half_size],  # 左上後
                [cube_center[0] - half_size, cube_center[1] - half_size, cube_center[2] + half_size],  # 左下前
                [cube_center[0] + half_size, cube_center[1] - half_size, cube_center[2] + half_size],  # 右下前
                [cube_center[0] + half_size, cube_center[1] + half_size, cube_center[2] + half_size],  # 右上前
                [cube_center[0] - half_size, cube_center[1] + half_size, cube_center[2] + half_size],  # 左上前
            ],
        )

        # 定義方塊的面（三角形組成）
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # 後面
                [4, 5, 6],
                [4, 6, 7],  # 前面
                [0, 1, 5],
                [0, 5, 4],  # 底部
                [3, 2, 6],
                [3, 6, 7],  # 頂部
                [0, 3, 7],
                [0, 7, 4],  # 左面
                [1, 2, 6],
                [1, 6, 5],  # 右面
            ],
        )

        # 定義方塊的顏色（可選）
        color = [0.0, 1.0, 0.0]  # 綠色

        # 將方塊添加到世界
        providers.rerun.log("world/vehicle", rr.Mesh3D(vertices=vertices, indices=faces, colors=color))
