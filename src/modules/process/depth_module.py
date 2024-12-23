# src/modules/process/depth_module.py

import time

import cv2
import rerun as rr
import torch

import src.utils.depth as depth_utils
from src.modules.base_module import BaseModule
from src.modules.collect.camera_module import CameraModule
from src.view_controller import Providers, on_view_update


class DepthModule(BaseModule):
    def __mount__(self) -> None:
        self.model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True, trust_repo=True)
        self.camera_module = self.get_component(CameraModule)
        self.pred_depth = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self._fps: float = 0.0

    def __unmount__(self) -> None:
        pass

    def update(self) -> None:
        if self.camera_module.frame is None:
            return
        rgb_origin = self.camera_module.frame[:, :, ::-1]
        intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
        input_size = (616, 1064)  # for vit model

        rgb, intrinsic, pad_info = depth_utils.resize_and_pad_image(rgb_origin, intrinsic, input_size)
        # Normalize the image
        rgb = depth_utils.normalize_image(rgb, device=self.device)

        # Inference
        with torch.no_grad():
            self.pred_depth, confidence, output_dict = self.model.inference({"input": rgb})

    def rare_update(self) -> None:
        pass

    @on_view_update(interval=1 / 10)
    def display_frame(self, providers: Providers) -> None:
        if self.pred_depth is not None:
            depth_image = self.pred_depth.squeeze().cpu().numpy()
            h, w = self.camera_module.frame.shape[:2]

            # Resize the depth image to the original size
            depth_image = cv2.resize(depth_image, (w, h), interpolation=cv2.INTER_LINEAR)
            hfov = 90
            intrinsic = depth_utils.calculate_intrinsic_matrix(w, h, hfov)
            points_3d, colors = depth_utils.generate_point_cloud(depth_image, self.camera_module.frame, intrinsic)

            providers.rerun.set_time_seconds("time", time.time())
            providers.rerun.log(
                "camera/image",
                rr.Pinhole(
                    width=depth_image.shape[1],
                    height=depth_image.shape[0],
                    focal_length=0.5 * depth_image.shape[1],
                    image_plane_distance=40.0,
                ),
            )

            providers.rerun.log("camera/image/depth", rr.DepthImage(depth_image))
            providers.rerun.log("world/points", rr.Points3D(points_3d, colors=colors, radii=0.05))
