# src/modules/process/depth_module.py

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
            depth_image = self.pred_depth
            providers.rerun.log(f"camera/depth/image{depth_image.shape}", rr.DepthImage(depth_image))
            providers.rerun.log(
                "camera/depth/intrinsic",
                rr.Pinhole(
                    width=depth_image.shape[1],
                    height=depth_image.shape[0],
                    focal_length=200,
                ),
            )