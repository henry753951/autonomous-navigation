import logging

import numpy as np
import torch

from src.models.lane.backbone import ResNet


class ConvBnRelu(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ParsingNet(torch.nn.Module):
    def __init__(
        self,
        size: tuple[int, int] = (288, 800),
        pretrained: bool = True,
        backbone: str = "50",
        cls_dim: tuple[int, int, int] = (37, 10, 4),
        use_aux: bool = False,
    ) -> None:
        super().__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim  # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = int(np.prod(cls_dim))

        # Input : nchw,
        # Output: (w+1) * sample_rows * 4
        self.model = ResNet(backbone, pretrained=pretrained)

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                ConvBnRelu(128, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"]
                else ConvBnRelu(512, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                ConvBnRelu(256, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"]
                else ConvBnRelu(1024, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                ConvBnRelu(512, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"]
                else ConvBnRelu(2048, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                ConvBnRelu(384, 256, 3, padding=2, dilation=2),
                ConvBnRelu(256, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1, 1),
                # Output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

        self.pool = torch.nn.Conv2d(512, 8, 1) if backbone in ["34", "18"] else torch.nn.Conv2d(2048, 8, 1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x2, x3, fea = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode="bilinear")
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode="bilinear")
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view(-1, 1800)

        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls


def initialize_weights(*models: torch.nn.Module) -> None:
    for model in models:
        real_init_weights(model)


def real_init_weights(m: torch.nn.Module | list) -> None:
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0.0, std=0.01)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Module):
        for mini_m in m.children():
            real_init_weights(mini_m)
    else:
        logging.warning("unknown module %s", m)
