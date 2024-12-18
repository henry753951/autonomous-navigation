from typing import Literal

import cv2
import numpy as np
import torch


def resize_and_pad_image(
    rgb_origin: np.ndarray,
    intrinsic: list[float],
    input_size: tuple[int, int],
    padding_value: list[float] = [123.675, 116.28, 103.53],
) -> tuple[np.ndarray, list[float], list[int]]:
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    return rgb, intrinsic, pad_info


def normalize_image(
    rgb: np.ndarray,
    mean: list[float] = [123.675, 116.28, 103.53],
    std: list[float] = [58.395, 57.12, 57.375],
    device: Literal["cpu", "cuda"] = "cpu",
) -> torch.Tensor:
    mean = torch.tensor(mean).float()[:, None, None]
    std = torch.tensor(std).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    if device == "cuda":
        rgb = rgb.cuda()
        mean = mean.cuda()
        std = std.cuda()
    rgb = (rgb - mean) / std
    return rgb.unsqueeze(0).to(device)
