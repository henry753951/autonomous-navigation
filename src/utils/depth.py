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


def preprocess_depth(depth_image: np.ndarray) -> np.ndarray:
    # 使用 5x5 的中值濾波
    return cv2.medianBlur(depth_image, 5)


def preprocess_depth_bilateral(depth_image: np.ndarray) -> np.ndarray:
    # 參數可根據需求調整
    return cv2.bilateralFilter(depth_image, d=9, sigmaColor=75, sigmaSpace=75)


def generate_point_cloud(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    intrinsic: list[float],
) -> np.ndarray:
    """Generate 3D point cloud from depth image and intrinsic matrix."""
    # 預處理深度圖像
    depth_image = preprocess_depth(depth_image)

    h, w = depth_image.shape
    f_x, f_y, c_x, c_y = intrinsic

    # 生成 (u, v) 網格坐標
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 計算 3D 點
    x = (u - c_x) * depth_image / f_x
    y = (v - c_y) * depth_image / f_y
    z = depth_image

    points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0  # 正規化到 [0, 1]

    return points_3d, colors


def calculate_intrinsic_matrix(width: int, height: int, hfov: float) -> tuple[np.ndarray, float]:
    """
    Calculate intrinsic matrix and focal length from width, height, and horizontal field of view.

    Parameters:
    - width: int, image width in pixels
    - height: int, image height in pixels
    - hfov: float, horizontal field of view in degrees

    Returns:
    - intrinsic: np.ndarray, 3x3 intrinsic matrix
    - focal_length: float, focal length (average of f_x and f_y)
    """
    # Calculate f_x and f_y
    f_x = width / (2 * np.tan(np.radians(hfov / 2)))
    vfov = 2 * np.arctan((height / width) * np.tan(np.radians(hfov / 2)))
    f_y = height / (2 * np.tan(vfov / 2))

    # Calculate optical center
    c_x = width / 2
    c_y = height / 2

    # Construct the intrinsic matrix
    intrinsic = np.array(
        [f_x, f_y, c_x, c_y],
    )

    # Compute the average focal length
    focal_length = (f_x + f_y) / 2

    return intrinsic, focal_length
