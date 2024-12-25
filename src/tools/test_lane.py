import os
import random
from typing import Any

import cv2
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from torch.backends import cudnn

from src.models.lane.registry import build_net
from src.utils.config import Config
from src.utils.net_utils import load_network


class Runner:
    def __init__(self, cfg: Any):
        """
        Initialize the Runner with a configuration object.

        :param cfg: Configuration object containing model and runtime settings.
        """
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg

        # Initialize and load network
        self.net: MMDataParallel = build_net(self.cfg)
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        load_network(self.net, self.cfg.load_from)
        self.net.eval()  # Set to evaluation mode

    def to_cuda(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Move a batch of data to CUDA.

        :param batch: Dictionary containing input data.
        :return: Dictionary with all tensor values moved to CUDA.
        """
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def inference(self, image: torch.Tensor) -> Any:
        """
        Perform inference on a single image.

        :param image: A preprocessed tensor image ready for the network.
                      Shape: [batch_size, channels, height, width].
        :return: Inference output from the model, which is model-specific.
        """
        with torch.no_grad():
            output = self.net(image)
            output = self.net.module.heads.get_lanes(output)
        return output


def read_and_infer(runner: Runner, image_path: str) -> Any:
    """
    Read an image from a file path, preprocess it, and perform inference.

    :param runner: An instance of the Runner class.
    :param image_path: Path to the input image.
    :return: Inference output from the model, which is model-specific.
    """
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Resize or preprocess the image as needed by your model
    input_size = (runner.cfg.input_width, runner.cfg.input_height)
    image_resized = cv2.resize(image, input_size)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

    # Perform inference
    output = runner.inference(image_tensor)
    return output


def main() -> None:
    # 固定參數設定
    config_path = "configs/DLA_CULane.py"  # 訓練配置檔案路徑
    load_from_path = "data/models/dla34_8087.pth"  # 預訓練權重檔案路徑
    gpus = [0]  # 使用的 GPU 編號
    seed = 0  # 隨機種子
    distillation = False  # 是否使用蒸餾技術

    # 設定 CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)

    # 讀取配置
    cfg = Config.fromfile(config_path)
    cfg.gpus = len(gpus)

    # 設定參數
    cfg.load_from = load_from_path
    cfg.resume_from = None  # 如果需要從特定檢查點恢復，可在此設定
    cfg.finetune_from = None  # 如果需要微調，可在此設定
    cfg.view = False  # 是否啟用視覺化
    cfg.seed = seed
    cfg.distillation = distillation
    cfg.teacher_model_cfg = None  # 如果使用蒸餾，可在此設定
    cfg.teacher_model_path = None  # 如果使用蒸餾，需指定教師模型檔案路徑

    # 設定工作目錄
    cfg.work_dirs = None  # 可選，若需要特定工作目錄，則在此設定

    # 啟用 cuDNN 的優化選項
    cudnn.benchmark = True

    # 初始化 Runner 並執行推理測試
    runner = Runner(cfg)

    # 直接進行推理測試
    test_image_path = "path/to/your/test/image.jpg"  # 測試圖片路徑
    output = read_and_infer(runner, test_image_path)
    print("Inference Output:", output)


if __name__ == "__main__":
    main()
