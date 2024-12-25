import logging
import os
import sys

import cv2
import numpy as np
import scipy.special
import torch
from torchvision import transforms

from src.models.lane.model import ParsingNet


def generate_row_anchor(start: int, end: int, num_points: int) -> list:
    gap = (end - start) / (num_points - 1)  # 計算間距
    return [int(start + i * gap) for i in range(num_points)]


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 獲取輸入路徑
    if len(sys.argv) < 2:
        logging.error("Usage: python script.py <input_path>")
        sys.exit()
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        logging.error(f"Error: File {input_path} does not exist.")
        sys.exit()

    # 固定配置參數
    cfg = {
        "backbone": "18",  # ResNet backbone
        "griding_num": 200,  # Griding number
        "num_lanes": 4,  # Number of lanes
        "test_model": "./data/models/culane_18.pth",
    }

    # 驗證配置參數
    assert cfg["backbone"] in ["18", "34", "50", "101", "152", "50next", "101next", "50wide", "101wide"]

    cls_num_per_lane = 18
    row_anchor = generate_row_anchor(287, 121, cls_num_per_lane)
    img_w, img_h = 1280, 720

    # 加載模型
    net = ParsingNet(
        pretrained=False,
        backbone=cfg["backbone"],
        cls_dim=(cfg["griding_num"] + 1, cls_num_per_lane, cfg["num_lanes"]),
        use_aux=False,
    ).cuda()

    state_dict = torch.load(cfg["test_model"], map_location="cpu")["model"]
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # 圖像變換
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((288, 800)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ],
    )

    def process_frame(frame):
        vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vis = cv2.resize(vis, (1280, 720))
        resized_frame = cv2.resize(vis, (800, 288))
        img = img_transforms(resized_frame).unsqueeze(0).cuda()

        # 車道線檢測
        with torch.no_grad():
            out = net(img)

        col_sample = np.linspace(0, 800 - 1, cfg["griding_num"])
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg["griding_num"]) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg["griding_num"]] = 0
        out_j = loc

        # 繪製 row_anchor
        for anchor in row_anchor:
            y_pos = int(img_h * (anchor / 288))  # 映射到 1280x720 的圖像尺寸
            cv2.line(vis, (0, y_pos), (img_w, y_pos), (255, 0, 0), 1)  # 在圖像上畫藍線表示 row_anchor

        # 繪製車道檢測結果
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (
                            int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                            (int(img_h * (row_anchor[k] / 288)) - 1),
                        )
                        cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        return vis_bgr

    # 判斷文件類型
    file_ext = os.path.splitext(input_path)[-1].lower()
    if file_ext in [".jpg", ".png"]:
        # 圖片模式
        frame = cv2.imread(input_path)
        if frame is None:
            logging.error(f"Error: Could not read image {input_path}.")
            sys.exit()
        result = process_frame(frame)
        cv2.imshow("Lane Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif file_ext in (".mp4", ".mkv"):
        # 視頻模式
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error(f"Error: Could not read video {input_path}.")
            sys.exit()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = process_frame(frame)

            # 顯示結果
            cv2.imshow("Lane Detection", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        logging.error(f"Error: Unsupported file format {file_ext}. Supported formats are jpg, png, mp4.")
        sys.exit()
