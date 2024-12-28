import logging
import sys

import cv2
import numpy as np
import scipy.special
import torch
from torchvision import transforms

from src.models.lane.model import ParsingNet

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 固定配置參數
    cfg = {
        "backbone": "18",  # ResNet backbone
        "griding_num": 100,  # Griding number
        "num_lanes": 4,  # Number of lanes
        "test_model": "./data/models/tusimple_18.pth",
    }

    # 驗證配置參數
    assert cfg["backbone"] in ["18", "34", "50", "101", "152", "50next", "101next", "50wide", "101wide"]

    row_anchor = [
        64,
        68,
        72,
        76,
        80,
        84,
        88,
        92,
        96,
        100,
        104,
        108,
        112,
        116,
        120,
        124,
        128,
        132,
        136,
        140,
        144,
        148,
        152,
        156,
        160,
        164,
        168,
        172,
        176,
        180,
        184,
        188,
        192,
        196,
        200,
        204,
        208,
        212,
        216,
        220,
        224,
        228,
        232,
        236,
        240,
        244,
        248,
        252,
        256,
        260,
        264,
        268,
        272,
        276,
        280,
        284,
    ]
    cls_num_per_lane = 56
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
            transforms.ToTensor(),  # 轉換 numpy.ndarray 為 torch.Tensor
            transforms.Resize((288, 800)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ],
    )

    # 讀取單張圖像
    image_path = "./data/test_images/1cMwXM1_81ya_G3O8xAi_Wg.jpg"
    vis = cv2.imread(image_path)
    if vis is None:
        logging.error(f"Error: Could not read image {image_path}.")
        sys.exit()
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    height, width, _ = vis.shape

    # 內縮比例，例如 80% 的寬度和高度
    scale = 0.7
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 計算中上裁切區域
    x_start = (width - new_width) // 2
    y_start = 0  # 靠中上，因此從頂部開始裁切
    x_end = x_start + new_width
    y_end = y_start + new_height

    # 裁切圖像
    vis = vis[y_start:y_end, x_start:x_end]
    vis = cv2.resize(vis, (1280, 720))
    resized_frame = cv2.resize(vis, (800, 288))
    img = img_transforms(resized_frame).unsqueeze(0).cuda()

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

    print(out_j)

    # 在圖像上繪製檢測結果
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (
                        int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                        int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1,
                    )
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

    # 顯示並保存結果
    cv2.imshow("Lane Detection", vis)
    cv2.imwrite("result.png", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
