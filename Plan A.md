- https://github.com/CAIC-AD/YOLOPv2 物件偵測 語意 車道線 (多工E2E
input : mono rgb
output : 物件偵測 語意 車道線
- https://github.com/YvanYin/Metric3D?tab=readme-ov-file Depth
input : mono rgb
output : depth image


車道線 我按準確度排
- https://github.com/hirotomusiker/clrernet  [[paper](https://arxiv.org/pdf/2305.08366)]  FPS 185 on 3090
- https://github.com/weiqingq/CLRKDNet  [[paper](https://arxiv.org/pdf/2405.12503)]  FPS 265(DLA34) on 3090
- https://github.com/czyczyyzc/CondLSTR [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf)]  算力好像要很高
- https://github.com/weiqingq/CLRKDNet  [[paper](https://arxiv.org/pdf/2405.12503)]  FPS 450(ResNet18) on 3090


**FLOW ** (純車道與路徑規劃 會撞車那種)
- Depth Est  >> Depth2PointCloud (每隔像素的三圍座標)
- 車道線偵測 >>Mapping到該像素的三圍座標
- 找道路中線、車道中線 >> 路徑規劃(先車道維持)


**FLOW ** (路徑規劃不撞車)
上面的路徑規劃要加入停止線計算
- 用點雲 看看有沒有將點雲變成 Occupancy Grid Map 的方法 (我猜還是要 Segmentation，把不是道路的部分才算 Occupancy)
- 決定 Path Planning 的截斷點


**FLOW ** (控制系統)
根據 Path Planning 的結果，決定車輛的速度與轉向