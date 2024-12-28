num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)
epochs = 20
batch_size = 24
eval_ep = 2
save_ep = 10

iou_loss_weight = 3.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

att_loss_weight = 0.0
log_loss_weight = 5.0
priors_loss_weight = 5.0

optimizer = dict(type="AdamW", lr=0.6e-3)  # 3e-4 for batchsize 8
total_iter = (55698 // batch_size) * epochs
scheduler = dict(type="CosineAnnealingLR", T_max=total_iter)
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)
work_dirs = "work_dirs/DLA_culane"

net = dict(type="Detector")
backbone = dict(type="DLAWrapper", dla="dla34", pretrained=True)
neck = dict(type="Aggregator", in_channels=[512], out_channels=64)
heads = dict(type="CLRHead", num_priors=192, refine_layers=1, fc_hidden_dim=64, sample_points=36)

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1.0, 1.0, 1.0])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_process = [
    dict(
        type="GenerateLaneLine",
        transforms=[
            dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
            dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
            dict(name="ChannelShuffle", parameters=dict(p=1.0), p=0.1),
            dict(name="MultiplyAndAddToBrightness", parameters=dict(mul=(0.85, 1.15), add=(-10, 10)), p=0.6),
            dict(name="AddToHueAndSaturation", parameters=dict(value=(-10, 10)), p=0.7),
            dict(
                name="OneOf",
                transforms=[
                    dict(name="MotionBlur", parameters=dict(k=(3, 5))),
                    dict(name="MedianBlur", parameters=dict(k=(3, 5))),
                ],
                p=0.2,
            ),
            dict(
                name="Affine",
                parameters=dict(translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)), rotate=(-10, 10), scale=(0.8, 1.2)),
                p=0.7,
            ),
            dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
        ],
    ),
    dict(type="ToTensor", keys=["img", "lane_line", "seg"]),
]

val_process = [
    dict(
        type="GenerateLaneLine",
        transforms=[
            dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
        ],
        training=False,
    ),
    dict(type="ToTensor", keys=["img"]),
]

dataset_path = "./data/CULane"
dataset_type = "CULane"
diff_path = "data/CULane/list/train_diffs.npz"
threshold = 15
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split="train",
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split="test",
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split="test",
        processes=val_process,
    ),
)

workers = 10
log_interval = 1000
# seed = 0
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
