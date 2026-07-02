"""
crack_detect_v1 训练脚本 — 保守策略（综合最优）
YOLOv8s, 640px, 默认增强.
"""

import torch

from ultralytics import YOLO

# ===== 配置 =====
data_yaml = "yolo_dataset/data.yaml"
project = "runs"
name = "crack_detect_v1"
# ================

model = YOLO("yolov8s.pt")

results = model.train(
    data=data_yaml,
    epochs=150,
    imgsz=640,
    batch=16,
    device="0" if torch.cuda.is_available() else "cpu",
    workers=0,
    seed=0,
    # 优化器
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.001,
    warmup_epochs=3.0,
    cos_lr=True,
    # 数据增强 — 保守
    degrees=0.0,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.4,
    # 训练策略
    single_cls=True,
    rect=True,
    amp=True,
    cache=True,
    close_mosaic=10,
    patience=20,
    # 保存
    val=True,
    save=True,
    save_period=50,
    exist_ok=True,
    project=project,
    name=name,
    plots=True,
    save_json=True,
)

# ========== 验证最佳模型 ==========
save_dir = results.save_dir  # YOLOv8 自动返回的保存目录
best_pt = f"{save_dir}/weights/best.pt"
print(f"\n最佳模型: {best_pt}")

model = YOLO(best_pt)
metrics = model.val(data=data_yaml, split="val", imgsz=640)
print(f"验证完成 — mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

# ========== 导出 ONNX ==========
model.export(format="onnx", imgsz=640, simplify=True)
print(f"ONNX已导出到 {save_dir}/weights/best.onnx")
