"""
YOLOv8 隐裂检测训练脚本 — v3.

相比 v2 的调整（解决过拟合）：
1. 分辨率 1280→960（降分辨率提升泛化性）
2. 减弱几何增强（旋转15°→5°，剪切5°→2°）
3. 关闭 mixup（细线检测不适合混合样本）
4. 关闭 randaugment（自动增强可能产生不真实图像）
5. 减少 erasing（0.4→0.2）
6. 恢复默认 LR（960可用更高LR加速收敛）
"""

import torch

from ultralytics import YOLO


def main():
    # ========== 配置 ==========
    data_yaml = "yolo_dataset/data.yaml"
    resume_checkpoint = "runs/detect/runs/crack_detect_v3/weights/last.pt"
    project = "runs"
    name = "crack_detect_v3"
    epochs = 200
    imgsz = 960
    batch = 16
    device = "0" if torch.cuda.is_available() else "cpu"
    # =========================

    # 从 checkpoint 续训
    model = YOLO(resume_checkpoint)
    print(f"从断点续训: {resume_checkpoint}")

    # 训练配置
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=0,  # Windows必须为0
        seed=42,
        # ===== 优化器设置 =====
        lr0=0.005,  # 960分辨率用默认LR
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        # ===== 数据增强（适度） =====
        degrees=5.0,  # 旋转±5°（裂纹可能微斜）
        translate=0.1,
        scale=0.5,
        shear=2.0,  # 剪切±2°
        perspective=0.0,
        flipud=0.5,  # 上下翻转（裂纹无方向）
        fliplr=0.5,  # 左右翻转
        hsv_h=0.0,  # 灰度图无需色调增强
        hsv_s=0.0,  # 灰度图无需饱和度增强
        hsv_v=0.3,  # 亮度变化（模拟光照差异）
        # 混合增强
        mosaic=1.0,
        mixup=0.0,  # 关闭mixup（细裂纹不适合混合）
        copy_paste=0.0,
        cutmix=0.0,
        # 随机擦除
        erasing=0.2,  # 适度擦除
        # ===== 训练策略 =====
        close_mosaic=10,
        single_cls=True,
        amp=True,
        cache=True,
        # ===== 保存与验证 =====
        val=True,
        save=True,
        save_period=50,
        resume=True,  # 续训
        patience=30,  # 适度早停
        exist_ok=True,
        project=project,
        name=name,
        verbose=True,
        plots=True,
        save_json=True,
    )

    # ========== 验证最佳模型 ==========
    print("\n========== 验证最佳模型 ==========")
    best_model_path = f"{project}/detect/{name}/weights/best.pt"
    model = YOLO(best_model_path)
    metrics = model.val(data=data_yaml, split="val", imgsz=imgsz, plots=True)
    print(f"验证完成 — mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

    # ========== 导出 ONNX ==========
    print("\n========== 导出模型 ==========")
    model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"ONNX已导出到 {best_model_path.replace('.pt', '.onnx')}")

    print("\n训练完成！最佳模型:", best_model_path)


if __name__ == "__main__":
    main()
