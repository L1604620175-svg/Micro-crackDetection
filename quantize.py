"""
模型量化脚本 — FP16 / INT8 / 动态量化
对比量化前后的模型体积、推理速度、精度.
"""

import os
import time

import torch

from ultralytics import YOLO


def main():
    # ===== 配置 =====
    model_path = "runs/detect/runs/crack_detect_v1/weights/best.pt"
    test_images_dir = "yolo_dataset/images/test"
    # ================

    print("=" * 60)
    print("YOLOv8 模型量化对比")
    print("=" * 60)

    original = YOLO(model_path)

    # ========== 1. 原始模型基准 ==========
    print("\n[1/4] 原始模型 (FP32)...")
    t0 = time.time()
    results_fp32 = original.val(data="yolo_dataset/data.yaml", split="test", verbose=False)
    time.time() - t0
    size_fp32 = os.path.getsize(model_path) / 1024**2
    print(f"  体积: {size_fp32:.1f} MB")
    print(f"  mAP50: {results_fp32.box.map50:.4f}, mAP50-95: {results_fp32.box.map:.4f}")
    print(f"  Recall: {results_fp32.box.mr:.4f}")

    # ========== 2. FP16 导出 ==========
    print("\n[2/4] FP16 量化 (ONNX Half)...")
    fp16_path = model_path.replace(".pt", ".onnx")  # YOLOv8 FP16导出用原名.onnx
    original.export(format="onnx", half=True, simplify=True, imgsz=640)

    size_fp16 = os.path.getsize(fp16_path) / 1024**2
    print(f"  体积: {size_fp16:.1f} MB (压缩 {size_fp32 / size_fp16:.1f}x)")

    model_fp16 = YOLO(fp16_path)
    t0 = time.time()
    results_fp16 = model_fp16.val(data="yolo_dataset/data.yaml", split="test", verbose=False)
    time.time() - t0
    print(f"  mAP50: {results_fp16.box.map50:.4f}, mAP50-95: {results_fp16.box.map:.4f}")

    # ========== 3. 动态 INT8 ==========
    print("\n[3/4] 动态 INT8 量化 (PyTorch)...")
    model_fp32 = original.model
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )
    int8_path = model_path.replace(".pt", "_int8.pt")
    torch.save({"model": model_int8.state_dict()}, int8_path)
    size_int8 = os.path.getsize(int8_path) / 1024**2
    print(f"  体积: {size_int8:.1f} MB (压缩 {size_fp32 / size_int8:.1f}x)")

    int8_yolo = YOLO(model_path)
    int8_yolo.model = model_int8
    t0 = time.time()
    results_int8 = int8_yolo.val(data="yolo_dataset/data.yaml", split="test", verbose=False)
    time.time() - t0
    print(f"  mAP50: {results_int8.box.map50:.4f}, mAP50-95: {results_int8.box.map:.4f}")

    # ========== 4. 单张推理速度 ==========
    print("\n[4/4] 推理速度对比 (640x640)...")
    test_imgs = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)[:10]]
    if not test_imgs:
        print("  无测试图片，跳过")
        return

    def benchmark(m, imgs, n=50):
        for _ in range(5):
            for img in imgs:
                m.predict(img, verbose=False)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            for img in imgs:
                m.predict(img, verbose=False)
        torch.cuda.synchronize()
        return (time.time() - t0) / (n * len(imgs)) * 1000

    ms_fp32_b = benchmark(original, test_imgs)
    ms_fp16_b = benchmark(model_fp16, test_imgs)
    print(f"  FP32: {ms_fp32_b:.1f} ms/图")
    print(f"  FP16: {ms_fp16_b:.1f} ms/图 ({ms_fp32_b / ms_fp16_b:.1f}x)")
    ms_int8_b = benchmark(int8_yolo, test_imgs)
    print(f"  INT8: {ms_int8_b:.1f} ms/图 ({ms_fp32_b / ms_int8_b:.1f}x)")

    # ========== 汇总 ==========
    print("\n" + "=" * 60)
    print("量化效果汇总")
    print("=" * 60)
    print(f"{'方案':<12} {'体积(MB)':>10} {'mAP50':>8} {'Recall':>8} {'推理':>10}")
    print("-" * 52)
    print(
        f"{'FP32 原始':<12} {size_fp32:>10.1f} {results_fp32.box.map50:>8.4f} {results_fp32.box.mr:>8.4f} {ms_fp32_b:>8.1f}ms"
    )
    print(
        f"{'FP16 ONNX':<12} {size_fp16:>10.1f} {results_fp16.box.map50:>8.4f} {results_fp16.box.mr:>8.4f} {ms_fp16_b:>8.1f}ms"
    )
    print(
        f"{'INT8 动态':<12} {size_int8:>10.1f} {results_int8.box.map50:>8.4f} {results_int8.box.mr:>8.4f} {ms_int8_b:>8.1f}ms"
    )

    print("\n模型文件:")
    print(f"  FP32: {model_path}")
    print(f"  FP16: {fp16_path}")
    print(f"  INT8: {int8_path}")


if __name__ == "__main__":
    main()
