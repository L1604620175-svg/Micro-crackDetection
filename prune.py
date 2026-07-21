"""
模型剪枝脚本 — 两种方法对比
方法1: 全局非结构化 L1 剪枝（安全，减小体积但不加速）
方法2: 结构化剪枝 + 依赖感知 + 微调恢复（需要 torch-pruning 库）.
"""

import copy
import os

import torch
import torch.nn.utils.prune as prune

from ultralytics import YOLO

# ===== 配置 =====
model_path = "runs/detect/runs/crack_detect_v1/weights/best.pt"
prune_amount = 0.15  # 剪枝比例
# =================


def prune_unstructured_l1(model, amount=0.2):
    """全局 L1 非结构化剪枝（安全方式） 将权重最小的 20% 连接置零，不改变结构 优点：不破坏网络结构，可用稀疏推理加速 缺点：需要硬件支持稀疏矩阵.
    """
    model = copy.deepcopy(model)
    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.out_channels > 16:
            prune.l1_unstructured(module, name="weight", amount=amount)
            pruned_params += module.weight_mask.sum().item()
            total_params += module.weight_mask.numel()
            # 永久化剪枝
            prune.remove(module, "weight")

    sparsity = 1 - pruned_params / total_params if total_params > 0 else 0
    print(f"  全局稀疏度: {sparsity * 100:.1f}%")
    print(f"  剩余非零参数: {int(pruned_params):,} / {int(total_params):,}")
    return model


def main():
    print("=" * 60)
    print("YOLOv8 模型剪枝")
    print("=" * 60)

    # ========== 1. 原始模型 ==========
    print("\n[1/4] 加载模型...")
    original = YOLO(model_path)
    ori_params = sum(p.numel() for p in original.model.parameters())
    print(f"  原始参数量: {ori_params:,}")

    print("  评估原始模型...")
    base = original.val(data="yolo_dataset/data.yaml", split="test", verbose=False)
    print(f"  mAP50: {base.box.map50:.4f}, Recall: {base.box.mr:.4f}")

    # ========== 2. 非结构化 L1 剪枝 ==========
    print(f"\n[2/4] 全局 L1 非结构化剪枝 ({prune_amount * 100:.0f}%)...")
    pruned_model = prune_unstructured_l1(original.model, prune_amount)

    # 计算有效参数量（非零权重）
    nonzero = sum((p != 0).sum().item() for p in pruned_model.parameters())
    total = sum(p.numel() for p in pruned_model.parameters())
    print(f"  有效参数: {nonzero:,} / {total:,} ({nonzero / total * 100:.1f}%)")

    # 剪枝后评估
    pruned_yolo = YOLO(model_path)
    pruned_yolo.model = pruned_model
    print("  剪枝后精度 (微调前)...")
    prune_result = pruned_yolo.val(data="yolo_dataset/data.yaml", split="test", verbose=False)
    print(f"  mAP50: {prune_result.box.map50:.4f}, Recall: {prune_result.box.mr:.4f}")

    # 保存剪枝模型
    prune_path = model_path.replace(".pt", "_pruned.pt")
    torch.save(
        {
            "model": pruned_model.state_dict(),
            "prune_method": "l1_unstructured",
            "sparsity": 1 - nonzero / total,
        },
        prune_path,
    )

    # ========== 3. 微调恢复 ==========
    print("\n[3/4] 微调恢复 (20 epochs)...")
    pruned_yolo.model = pruned_model
    pruned_yolo.train(
        data="yolo_dataset/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device="0" if torch.cuda.is_available() else "cpu",
        workers=0,
        lr0=0.0005,
        cos_lr=True,
        single_cls=True,
        rect=True,
        amp=True,
        exist_ok=True,
        project="runs",
        name="prune_finetune",
        patience=10,
        plots=True,
        verbose=False,
    )

    # ========== 4. 微调后评估 ==========
    ft_path = "runs/detect/runs/prune_finetune/weights/best.pt"
    if not os.path.exists(ft_path):
        ft_path = "runs/detect/prune_finetune/weights/best.pt"

    print("\n[4/4] 微调后评估...")
    finetuned = YOLO(ft_path) if os.path.exists(ft_path) else pruned_yolo
    ft_metrics = finetuned.val(data="yolo_dataset/data.yaml", split="test", verbose=False)
    ft_params = sum(p.numel() for p in finetuned.model.parameters())

    # ========== 汇总 ==========
    print("\n" + "=" * 60)
    print("剪枝效果汇总")
    print("=" * 60)
    print(f"{'':<20} {'有效参数':>12} {'mAP50':>8} {'Recall':>8}")
    print("-" * 52)
    print(f"{'原始模型':<20} {ori_params:>12,} {base.box.map50:>8.4f} {base.box.mr:>8.4f}")
    print(f"{'L1剪枝(微调前)':<20} {nonzero:>12,} {prune_result.box.map50:>8.4f} {prune_result.box.mr:>8.4f}")
    print(f"{'L1剪枝+微调':<20} {ft_params:>12,} {ft_metrics.box.map50:>8.4f} {ft_metrics.box.mr:>8.4f}")

    print("\n模型文件:")
    print(f"  剪枝模型: {prune_path}")
    print(f"  微调模型: {ft_path}")


if __name__ == "__main__":
    main()
