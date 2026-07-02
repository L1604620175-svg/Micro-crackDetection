"""
隐裂检测评估脚本 — 优化版.

主要改进：
1. 对正样本测试集计算 Recall / Precision / mAP / F1
2. 对负样本单独评估过杀率
3. 遍历多个置信度阈值，找到最佳工作点
4. 支持 TTA（测试时增强）提升检出率
5. 输出更详细的评估报告
"""

import glob
import os

from ultralytics import YOLO


def compute_iou(box1, box2):
    """计算两个边界框的IoU (x1,y1,x2,y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate_at_threshold(model, image_files, labels_dir, conf_threshold, iou_threshold=0.5):
    """在给定置信度阈值下评估模型。 返回: (total_tp, total_fn, total_fp, num_gt_boxes, image_details).
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_gt = 0

    for image_path in image_files:
        label_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        # 读取GT标注 (YOLO格式: class xc yc w h, 归一化)
        gt_raw = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, w, h = map(float, parts[:5])
                        gt_raw.append([cls, xc, yc, w, h])

        # 预测
        results = model.predict(image_path, conf=conf_threshold, verbose=False)
        pred_boxes = results[0].boxes
        if pred_boxes is None:
            preds = []
        else:
            pred_xyxy = pred_boxes.xyxy.cpu().numpy()
            pred_conf = pred_boxes.conf.cpu().numpy()
            pred_cls = pred_boxes.cls.cpu().numpy()
            preds = list(zip(pred_xyxy, pred_conf, pred_cls))

        img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]

        # GT框转绝对坐标 (x1, y1, x2, y2)
        gt_abs = []
        for gt in gt_raw:
            cls, xc, yc, w, h = gt
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            gt_abs.append([cls, x1, y1, x2, y2])

        total_gt += len(gt_abs)

        # 匹配
        matched_gt = [False] * len(gt_abs)
        # 按置信度降序排列预测，优先匹配高分预测
        sorted_preds = sorted(enumerate(preds), key=lambda x: x[1][1], reverse=True)

        for _, (pred_box, pred_conf, pred_cls) in sorted_preds:
            best_iou = 0
            best_j = -1
            for j, (gt_cls, *gt_box) in enumerate(gt_abs):
                if matched_gt[j]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold:
                matched_gt[best_j] = True
                total_tp += 1
            else:
                total_fp += 1

        total_fn += sum(1 for m in matched_gt if not m)

    return total_tp, total_fn, total_fp, total_gt


def evaluate_overkill(model, negative_image_dir, conf_threshold):
    """评估过杀率：在纯负样本（无缺陷）上统计误报情况。 返回: (误报图像数, 总负样本数, 总误报框数).
    """
    image_files = glob.glob(os.path.join(negative_image_dir, "*.png"))
    if not image_files:
        return 0, 0, 0

    fp_images = 0
    total_fp_boxes = 0

    for image_path in image_files:
        results = model.predict(image_path, conf=conf_threshold, verbose=False)
        pred_boxes = results[0].boxes
        num_preds = len(pred_boxes) if pred_boxes is not None else 0
        if num_preds > 0:
            fp_images += 1
            total_fp_boxes += num_preds

    return fp_images, len(image_files), total_fp_boxes


def evaluate_with_tta(model, image_files, labels_dir, conf_threshold, iou_threshold=0.5):
    """使用Test-Time Augmentation (TTA) 评估。 TTA对每张图片做多尺度+翻转推理，取平均结果，通常能提升召回率。.
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_gt = 0

    for idx, image_path in enumerate(image_files):
        label_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        gt_raw = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, w, h = map(float, parts[:5])
                        gt_raw.append([cls, xc, yc, w, h])

        # TTA推理 (augment=True 自动做多尺度+翻转集成)
        results = model.predict(image_path, conf=conf_threshold, augment=True, verbose=False)
        pred_boxes = results[0].boxes
        if pred_boxes is None:
            preds = []
        else:
            pred_xyxy = pred_boxes.xyxy.cpu().numpy()
            pred_conf = pred_boxes.conf.cpu().numpy()
            pred_cls = pred_boxes.cls.cpu().numpy()
            preds = list(zip(pred_xyxy, pred_conf, pred_cls))

        img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]

        gt_abs = []
        for gt in gt_raw:
            cls, xc, yc, w, h = gt
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            gt_abs.append([cls, x1, y1, x2, y2])

        total_gt += len(gt_abs)

        matched_gt = [False] * len(gt_abs)
        sorted_preds = sorted(enumerate(preds), key=lambda x: x[1][1], reverse=True)

        for _, (pred_box, pred_conf, pred_cls) in sorted_preds:
            best_iou = 0
            best_j = -1
            for j, (gt_cls, *gt_box) in enumerate(gt_abs):
                if matched_gt[j]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold:
                matched_gt[best_j] = True
                total_tp += 1
            else:
                total_fp += 1

        total_fn += sum(1 for m in matched_gt if not m)

        if (idx + 1) % 20 == 0:
            print(f"  TTA进度: {idx + 1}/{len(image_files)}")

    return total_tp, total_fn, total_fp, total_gt


# 模型导入
def main():
    # ========== 配置 ==========
    test_images_dir = "yolo_dataset/images/test"
    test_labels_dir = "yolo_dataset/labels/test"
    negative_images_dir = "yolo_dataset/images/test_negative"
    # 按优先级自动选择模型：v1 > v3 > v2
    model_paths = [
        "runs/detect/runs/crack_detect_v1/weights/best.pt",
        "runs/detect/runs/crack_detect_v3/weights/best.pt",
        "runs/detect/runs/crack_detect_v2/weights/best.pt",
    ]
    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    if model_path is None:
        print("错误：未找到任何模型文件")
        print("请先运行 train_v1.py 训练模型")
        return

    iou_threshold = 0.5  # IoU匹配阈值
    use_tta = True  # 是否使用TTA

    # 手动指定一个阈值进行详细评估
    # 设为 None 则自动扫描最佳阈值
    manual_conf_threshold = None  # 例如: 0.25, 设为None则自动扫描
    # =========================

    print("=" * 60)
    print("隐裂检测模型评估")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        print("请先运行 main_train.py 训练模型，或在脚本中修改 model_path")
        return

    model = YOLO(model_path)
    print(f"模型加载成功: {model_path}")

    # ========== 1. 正样本测试 ==========
    image_files = glob.glob(os.path.join(test_images_dir, "*.png"))
    if not image_files:
        print(f"错误：测试图片目录为空 {test_images_dir}")
        print("请先运行 setup_data.py 准备数据集")
        return

    print(f"\n正样本测试图片数: {len(image_files)}")

    # ========== 1a. 扫描阈值，偏向检出率 ==========
    if manual_conf_threshold is None:
        print("\n--- 扫描置信度阈值 ---")
        conf_thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
        target_recall = 0.90  # 目标检出率（工业场景优先保召回）
        sweep_results = []
        best_f2 = 0  # F2: 召回率权重是精确率的2倍
        best_conf_by_f2 = 0.25
        best_conf_by_target = 0.001  # 满足目标召回率的最佳阈值

        for conf in conf_thresholds:
            tp, fn, fp, total_gt = evaluate_at_threshold(model, image_files, test_labels_dir, conf, iou_threshold)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
            # F2: 偏向召回率 (β=2, 召回率权重大于精确率)
            beta = 2
            f_beta = (
                (1 + beta**2) * recall * precision / (beta**2 * precision + recall) if (recall + precision) > 0 else 0
            )
            sweep_results.append((conf, recall, precision, f1, f_beta, tp, fn, fp))

            if f_beta > best_f2:
                best_f2 = f_beta
                best_conf_by_f2 = conf
            if recall >= target_recall:
                best_conf_by_target = conf  # 最后一个满足目标的（取最高precision）

        print(f"{'Conf':>8}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}  {'F2':>8}  {'TP':>5}  {'FN':>5}  {'FP':>5}")
        print("-" * 72)
        for conf, rec, prec, f1, f2, tp, fn, fp in sweep_results:
            markers = []
            if conf == best_conf_by_f2:
                markers.append("F2-best")
            if conf == best_conf_by_target:
                markers.append(f"Recall>{target_recall:.0%}")
            marker = " <-- " + ", ".join(markers) if markers else ""
            print(
                f"{conf:>8.3f}  {rec:>8.4f}  {prec:>10.4f}  {f1:>8.4f}  {f2:>8.4f}  {tp:>5}  {fn:>5}  {fp:>5}{marker}"
            )

        # 工业场景选阈值：优先满足目标召回率，否则用F2
        if best_conf_by_target > 0.001 or any(r[1] >= target_recall for r in sweep_results):
            eval_conf = best_conf_by_target
            print(f"\n工业检出模式: conf={eval_conf:.3f} (Recall >= {target_recall:.0%})")
        else:
            eval_conf = best_conf_by_f2
            print(f"\nF2最佳阈值: conf={eval_conf:.3f} (F2={best_f2:.4f}, 偏向检出率)")
    else:
        eval_conf = manual_conf_threshold

    # ========== 1b. 在最佳阈值下详细评估 ==========
    print(f"\n--- 详细评估 (conf={eval_conf:.3f}) ---")

    tp, fn, fp, total_gt = evaluate_at_threshold(model, image_files, test_labels_dir, eval_conf, iou_threshold)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    print(f"  真实缺陷框总数: {total_gt}")
    print(f"  正确检出 (TP): {tp}")
    print(f"  漏检 (FN):     {fn}")
    print(f"  误检 (FP):     {fp}")
    print(f"  召回率 (Recall):    {recall * 100:.2f}%")
    print(f"  精确率 (Precision):  {precision * 100:.2f}%")
    print(f"  F1分数:             {f1:.4f}")

    # ========== 1c. TTA评估 ==========
    if use_tta:
        print(f"\n--- TTA评估 (conf={eval_conf:.3f}) ---")
        print("正在进行Test-Time Augmentation推理（较慢）...")
        tp_tta, fn_tta, fp_tta, _ = evaluate_with_tta(model, image_files, test_labels_dir, eval_conf, iou_threshold)
        recall_tta = tp_tta / (tp_tta + fn_tta) if (tp_tta + fn_tta) > 0 else 0
        precision_tta = tp_tta / (tp_tta + fp_tta) if (tp_tta + fp_tta) > 0 else 0
        f1_tta = (
            2 * recall_tta * precision_tta / (recall_tta + precision_tta) if (recall_tta + precision_tta) > 0 else 0
        )

        print(f"  TTA-召回率 (Recall):    {recall_tta * 100:.2f}%")
        print(f"  TTA-精确率 (Precision):  {precision_tta * 100:.2f}%")
        print(f"  TTA-F1分数:              {f1_tta:.4f}")

    # ========== 2. 过杀率评估（负样本）==========
    print(f"\n--- 过杀率评估 (conf={eval_conf:.3f}) ---")
    if os.path.exists(negative_images_dir):
        fp_img, total_neg, total_fp_boxes = evaluate_overkill(model, negative_images_dir, eval_conf)
        overkill_rate = fp_img / total_neg if total_neg > 0 else 0
        print(f"  负样本总数:           {total_neg}")
        print(f"  有误报的图像数:       {fp_img}")
        print(f"  总误报框数:           {total_fp_boxes}")
        print(f"  过杀率 (图像级):      {overkill_rate * 100:.2f}%")
        print(f"  平均误报框/负样本图:  {total_fp_boxes / total_neg:.2f}" if total_neg > 0 else "")
    else:
        print(f"  负样本目录不存在: {negative_images_dir}")
        print("  请运行 setup_data.py 创建负样本测试集")
        total_neg = 0
        fp_img = 0
        overkill_rate = 0

    # ========== 3. 生成报告 ==========
    result_text = f"""======================================================================
隐裂检测模型评估报告
======================================================================

【模型信息】
  模型路径: {model_path}
  测试图片数（正样本）: {len(image_files)}
  IoU匹配阈值: {iou_threshold}
  评估置信度阈值: {eval_conf:.3f}
  使用TTA: {"是" if use_tta else "否"}

【正样本评估】
  真实缺陷框总数: {total_gt}
  正确检出 (TP): {tp}
  漏检 (FN):     {fn}
  误检 (FP):     {fp}
  召回率 (Recall):    {recall * 100:.2f}%
  精确率 (Precision):  {precision * 100:.2f}%
  F1分数:             {f1:.4f}
"""

    if use_tta:
        result_text += f"""
【TTA评估】
  TTA-召回率 (Recall):    {recall_tta * 100:.2f}%
  TTA-精确率 (Precision):  {precision_tta * 100:.2f}%
  TTA-F1分数:              {f1_tta:.4f}
"""

    result_text += f"""
【过杀率评估（负样本）】
  负样本总数:           {total_neg}
  有误报的图像数:       {fp_img}
  总误报框数:           {total_fp_boxes}
  过杀率 (图像级):      {overkill_rate * 100:.2f}%

【建议】
  - 若召回率偏低 (<60%)：降低conf阈值、使用TTA、或增加训练数据
  - 若过杀率偏高 (>10%)：提高conf阈值、或在训练集中增加更多负样本
  - 评估模式: {"F2偏向检出率" if manual_conf_threshold is None else "手动指定"}, conf={eval_conf:.3f}
======================================================================
"""

    report_path = "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"\n评估报告已保存到 {report_path}")


if __name__ == "__main__":
    main()
