"""
数据处理脚本
1. 将原始训练集按 80/20 划分为 train/val
2. 将负样本按 70/30 加入 train/val（不加入测试集）
3. 测试集仅包含正样本（用于计算 Recall/mAP）
4. 负样本单独保存用于过杀率评估
"""
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

random.seed(42)


def convert_voc_to_yolo(xml_path, img_width, img_height):
    """VOC格式XML -> YOLO格式标注"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'QX':
            continue
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines



def process_positive_samples(src_dir, img_dst, lbl_dst):
    """处理正样本（有缺陷标注的图片）"""
    src_dir = Path(src_dir)
    png_files = list(src_dir.glob('*.png'))
    count = 0
    for png_file in png_files:
        xml_file = src_dir / (png_file.stem + '.xml')
        if not xml_file.exists():
            # 没有标签的图片 — 视为负样本，创建空标签
            shutil.copy(png_file, img_dst / png_file.name)
            (lbl_dst / (png_file.stem + '.txt')).touch()
            count += 1
            continue

        # 读取XML获取图片尺寸
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 转换标注
        yolo_lines = convert_voc_to_yolo(xml_file, img_width, img_height)
        if not yolo_lines:
            continue  # 跳过没有任何QX标注的图片

        # 拷贝并转换图片为RGB
        shutil.copy(png_file, img_dst / png_file.name)
        lbl_path = lbl_dst / (png_file.stem + '.txt')
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        count += 1
    return count


def process_negative_samples(src_dir, img_dst, lbl_dst):
    """处理负样本：只复制图片，创建空标注文件"""
    src_dir = Path(src_dir)
    png_files = list(src_dir.glob('*.png'))
    count = 0
    for png_file in png_files:
        shutil.copy(png_file, img_dst / png_file.name)
        (lbl_dst / (png_file.stem + '.txt')).touch()
        count += 1
    return count


def main():
    dataset_root = Path('隐裂数据集_20260113')
    yolo_root = Path('yolo_dataset')

    # 清理旧数据
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
        print("已清理旧的 yolo_dataset")

    # 创建目录结构
    for split in ['train', 'val', 'test', 'test_negative']:
        (yolo_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # ========== 1. 处理正样本训练集，按 80/20 划分 train/val ==========
    train_src = dataset_root / '训练集'
    all_train_pngs = [p for p in train_src.glob('*.png') if p.with_suffix('.xml').exists()]
    random.shuffle(all_train_pngs)
    val_count = int(len(all_train_pngs) * 0.2)
    val_pngs = set(p.name for p in all_train_pngs[:val_count])
    train_pngs = set(p.name for p in all_train_pngs[val_count:])

    print(f"训练集正样本: {len(train_pngs)} 张 (用于 train)")
    print(f"训练集正样本: {len(val_pngs)} 张 (用于 val)")

    # 处理 train 部分
    for png in train_src.glob('*.png'):
        if png.name in train_pngs:
            xml_file = png.with_suffix('.xml')
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            img_w, img_h = int(size.find('width').text), int(size.find('height').text)
            yolo_lines = convert_voc_to_yolo(xml_file, img_w, img_h)
            if yolo_lines:
                shutil.copy(png, yolo_root / 'images' / 'train' / png.name)
                with open(yolo_root / 'labels' / 'train' / (png.stem + '.txt'), 'w') as f:
                    f.write('\n'.join(yolo_lines))

    # 处理 val 部分
    for png in train_src.glob('*.png'):
        if png.name in val_pngs:
            xml_file = png.with_suffix('.xml')
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            img_w, img_h = int(size.find('width').text), int(size.find('height').text)
            yolo_lines = convert_voc_to_yolo(xml_file, img_w, img_h)
            if yolo_lines:
                shutil.copy(png, yolo_root / 'images' / 'val' / png.name)
                with open(yolo_root / 'labels' / 'val' / (png.stem + '.txt'), 'w') as f:
                    f.write('\n'.join(yolo_lines))

    # ========== 2. 负样本按 70/30 加入 train/val ==========
    neg_src = dataset_root / '负样本'
    neg_pngs = list(neg_src.glob('*.png'))
    random.shuffle(neg_pngs)
    neg_train_count = int(len(neg_pngs) * 0.7)
    neg_train_pngs = neg_pngs[:neg_train_count]
    neg_val_pngs = neg_pngs[neg_train_count:]

    print(f"负样本: {len(neg_train_pngs)} 张加入 train, {len(neg_val_pngs)} 张加入 val")

    for png in neg_train_pngs:
        shutil.copy(png, yolo_root / 'images' / 'train' / png.name)
        (yolo_root / 'labels' / 'train' / (png.stem + '.txt')).touch()

    for png in neg_val_pngs:
        shutil.copy(png, yolo_root / 'images' / 'val' / png.name)
        (yolo_root / 'labels' / 'val' / (png.stem + '.txt')).touch()

    # ========== 3. 测试集（仅正样本）==========
    test_src = dataset_root / '测试集'
    test_count = 0
    for png in test_src.glob('*.png'):
        xml_file = png.with_suffix('.xml')
        if xml_file.exists():
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            img_w, img_h = int(size.find('width').text), int(size.find('height').text)
            yolo_lines = convert_voc_to_yolo(xml_file, img_w, img_h)
            if yolo_lines:  # 只放有标注的正样本
                shutil.copy(png, yolo_root / 'images' / 'test' / png.name)
                with open(yolo_root / 'labels' / 'test' / (png.stem + '.txt'), 'w') as f:
                    f.write('\n'.join(yolo_lines))
                test_count += 1

    print(f"测试集正样本: {test_count} 张")

    # ========== 4. 负样本单独保存用于过杀率评估 ==========
    # 复用与 test 相同的负样本集合（方便评估脚本单独加载）
    neg_test_dir = yolo_root / 'images' / 'test_negative'
    neg_test_dir.mkdir(parents=True, exist_ok=True)
    for png in neg_src.glob('*.png'):
        shutil.copy(png, neg_test_dir / png.name)

    print(f"过杀率测试集（负样本）: {len(list(neg_src.glob('*.png')))} 张")

    # ========== 5. 创建 data.yaml ==========
    yaml_content = f"""path: {yolo_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

# 单类别：隐裂(QX)
nc: 1
names: ['crack']

# 数据集统计
# train: 含正样本+负样本
# val: 含正样本+负样本
# test: 仅正样本（用于mAP/Recall评估）
# test_negative: 负样本（用于过杀率评估）
"""
    with open(yolo_root / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # 打印统计
    print("\n========== 数据集统计 ==========")
    for split in ['train', 'val', 'test']:
        imgs = list((yolo_root / 'images' / split).glob('*.png'))
        lbls = list((yolo_root / 'labels' / split).glob('*.txt'))
        # 统计有多少是正样本（标签非空）
        pos_count = sum(1 for l in lbls if l.stat().st_size > 0)
        neg_count = len(lbls) - pos_count
        print(f"  {split}: {len(imgs)} 张图片 (正样本: {pos_count}, 负样本: {neg_count})")

    neg_imgs = list((yolo_root / 'images' / 'test_negative').glob('*.png'))
    print(f"  test_negative: {len(neg_imgs)} 张负样本（过杀率评估用）")

    print("\n数据准备完成！请运行 main_train.py 开始训练。")


if __name__ == '__main__':
    main()
