All project code is intended solely for personal learning purposes and not for commercial use. Feedback is welcome; please contact me if there is any copyright infringement so that I can modify or remove the content. The dataset is sourced from an engineering interview test at the Dongguan AI Research Institute.
setup_data.py为数据处理脚本包含数据格式从VOC到YOLO的转换、数据集结构的划分等;
train_v1.py为默认640x分辨率的训练脚本,碍于数据集规模,该训练策略的综合性能反而更好;
train_v3.py为第三个版本的训练策略脚本主要包括分辨率提高到960x、旋转5°、剪切2°、erasing减小到0.2等，出现了过拟合现象，根本原因在于数据集太少。
<img width="1200" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/1bec5fb8-1492-42ab-95e2-1d83fa4f9bc0" />
quantize.py是量化脚本，尝试了两种策略的量化，但实践证明，YOLOv8s本身就是轻量化模型，在边缘设备上FP32已达最优，无需量化。
<img width="643" height="221" alt="屏幕截图 2026-07-14 161602" src="https://github.com/user-attachments/assets/8a0ac59e-7040-475c-ba0e-8c1521a8e2e1" />
prune.py是剪枝脚本，尝试了两种剪枝策略。
<img width="650" height="215" alt="屏幕截图 2026-07-14 164910" src="https://github.com/user-attachments/assets/b4f45d90-6de3-4a42-8ce3-ee603890f573" />

