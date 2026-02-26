# LabGuardian 数据集目录结构说明
# ==================================
#
# 照片拍好后，按以下结构组织：
#
#   dataset/
#   ├── train/          ← 训练集 (80% 的图片)
#   │   ├── images/     ← 放 .jpg 图片
#   │   └── labels/     ← 放 .txt 标签 (文件名与图片一一对应)
#   ├── val/            ← 验证集 (20% 的图片)
#   │   ├── images/
#   │   └── labels/
#   └── data.yaml       ← 自动生成，不需要手动创建
#
# 标签格式 (OBB 四点):
#   每行: class_id x1 y1 x2 y2 x3 y3 x4 y4
#   - class_id: 0=CAPACITOR, 1=DIODE, 2=LED, 3=RESISTOR, 4=Push_Button, 5=Wire
#   - x1..y4: 旋转矩形四角坐标，归一化到 0~1
#
# 标注工具推荐:
#   1. Roboflow (roboflow.com) — 在线标注, 支持 OBB, 可导出 YOLO 格式
#   2. Label Studio — 开源, 支持旋转框
#   3. 项目自带工具: python -m tools.annotate_helper
#
# 数据集划分建议:
#   - 总量: 至少 200 张 (越多越好)
#   - train:val = 8:2
#   - 确保每个类别至少 30 个样本
#   - 拍摄多种光线、角度、面包板颜色的照片
