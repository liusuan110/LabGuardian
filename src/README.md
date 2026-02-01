# LabGuardian / src 目录说明

本文档说明 `src/` 下各文件/目录的用途，方便后续维护与清理。

## 入口与核心逻辑

- `main.py`
  - 项目主入口（GUI + 摄像头/图片输入 + YOLO 检测 + LLM 问答）。
  - 负责：加载视觉模型、构建电路上下文、调用 LLM（云端或本地）、展示网表、触发校准与原理图绘制。

- `calibration.py`
  - 相机/图片的面包板校准与坐标映射：透视变换、孔洞检测、像素坐标 → 逻辑孔位(row/col)。
  - 对外主要提供全局单例：`board_map`。

- `circuit_logic.py`
  - 电路拓扑建模与验证：
    - `CircuitComponent`：元件结构。
    - `CircuitAnalyzer`：把元件/导线转成电气网络图，并生成网表描述。
    - `CircuitValidator`（以及模块级 `validator`）：保存/加载“标准电路模板”，对比当前电路并输出差异。

- `schematic_viz.py`
  - 原理图可视化：基于 `CircuitAnalyzer.components` 用 `schemdraw` 绘制简化原理图（演示用途）。

## 数据集/标注/训练辅助脚本（不影响主程序运行）

- `annotate_helper.py`
  - OBB 标注小工具：用鼠标依次点击 4 个角点生成 OBB 标签（用于制作/补标演示数据）。

- `train_obb_demo.py`
  - YOLOv8-OBB 训练脚本：基于 `OneShot_Demo_Dataset/data.yaml` 做演示用的小数据训练，产出可被 `main.py` 自动加载的 `best.pt`。

- `fix_dataset.py`
  - 数据修正脚本：把某些标签从 `xywh` 格式转换为 OBB 的 4 点格式（`x1 y1 ... x4 y4`）。
  - 注意：当前脚本里使用了硬编码绝对路径，换机器需要改成相对路径或命令行参数。

- `replicate_labels.py`
  - 标签复制脚本：把某个 label 文件内容复制到同目录其他 label（快速生成/对齐标注）。
  - 注意：当前脚本里使用了硬编码绝对路径。

## 其他文本

- `new_labels.txt`
  - 标签/类别相关的临时文本（用于辅助标注或类别映射）。

## 模型与大文件（通常不建议直接进 Git）

- `yolo26n.pt`, `yolov8n.pt`
  - YOLO 权重文件（体积大，通常应使用 Git LFS 或 Release/网盘分发）。

- `openvino_tinyllama_model/`
  - OpenVINO 导出的 LLM 模型与 tokenizer 文件（可能较大）。
  - `model_cache/` 为运行时缓存（不建议提交）。

- `yolov8n_openvino_model/`
  - OpenVINO 导出的 yolov8n 模型文件。

## 数据集目录（一般只保留配置/标注，小心仓库体积）

- `datasets/`
  - 第三方数据集/样例（如 COCO）；通常不建议把图片完整提交到代码仓库。

- `Electronic-Components-1/`, `OneShot_Demo_Dataset/`
  - 训练/演示数据集目录（含 `data.yaml`、图片与标签）。

## 运行结果

- `runs/`
  - Ultralytics 的训练/推理输出目录（权重、日志、可视化结果等），一般不建议提交到代码仓库。
