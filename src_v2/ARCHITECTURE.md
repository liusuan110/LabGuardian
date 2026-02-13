# LabGuardian v2 — 模块化架构
# 新架构目录结构:
#
# src_v2/
# ├── main.py                 # 精简入口 (~50行)
# ├── config.py               # 全局配置 (路径/参数/API, 跨平台)
# ├── vision/
# │   ├── __init__.py
# │   ├── detector.py          # YOLO 推理封装
# │   ├── calibrator.py        # 面包板校准 (从 calibration.py 迁移)
# │   └── stabilizer.py        # 多帧稳定化 (新增)
# ├── logic/
# │   ├── __init__.py
# │   ├── circuit.py           # CircuitAnalyzer + CircuitComponent
# │   ├── validator.py         # CircuitValidator (模板对比)
# │   └── schematic.py         # 原理图可视化 (从 schematic_viz.py 迁移)
# ├── ai/
# │   ├── __init__.py
# │   └── llm_engine.py        # LLM 推理引擎 (云端/本地统一接口)
# ├── gui/
# │   ├── __init__.py
# │   ├── app.py               # 主窗口类
# │   ├── video_panel.py       # 视频显示组件
# │   └── chat_panel.py        # AI聊天组件
# └── tools/
#     ├── annotate_helper.py   # OBB标注工具
#     ├── train_obb.py         # 训练脚本
#     └── fix_dataset.py       # 数据修正工具
