# LabGuardian 项目现状分析与开发规划（v2.0 修订版）

> **项目**：LabGuardian — 基于边缘AI的智能理工科实验助教系统  
> **竞赛**：2026年（第十三届）英特尔杯大学生电子设计竞赛 · 嵌入式AI专题赛  
> **竞赛平台**：Intel® Core™ Ultra 5 225U · DK-2500 开发套件  
> **初版日期**：2026年2月10日  
> **本次修订**：2026年2月15日（基于实际代码审计）  

---

## 一、竞赛关键时间节点

| 日期 | 事件 | 状态 |
|:---|:---|:---|
| 2026.01.20 | 报名截止 | ✅ 已完成 |
| **2026.03.25** | **提交设计方案（报告+PPT）** | 🔴 **距今约38天** |
| 2026.03.26-31 | 初选评审 | — |
| **2026.04 初** | **平台发放（DK-2500寄出）** | ⏳ 等待 |
| 2026.05.31 | 队员锁定 | — |
| **2026.06.30** | **提交最终作品（实物+中英文报告）** | — |
| 2026.07.10-15 | 分区评审 | — |
| 2026.07.20-25 | 全国评审（上海） | — |

---

## 二、当前项目文件结构（src_v2 模块化架构）

> **重大变更**：自 v1.0 规划以来，代码已从 `src/main.py` 单文件巨石架构完全重构为 `src_v2/` 模块化架构。总代码量 **~7,200 行**，**25+ Python 源文件**。

```
inter/LabGuardian/src_v2/
├── config.py                 (422行)  8个dataclass配置 + 环境变量覆盖
├── build_knowledge_base.py   (160行)  RAG知识库CLI管理工具
├── ARCHITECTURE.md                    架构说明 (需更新)
│
├── ai/                        ─── 认知层 ───
│   ├── llm_engine.py         (557行)  Cloud + Local + Rule 三级LLM引擎
│   ├── rag_engine.py         (530行)  ChromaDB + text2vec 检索增强
│   └── ocr_engine.py         (611行)  PaddleOCR + EasyOCR 元件识读
│
├── vision/                    ─── 感知层 ───
│   ├── detector.py           (~190行) YOLOv8 HBB+OBB 双模检测
│   ├── stabilizer.py         (~120行) 多帧滑窗投票稳定器
│   └── calibrator.py         (626行)  四级孔洞检测 + Grid RANSAC
│
├── logic/                     ─── 推理层 ───
│   ├── circuit.py            (488行)  NetworkX图论网表 + 电源网络
│   ├── validator.py          (359行)  VF2++图同构验证 + 3级诊断
│   ├── polarity.py           (320行)  二极管/三极管/电容极性推理
│   └── schematic.py          (~95行)  schemdraw原理图绘制
│
├── gui_qt/                    ─── 界面层 (PySide6) ───
│   ├── main_window.py        (848行)  5页无边框主窗口 + 视频管线
│   ├── workers.py            (221行)  视频/模型加载/LLM工作线程
│   ├── dashboard.py          (238行)  仪表盘 (状态卡 + 模块指示)
│   ├── chat_panel.py         (250行)  AI聊天面板 (气泡 + 快捷按钮)
│   ├── circuit_page.py       (~120行) 电路验证页 (7个操作按钮)
│   ├── video_panel.py        (200行)  视频面板 + 置信度滑块
│   ├── settings_page.py      (203行)  配置面板 (摄像头/YOLO/LLM)
│   ├── sidebar.py            (181行)  可折叠导航栏
│   ├── styles.py             (339行)  PyDracula暗色主题QSS
│   └── resources.py          (~30行)  图标常量
│
├── knowledge_base/builtin/    ─── RAG知识文档 ───
│   ├── transistor_8050.md
│   ├── transistor_8550.md
│   ├── op_amp_basics.md
│   ├── ne555_timer.md
│   ├── chip_pinout_guide.md
│   └── analog_lab_guide.md
│
├── models/                    ─── 模型资产 ───
│   ├── yolov8n.pt                     通用预训练 (回退)
│   ├── yolov8n-obb.pt                 OBB预训练
│   ├── yolo26n.pt                     疑似自训练
│   ├── yolov8n_openvino/              YOLO OpenVINO IR
│   └── tinyllama_openvino/            TinyLlama INT4 (中文差)
│
├── tools/                     ─── 辅助工具 ───
│   ├── annotate_helper.py             OBB标注辅助
│   ├── train_obb.py                   YOLO OBB训练
│   ├── fix_dataset.py                 数据集修正
│   └── replicate_labels.py            标签复制
│
├── .env.example                       环境变量模板
└── .gitignore                         排除规则
```

---

## 三、v1.0 问题清单：解决进度

### ~~问题 1：YOLO模型泛化能力严重不足（🔴 致命）~~

**状态：⚠️ 部分缓解，核心问题仍存**

- ✅ 检测框架已支持 HBB + OBB 双模式自动切换
- ✅ `DetectionStabilizer` 多帧投票（5帧窗口, IoU匹配, ≥3票确认）大幅降低误检
- ❌ **训练数据集仍为预训练权重**，未采集领域数据
- ❌ 当前类别仅 6 个 (CAPACITOR/DIODE/LED/RESISTOR/Push_Button/Wire)，**缺少 IC/TRANSISTOR/NPN** 等芯片类别
- ❌ OCR 功能虽已完整，但需 IC 类别检测触发才能发挥作用

**结论**：这仍然是 **当前最大瓶颈**，直接决定初选方案的可信度。

---

### ~~问题 2：面包板孔洞识别鲁棒性不足（🟡 重要）~~

**状态：✅ 已解决**

- ✅ `BreadboardCalibrator` 重写为 626 行的四级检测管线：
  1. `cv2.findCirclesGrid()` → 强结构先验
  2. Blob Ensemble (多参数扫描 + NMS) → 宽容度检测
  3. `cv2.HoughCircles()` → 补充检测
  4. Contour Analysis → 最后兜底
- ✅ CLAHE 光照归一化预处理
- ✅ Grid RANSAC 拟合 + 双线性插值推断缺失孔位
- ✅ 手动辅助模式 (4角 + 中心点点选校准)

---

### ~~问题 3：电路逻辑推理能力有限（🟡 重要）~~

**状态：✅ 已解决**

- ✅ `DetectionStabilizer` 多帧投票消除抖动
- ✅ `CircuitAnalyzer` 重写：NetworkX 双节点图（Net节点 + Comp节点），极性感知，电源网络自动识别，自然语言网表导出
- ✅ `CircuitValidator` 重写：VF2++ 图同构匹配 + 3级验证（拓扑→极性→启发式），JSON 模板持久化
- ✅ `PolarityResolver` 新增：二极管/LED OBB方向推理，三极管 TO-92 引脚约定

---

### ~~问题 4：代码架构不利于移植（🟡 重要）~~

**状态：✅ 已解决**

- ✅ 941行单文件 → `src_v2/` 模块化（ai/ + vision/ + logic/ + gui_qt/）
- ✅ Tkinter → PySide6 (5页无边框主窗口, PyDracula 暗色主题)
- ✅ API Key → 环境变量 (`.env` / `.env.example`)
- ✅ 全部 `pathlib.Path` + 相对路径
- ✅ 8 个 `@dataclass` 配置类，支持环境变量覆盖
- ✅ 摄像头/模型路径/设备全部参数化

---

### ~~问题 5：LLM 本地化方案风险（🟡 重要）~~

**状态：✅ 已解决**

- ✅ 三级降级策略：Cloud DeepSeek → Local OpenVINO GenAI → Rule-based (9套元件知识)
- ✅ RAG 检索增强：ChromaDB + `text2vec-base-chinese`，6 篇知识文档 → 49 个文本块
- ✅ OCR 芯片识读：PaddleOCR (主) + EasyOCR (备)，13 种正则 + 常见错字修正
- ✅ 断网场景：Local LLM + RAG + Rule-based 三重保障，零网络依赖
- ⚠️ 尚未下载推荐的 `qwen2.5_1.5b_ov` / `minicpm_1b_ov`（等 DK-2500 到手后在 NPU 上评估）

---

### ~~问题 6：AR功能已放弃，需要新差异化亮点（🟢 策略）~~

**状态：✅ 方向已明确**

差异化亮点重新定义为：
1. **OCR + RAG 自动识读**：拍到芯片 → OCR识别型号 → RAG检索数据手册 → 自动展示引脚图和用法
2. **三级 LLM 智能问答**：对着电路"问一句答一句"，利用网表上下文生成精准回答
3. **一键电路纠错**：VF2++ 图同构对比 + 极性诊断 + Ghost Wire 叠加标注
4. **Intel 异构计算全栈**：YOLO→iGPU, LLM→NPU, OCR→CPU, 多设备并行

---

## 四、当前剩余问题

### 问题 A：YOLO 领域数据缺失（🔴 致命 — 最高优先级）

- 当前仅使用预训练 `yolov8n.pt`，未针对面包板电路场景训练
- 类别不全，缺少 IC/TRANSISTOR 等芯片类别
- 直接影响 OCR 触发链路（需先检测到 IC 才能 OCR）
- **初选报告必须包含模型训练数据和 mAP 指标**

### 问题 B：电容极性检测为 STUB（🟡 中等）

- `polarity.py` 中电解电容标记为 `UNKNOWN`，注释说需颜色分析
- 不影响初选，但决赛需要补全

### 问题 C：原理图模块基础版（🟡 中等）

- `schematic.py` 仅 95 行，不支持三极管/IC 符号，布线算法简单
- 可在决赛前完善

### 问题 D：校准弹窗未迁移 Qt（🟢 低）

- `main_window.py` 中校准仍使用 `cv2.namedWindow` 弹窗，未迁移到 PySide6 QDialog
- 体验略差但功能不受影响

### 问题 E：Settings 页面无持久化（🟢 低）

- 控件 UI 已搭建，但修改后未同步回 config 实例 / 写入磁盘
- 可在决赛调优阶段补全

### 问题 F：ARCHITECTURE.md 过时（🟢 低）

- 仍反映 v1 旧结构，缺少 rag_engine, ocr_engine, polarity, gui_qt/ 等

---

## 五、已放弃的功能

| 功能 | 放弃原因 | 替代方案 |
|:---|:---|:---|
| AR虚拟现实叠加 | DK-2500 算力限制 + 实现复杂度高 | 屏幕端"Ghost Wire 纠错可视化标注" |
| 3D姿态解算 | 依赖AR，单摄像头效果有限 | 透视变换校正到正俯视即可 |

---

## 六、修订版开发规划

### 第一阶段：初选冲刺（现在 → 3月25日）

> **核心目标**：通过初选评审，拿到入围资格  
> **已完成工作量**：~7,200 行代码，框架已完整。剩余工作集中在 **数据/模型** 和 **文档/演示**。

#### Sprint A: 数据与模型（立即开始 → 3.05）🔴 最高优先级

| 任务 | 优先级 | 详情 | 工作量 |
|:---|:---|:---|:---|
| 数据采集 | 🔴 P0 | 用手机在不同环境拍摄 200-300 张面包板电路照片（不同面包板型号、元件组合、光照、角度）。务必包含 IC 芯片和三极管 | 3-5天 |
| OBB数据标注 | 🔴 P0 | 使用 Roboflow 或 `tools/annotate_helper.py`；类别扩展为: Resistor, LED, Capacitor, Wire, **IC, Transistor**, Button, Diode | 5-7天 |
| 合并公开数据集 | 🟡 P1 | 整合 `arduino-1/`, `Arduino-pin-detection-3/`, `Electronic-components-1/` 中可用的数据 | 2天 |
| 数据增强配置 | 🟡 P1 | Ultralytics 内置增强: Mosaic, MixUp, 旋转(-30°~30°), 色彩抖动, Copy-Paste | 1天 |
| 模型重新训练 | 🔴 P0 | `yolov8s-obb.pt` (升级到 small 版), epochs=300, imgsz=960, batch=8 | 2-4小时 |
| 模型评估 | 🔴 P0 | 独立测试集评估 mAP@0.5，目标 ≥ 80%；更新 `config.py` 类别列表 | 1天 |

#### Sprint B: 初选材料（3.05 → 3.25）🔴 关键路径

| 任务 | 优先级 | 详情 | 工作量 |
|:---|:---|:---|:---|
| 重写设计报告 | 🔴 P0 | 基于 `Design_Doc.md` 重构：突出四层架构 (感知→稳定→推理→认知)、Intel 异构计算 (iGPU/NPU/CPU)、三级 LLM + RAG + OCR 创新点、模型训练指标 | 5天 |
| 制作PPT | 🔴 P0 | 15-20页：系统架构图 / Demo截图+GIF / KPI表格 / Roadmap / 差异化亮点 | 3天 |
| 录制Demo视频 | 🟡 P1 | 全链路演示：摄像头实时检测 → OCR芯片识读 → 网表生成 → 电路纠错 → AI问答 | 1天 |
| KPI数据整理 | 🟡 P1 | `eval/compute_kpis.py` 填入真实测试数据 (mAP, 延迟, FPS) | 1天 |
| ARCHITECTURE.md 更新 | 🟢 P2 | 同步为 v2 结构 | 0.5天 |

#### Sprint C: 小修补（穿插进行）

| 任务 | 优先级 | 说明 |
|:---|:---|:---|
| config.py 类别扩充 | 🟡 P1 | 训练完成后添加 IC/Transistor 到 `COMPONENT_CLASSES` |
| 校准弹窗 Qt 化 | 🟢 P2 | `cv2.namedWindow` → PySide6 QDialog |
| Settings 持久化 | 🟢 P2 | 控件值同步回 config 并写磁盘 |

---

### 第二阶段：平台移植（4月初 → 5月中）

> **核心目标**：在 DK-2500 上跑通全链路  
> **前置条件**：收到 DK-2500 硬件

#### Sprint D: 环境搭建与模型部署（4月第1-2周）

| 任务 | 详情 |
|:---|:---|
| DK-2500 开箱测试 | 确认 Ubuntu 版本、iGPU/NPU 驱动、OpenVINO Runtime |
| OpenVINO 环境 | 安装 OpenVINO 2024.x + GPU/NPU 插件 |
| YOLO OpenVINO 导出 | `model.export(format="openvino", imgsz=960, half=True)` → iGPU |
| iGPU 推理验证 | 目标: ≥ 15 FPS @ 960p |
| 摄像头适配 | USB摄像头 Linux (V4L2) 驱动与参数调优 |
| PySide6 验证 | 确认 GUI 在 Ubuntu + Wayland/X11 下正常运行 |

#### Sprint E: LLM NPU 部署（4月第3周 → 5月初）

| 任务 | 详情 |
|:---|:---|
| LLM 模型选型 | NPU 上评估: Qwen2.5-1.5B (INT4) vs MiniCPM-1B (INT4) vs Phi-3-mini (INT4) |
| NPU 推理验证 | `openvino_genai` 库, 测试 TTFT < 2s, 吞吐 > 10 tokens/s |
| RAG 嵌入模型 | 确认 `text2vec-base-chinese` 在 DK-2500 CPU 上的延迟 |
| OCR 验证 | PaddleOCR 在 DK-2500 上的兼容性和延迟 |

> **注意**：代码重构 (Sprint 5 原计划) **已提前完成**，无需在此阶段重做。

---

### 第三阶段：功能打磨与报告（5月中 → 6月30日）

> **核心目标**：拿出有竞争力的最终作品

#### Sprint F: 功能完善与差异化（5月中 → 6月初）

| 功能 | 状态 | 描述 | 技术要点 |
|:---|:---|:---|:---|
| **一键纠错** | ✅ 框架完成 | 拍照 → VF2++ 对比 → 高亮差异 | 完善 Ghost Wire 标注样式 |
| **OCR + RAG 自动识读** | ✅ 框架完成 | 芯片 OCR → RAG 检索 → 自动展示 | 扩充知识文档、优化检索精度 |
| **安全预检** | 🔴 待开发 | LED无限流电阻告警 | 拓扑规则推理引擎 |
| **电容极性检测** | ⚠️ 待完善 | 电解电容正负极辨识 | 颜色分析 / 标记检测 |
| **原理图增强** | ⚠️ 待完善 | 三极管/IC 符号支持 | schemdraw 扩展 |
| **语音交互** | 🔵 可选 | Whisper-small OpenVINO | NPU 推理, "拍一下问一句" |

#### Sprint G: 最终报告与调优（6月初 → 6月30日）

| 任务 | 详情 |
|:---|:---|
| 系统性能调优 | Pipeline 并行化 (视觉线程 / LLM 线程 / OCR 线程已分离，优化调度) |
| 中文设计报告 | ~15000字, 原创性声明、中英摘要、四层架构方案、Intel 异构计算方案、测试数据、源码附录 |
| 英文设计报告 | ≤ 6页, Abstract, System Design, Test Results |
| 作品简介 | 竞赛组委会模板 |
| 压力测试 | 连续 2 小时运行、切换不同电路、断网环境、不同光照 |

---

## 七、完成度汇总表

| 模块 | 文件 | 行数 | 完成度 | 备注 |
|:---|:---|:---|:---|:---|
| 配置系统 | `config.py` | 422 | ✅ 100% | 8个dataclass, 环境变量覆盖 |
| LLM 引擎 | `ai/llm_engine.py` | 557 | ✅ 100% | Cloud+Local+Rule 三级 + RAG |
| RAG 引擎 | `ai/rag_engine.py` | 530 | ✅ 100% | ChromaDB 增量索引, 49个文本块 |
| OCR 引擎 | `ai/ocr_engine.py` | 611 | ✅ 100% | PaddleOCR+EasyOCR, 13种正则 |
| YOLO 检测 | `vision/detector.py` | ~190 | ✅ 100% | HBB+OBB 自动切换 |
| 检测稳定器 | `vision/stabilizer.py` | ~120 | ✅ 100% | 5帧滑窗, ≥3票确认 |
| 面包板校准 | `vision/calibrator.py` | 626 | ✅ 100% | 四级检测 + Grid RANSAC |
| 电路分析 | `logic/circuit.py` | 488 | ✅ 100% | NetworkX 双节点图 + 网表 |
| 电路验证 | `logic/validator.py` | 359 | ✅ 100% | VF2++ 同构 + 3级诊断 |
| 极性推理 | `logic/polarity.py` | 320 | ⚠️ 85% | **电容极性=STUB** |
| 原理图 | `logic/schematic.py` | ~95 | ⚠️ 70% | 无三极管/IC, 布线简单 |
| 主窗口 | `gui_qt/main_window.py` | 848 | ⚠️ 95% | 校准弹窗仍用 OpenCV |
| 工作线程 | `gui_qt/workers.py` | 221 | ✅ 100% | Video/Model/LLM 三线程 |
| 仪表盘 | `gui_qt/dashboard.py` | 238 | ✅ 100% | 4状态卡 + 模块指示 |
| 聊天面板 | `gui_qt/chat_panel.py` | 250 | ✅ 100% | 气泡 + 快捷按钮 |
| 电路页 | `gui_qt/circuit_page.py` | ~120 | ✅ 100% | 7按钮, Signal驱动 |
| 视频面板 | `gui_qt/video_panel.py` | 200 | ✅ 100% | FPS叠加 + 置信度滑块 |
| 设置页 | `gui_qt/settings_page.py` | 203 | ⚠️ 90% | **apply/save 未实现** |
| 主题样式 | `gui_qt/styles.py` | 339 | ✅ 100% | PyDracula QSS |
| 知识库 | `knowledge_base/builtin/` | 6文档 | ✅ 100% | 49个文本块已索引 |
| RAG CLI | `build_knowledge_base.py` | 160 | ✅ 100% | rebuild/query/list/stats |

**代码总完成度：~95%（功能框架层面）**

---

## 八、开发环境

### 当前（无DK-2500）→ Windows + venv

```
Windows 11
├── Python 3.12 venv (D:\desktop\inter\.venv)
├── PySide6 GUI
├── PaddleOCR (paddlepaddle 3.2.2) + EasyOCR
├── ultralytics / OpenVINO
├── ChromaDB + text2vec-base-chinese
├── VS Code (直接开发)
└── GPU训练: 本机 CUDA 或 Colab
```

### 4月后（有DK-2500）→ Ubuntu

```
DK-2500 (Ubuntu 22.04)
├── OpenVINO 2024.x
├── YOLO (OpenVINO format) → iGPU
├── LLM (INT4) → NPU
├── OCR (PaddleOCR) → CPU
├── PySide6 GUI
├── USB Camera → V4L2
└── 完整 LabGuardian 系统
```

---

## 九、风险与预案（修订版）

| 风险 | 概率 | 影响 | 预案 |
|:---|:---|:---|:---|
| 初选未入围 | 中 | 致命 | 全力打磨报告+PPT；Demo视频展示完整链路；强调已有 7200 行可运行代码 |
| 数据不足导致 mAP 低 | 中 | 高 | 合并公开数据集 (arduino-1 等)；数据增强 Mosaic/MixUp；降级为 HBB 模式 |
| NPU 跑不动目标 LLM | 低 | 中 | **三级降级已内置**：Cloud → Local(更小模型) → Rule-based |
| DK-2500 iGPU 帧率不足 | 中 | 高 | 降低分辨率 960→640；用 YOLO-nano 替代 small |
| PaddleOCR 在 Ubuntu 兼容问题 | 低 | 低 | EasyOCR 备用后端自动切换 |
| 面包板校准现场失败 | 低 | 中 | 四级检测管线 + 手动辅助模式已实现 |
| 评审现场断网 | 高 | 低 | **已解决**：Local LLM + RAG + Rule-based 零网络依赖 |

---

## 十、KPI 目标（修订版）

| 指标 | v1.0 目标 | v2.0 修订目标 | 当前状态 | 说明 |
|:---|:---|:---|:---|:---|
| 元器件识别率 | mAP@0.5 > 80% | **mAP@0.5 > 80%** | ❌ 待训练 | 需领域数据重训 |
| 端到端纠错延迟 | < 200ms | **< 500ms** | ⚠️ 待测 | 含 OCR + RAG 的完整链路 |
| LLM 首字生成时间 | < 3s | **< 3s (NPU)** | ✅ Cloud < 1s | NPU 待验证 |
| OCR 芯片识别率 | — | **> 90%** | ✅ 100% (7/7) | PaddleOCR 测试通过 |
| RAG 检索相关度 | — | **> 80%** | ✅ 87.3% | text2vec 嵌入 |
| 系统连续运行 | ≥ 2h 无崩溃 | **≥ 2h 无崩溃** | ⚠️ 待测 | 需压力测试 |
| 支持元器件种类 | 7类 | **8类** | ⚠️ 6类 | +IC +Transistor 需重训 |
| 校准成功率 | — | **> 95%** | ✅ 四级管线 | 多种光照已测试 |

---

## 十一、关键里程碑时间线

```
2月15日 ────── 现在 (代码框架 ~95% 完成)
    │
    ├── 2.15 - 3.05  Sprint A: 数据采集 + OBB标注 + 模型训练 (🔴 最高优先级)
    │
    ├── 3.05 - 3.20  Sprint B: 设计报告 + PPT + Demo视频
    │
    ├── 3.20 - 3.25  最终检查 + 提交
    │
3月25日 ────── 初选提交截止 ⏰
    │
    ├── 4月初       收到 DK-2500
    │
    ├── 4.01 - 4.14  Sprint D: 平台环境搭建 + YOLO/OCR 移植
    │
    ├── 4.15 - 5.01  Sprint E: LLM NPU 部署 + 端到端验证
    │
    ├── 5.01 - 6.01  Sprint F: 功能完善 + 差异化打磨
    │
    ├── 6.01 - 6.30  Sprint G: 最终报告 + 压力测试
    │
6月30日 ────── 最终作品提交 ⏰
```

---

*文档版本: v2.0 | 基于 src_v2 代码审计结果修订 | 下次更新: 模型训练完成后 (预计 3月5日)*
