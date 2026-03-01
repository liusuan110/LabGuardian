# LabGuardian v5.0 — 2026-03-01 变更日志

> **日期**: 2026 年 3 月 1 日
> **版本**: v3.2 → v5.0 (重大架构升级)
> **代码量**: ~9,500 行 → **~10,900 行** / 41 个 Python 源文件
> **新增文件**: 4 个  |  **修改文件**: 10 个

---

## 一、变更概览

本次升级涉及 **4 大改进方向**, 从感知层到 GUI 层全面重构:

| 改进方向 | 核心变更 | 影响层 |
|----------|----------|--------|
| **面包板电源轨精细建模** | 2 轨 → 4 轨, 系统推断 → 学生主动标注 | 推理层 + GUI 层 |
| **导线端点精准识别** | OBB 短边中点 → 骨架化端点检测 + 颜色分类 | 感知层 |
| **Image-only 架构迁移** | 视频流 → 图片上传分析 | 全栈 (架构级) |
| **元件库扩展** | 6 类 → 9 类 (三极管/IC/变阻器), IC 多引脚建模 | 推理层 + 感知层 |

---

## 二、详细变更

### 改进 1: 面包板电源轨精细建模

**动机**: 全尺寸面包板上下各有 2 条电源轨道 (共 4 条), 不同实验的用途不同 (如 +5V、+3.3V、GND)。旧版只有 `RAIL_TOP` / `RAIL_BOTTOM` 两个节点, 且通过 LED 极性投票自动推断 VCC/GND, 推断可能出错。

**变更**:

#### 2.1.1 config.py
- 移除旧字段: `power_rail_rows`, `vcc_rail_rows`, `gnd_rail_rows`
- 新增: `rail_track_rows: dict` — 4 条独立轨道映射

```python
rail_track_rows = {
    "RAIL_TOP_1":    (1,),    # 顶部外侧轨道
    "RAIL_TOP_2":    (2,),    # 顶部内侧轨道
    "RAIL_BOTTOM_1": (64,),   # 底部内侧轨道
    "RAIL_BOTTOM_2": (65,),   # 底部外侧轨道
}
```

#### 2.1.2 logic/circuit.py
- `__init__`: 参数改为 `rail_track_rows: Dict`, 构建 `_row_to_rail` 反向映射
- `_get_node_name()`: 行号查 `_row_to_rail`, 映射到 4 条轨道之一
- **删除** `_infer_power_rail_polarity()`: 移除 70 行投票推断逻辑
- `_identify_power_nets()`: 改为读取 `rail_assignments` (学生标注结果)
- **新增** `_parse_rail_label(label)`: 模糊匹配 VCC/GND (支持 "+5V", "地", "GND", "VCC +3.3V" 等)
- **新增** 5 个轨道管理方法: `set_rail_assignment()`, `clear_rail_assignments()`, `get_active_rail_tracks()`, `get_unassigned_active_rails()`, `get_rail_summary()`
- `get_circuit_description()`: 增加电源轨状态段落

#### 2.1.3 app_context.py
- `CircuitAnalyzer` 构造改为传入 `circuit_cfg.rail_track_rows`
- **新增** 5 个线程安全的电源轨管理代理方法

#### 2.1.4 gui_qt/circuit_page.py
- **新增** "电源轨配置" UI 分组:
  - 4 行 (每条轨道一行): 状态标签 + 预设下拉 (VCC +5V / +3.3V / +12V / GND) + 自定义输入
  - "清除所有标注" 按钮
- **新增** 信号: `rail_assigned(str, str)`, `rail_cleared`
- **新增** 方法: `highlight_unassigned_rails()`, `update_rail_status()`

#### 2.1.5 gui_qt/main_window.py
- 连接 pipeline 的 `on_rails_need_assignment` 回调
- 连接 circuit_page 的 `rail_assigned` / `rail_cleared` 信号

**数据流**:
```
检测到导线连接到电源轨行
  → CircuitAnalyzer.get_unassigned_active_rails() 返回 ["RAIL_TOP_1"]
  → 通知 GUI: "检测到电源轨连接, 请标注用途"
  → 学生选择 "VCC +5V"
  → set_rail_assignment("RAIL_TOP_1", "VCC +5V")
  → _identify_power_nets() 读取标注, RAIL_TOP_1 → VCC net
```

---

### 改进 2: 导线端点精准识别

**动机**: 导线可弯曲/交叉, OBB 矩形的短边中点不是真实端点位置。同色导线交叉时无法区分。

**变更**:

#### 2.2.1 vision/wire_analyzer.py (新建, 249 行)

`WireAnalyzer` 类, 核心算法:

1. **HSV 颜色分割**: 排除面包板白色背景, 提取彩色/黑色导线像素
2. **形态学清理**: 闭运算填补断裂 + 开运算去噪 + 保留最大连通域
3. **骨架化**: 优先使用 `cv2.ximgproc.thinning` (Zhang-Suen), 回退迭代形态学
4. **端点检测**: 8-邻域连接度 = 1 的骨架像素
5. **最远点对**: 如果端点 > 2 个 (分支), 用凸包加速选最远一对
6. **颜色分类**: 对导线像素 HSV 分布投票, 支持 红/蓝/绿/黄/橙/黑/白

```
输入: frame + wire bbox
  → 裁剪 + padding
  → HSV 分割 (非背景 ∩ (彩色 ∪ 黑色))
  → 最大连通域
  → 骨架化
  → 端点检测 → 最远对
  → 映射回原帧坐标
输出: ((端点1, 端点2), "red")
```

#### 2.2.2 vision/detector.py
- `Detection` dataclass 新增字段: `wire_color: Optional[str] = None`

#### 2.2.3 集成点 (image_analyzer.py)
- `_refine_wires()` 方法: 对所有 Wire 检测调用 `WireAnalyzer.analyze_wire()`
- 成功时覆盖 `det.pin1_pixel` / `det.pin2_pixel` 为骨架端点
- 设置 `det.wire_color`
- 失败时保留原始 OBB 端点 (安全降级)

---

### 改进 3: Image-only 架构迁移

**动机**: 竞赛场景中, 高分辨率手机俯拍照片比低分辨率 USB 摄像头视频流在准确率上严格更优。视频流增加系统复杂度 (多帧稳定化、帧差分、线程同步) 但收益有限。

**变更**:

#### 2.3.1 vision/image_analyzer.py (新建, 579 行)

`ImageAnalyzer` — 替代 `FramePipeline`, 一次性分析 1-3 张图片:

```python
@dataclass
class AnalysisResult:
    annotated_image: np.ndarray    # 标注图 (用于显示)
    detections: list               # 融合后的检测列表
    report: str                    # 结构化文本报告
    component_count: int = 0
    net_count: int = 0
    ocr_results: Dict[str, str]    # IC 型号识别结果
    issues: List[str]              # 检测到的问题
```

**分析管线**:
```
1-3 张图片
  → 每张独立 YOLO 检测 (支持 640/960/1280 分辨率)
  → Wire 骨架端点精炼 + 颜色分类
  → 多图 IoU 融合 (基准图 + 合并/新增)
  → 面包板校准 + 坐标映射
  → 电路拓扑建模 (含 IC 多引脚)
  → OCR 芯片型号识别
  → 极性解析 + 电路验证
  → 标注绘制 + 结构化报告生成
  → 返回 AnalysisResult
```

**多图融合算法**:
- 第1张为基准, 后续图片与基准做 IoU 匹配
- IoU ≥ 0.3: 同一元件, 取置信度更高的
- IoU < 0.3: 新元件 (被第1张遮挡)

#### 2.3.2 gui_qt/upload_page.py (新建, 296 行)

- 3 个 `ImageSlot` 缩略图槽位 (点击添加图片)
- 置信度滑块 (0.10 ~ 0.90) + 分辨率下拉 (640/960/1280)
- 进度条 + 分析/校准/清空按钮
- 分割器布局: 左侧标注结果图 + 右侧分析报告

#### 2.3.3 gui_qt/workers.py
- **新增** `ImageAnalysisWorker(QThread)`: 封装 `ImageAnalyzer.analyze()`, 信号 `progress(str)` / `finished(AnalysisResult)` / `error(str)`

#### 2.3.4 gui_qt/main_window.py (重写, 617 行)
- **移除**: VideoWorker 启动、摄像头初始化、帧处理回调、video 路由
- **新增**: UploadPage 为首页, ImageAnalysisWorker 驱动分析, 结果页显示标注图 + 报告
- 5 页路由: home (上传) → results (标注图+聊天) → chat → circuit → settings

#### 2.3.5 gui_qt/sidebar.py
- 导航项 "📹 视频检测" → "📷 分析结果"

#### 2.3.6 gui_qt/resources.py
- 新增图标: `UPLOAD` (📤), `ANALYZE` (🔍), `RESULT` (📊), `IMAGE` (🖼️), `CLEAR` (🗑️)

#### 2.3.7 app_context.py
- 移除视频状态: `_prev_gray`, `_cached_stable_dets`, `_current_frame`, `_video_running`, `_fps`
- `reset_analysis()` 简化为只重置 analyzer + OCR 缓存

**保留但不再导入的文件** (向后兼容):
- `gui_qt/frame_pipeline.py` (404 行)
- `gui_qt/video_panel.py` (161 行)

---

### 改进 4: 元件库扩展 (D1 + D2 + D3)

**动机**: 原 YOLO 模型仅支持 6 类元件, 缺少三极管、IC、变阻器。竞赛中运放电路 (LM324/LM358) 是常见题目。

#### 2.4.1 YOLO 数据集扩展 (D1)

config.py:
```python
# 旧: 6 类
COMPONENT_CLASSES = ["CAPACITOR", "DIODE", "LED", "RESISTOR", "Push_Button", "Wire"]

# 新: 9 类
COMPONENT_CLASSES = [
    "CAPACITOR", "DIODE", "LED", "RESISTOR", "Push_Button", "Wire",
    "TRANSISTOR", "IC_DIP", "POTENTIOMETER"
]
```

新增集合:
- `IC_DIP_COMPONENTS = {"IC_DIP", "IC", "CHIP", "DIP", "OPAMP"}`
- `THREE_PIN_COMPONENTS` 增加 `"POTENTIOMETER"`

detector.py 新增引脚延伸:
```python
_PIN_EXTENSION = {
    ...
    "TRANSISTOR":    0.10,  # TO-92 封装引脚
    "IC_DIP":        0.02,  # DIP 引脚紧贴芯片体
    "POTENTIOMETER": 0.08,  # 电位器引脚
}
```

#### 2.4.2 IC 引脚数据库 (D3) — logic/ic_pinout_db.py (新建, 198 行)

内置 3 款常见运放 IC 的完整引脚定义:

| IC 型号 | 封装 | 引脚数 | 别名 |
|---------|------|--------|------|
| **LM324** | DIP-14 | 14 | LM324N, LM324AN, LM2902 |
| **LM358** | DIP-8 | 8 | LM358N, LM358P, LM358D, LM358AN |
| **NE5532** | DIP-8 | 8 | NE5532N, NE5532P, SA5532 |

核心功能:
- `lookup_ic(model_name)`: 精确匹配 → 别名 → 前缀匹配 → 反向前缀
- `get_ic_pin_locs(info, top_row)`: DIP 布局计算, 左侧 'e' 列递增, 右侧 'f' 列递减

#### 2.4.3 circuit.py — IC 多引脚建模

新增 PinRole 值:
```python
WIPER         # 电位器滑动触点
TERMINAL_A/B  # 电位器端子
IC_PIN        # 通用 IC 引脚
IC_VCC/GND    # IC 电源
IC_OUTPUT     # IC 输出
IC_INPUT_INV  # IC 反相输入
IC_INPUT_NON  # IC 同相输入
```

IC Hub 节点模型:
```
面包板:  Row5_L ── IC_PIN_1 ──┐
         Row6_L ── IC_PIN_2 ──┤
         Row7_L ── IC_PIN_3 ──┼── [U1_hub] ──┬── IC_PIN_8 ── Row5_R
         Row8_L ── IC_PIN_4 ──┘              ├── IC_PIN_7 ── Row6_R
                                              ├── IC_PIN_6 ── Row7_R
                                              └── IC_PIN_5 ── Row8_R
```

IC_VCC/IC_GND 引脚自动纳入电源网络识别。

#### 2.4.4 polarity.py — 变阻器与 IC 极性解析

- **新增** `_resolve_potentiometer_pins()`: 与三极管同逻辑, pin1=Terminal_A, pin3=Wiper, pin2=Terminal_B
- **新增** `_resolve_ic_pins()`: 检查 `pin_locs` 是否已填充 (由 `image_analyzer._build_ic_component` 预设), 标记极性为 FORWARD

#### 2.4.5 image_analyzer.py — IC 检测到入图的完整链路

```
YOLO 检测到 IC_DIP bbox
  → OCR 识别丝印 → "LM324"
  → ic_pinout_db.lookup_ic("LM324") → ICPackageInfo (14 pin)
  → 校准: bbox 中心 → 面包板行号 (top_row)
  → get_ic_pin_locs(info, top_row=5) → [(5,'e'), (6,'e'), ..., (5,'f'), ...]
  → CircuitComponent(pin_locs=[...], ic_model="LM324")
  → build_topology_graph(): hub 节点 + 14 条边
```

---

## 三、文件变更清单

### 新建文件 (4)

| 文件 | 行数 | 职责 |
|------|------|------|
| `vision/wire_analyzer.py` | 249 | 导线骨架分析 + 端点检测 + 颜色分类 |
| `vision/image_analyzer.py` | 579 | 图片分析引擎 (替代 FramePipeline) |
| `logic/ic_pinout_db.py` | 198 | IC 引脚数据库 (LM324/LM358/NE5532) |
| `gui_qt/upload_page.py` | 296 | 图片上传与分析 UI |

### 修改文件 (10)

| 文件 | 原行数 | 新行数 | 主要变更 |
|------|--------|--------|----------|
| `config.py` | 422 | 425 | 9 类 COMPONENT_CLASSES, 4 轨道 rail_track_rows, IC_DIP_COMPONENTS |
| `logic/circuit.py` | 560 | 758 | IC hub 模型, 4 轨道, 学生标注, 电位器/IC PinRole |
| `logic/polarity.py` | 280 | 295 | 电位器/IC 极性解析方法 |
| `vision/detector.py` | 250 | 272 | wire_color 字段, 3 个新 PIN_EXTENSION |
| `app_context.py` | 180 | 241 | 移除视频状态, 新增 5 个电源轨方法 |
| `gui_qt/main_window.py` | 960 | 617 | 完全重写: Image-only 架构 |
| `gui_qt/workers.py` | 221 | 430 | 新增 ImageAnalysisWorker |
| `gui_qt/circuit_page.py` | 120 | 214 | 电源轨标注 UI (4 行 combo + 自定义输入) |
| `gui_qt/sidebar.py` | 181 | 148 | 视频检测 → 分析结果 |
| `gui_qt/resources.py` | 34 | 46 | UPLOAD/ANALYZE/RESULT/IMAGE/CLEAR 图标 |

### 未修改但标记为弃用 (2)

| 文件 | 状态 |
|------|------|
| `gui_qt/frame_pipeline.py` | 保留兼容, 不再导入 |
| `gui_qt/video_panel.py` | 保留兼容, 不再导入 |

---

## 四、架构对比 (Before / After)

### 系统核心链路

**v3.2 (旧)**:
```
USB 摄像头 → 帧采集 → YOLO 检测 → 多帧稳定 → 校准映射
  → 电路拓扑 → 极性推断 → VCC/GND 自动推断 → 实时标注
```

**v5.0 (新)**:
```
手机俯拍 → 图片上传 (1-3 张)
  → YOLO 检测 (高分辨率 1280)
  → Wire 骨架端点精炼 + 颜色分类
  → 多图 IoU 融合 (去遮挡)
  → 校准映射 + IC OCR + 引脚数据库查询
  → 电路拓扑 (含 IC hub 多引脚)
  → 极性推断 (含电位器/IC)
  → 电源轨: 学生主动标注 VCC/GND
  → 结构化报告 + 标注图
```

### 元件支持对比

| 元件 | v3.2 | v5.0 |
|------|------|------|
| 电阻 | ✅ 2-pin | ✅ 2-pin |
| LED | ✅ 极性检测 | ✅ 极性检测 |
| 二极管 | ✅ 极性检测 | ✅ 极性检测 |
| 电容 | ✅ 2-pin | ✅ 2-pin |
| 按键 | ✅ 2-pin | ✅ 2-pin |
| 导线 | ✅ OBB 短边中点 | ✅ **骨架端点 + 颜色** |
| 三极管 | ✅ E/B/C (TO-92) | ✅ E/B/C (TO-92) |
| **IC (DIP)** | ❌ | ✅ **OCR → 引脚数据库 → 多引脚映射** |
| **变阻器** | ❌ | ✅ **3-pin (Terminal_A/Wiper/Terminal_B)** |

### 电源轨模型对比

| 特性 | v3.2 | v5.0 |
|------|------|------|
| 轨道数 | 2 (RAIL_TOP / RAIL_BOTTOM) | **4** (TOP_1/2, BOTTOM_1/2) |
| VCC/GND 判定 | 投票推断 (LED 极性 + 三极管) | **学生手动标注** |
| +/- 轨区分 | 无 | **每条轨道独立标注** |
| 电压值 | 仅 VCC/GND | **VCC +5V / +3.3V / +12V / GND / 自定义** |

---

## 五、技术决策记录

### 决策 1: 放弃视频流, 采用 Image-only
- **原因**: 手机 12MP 俯拍远优于 USB 摄像头 640×480; 不需要多帧稳定化; 代码复杂度降低 40%
- **权衡**: 失去实时反馈能力, 但竞赛场景以结果准确性为首要目标

### 决策 2: 电源轨由学生标注而非系统推断
- **原因**: 不同实验的电源配置差异大 (+5V、+3.3V、双电源); 投票推断需要足够多的极性元件, 简单电路可能推断错误
- **权衡**: 增加一步学生操作, 但消除了推断失败的风险

### 决策 3: 导线用骨架化而非训练端点检测模型
- **原因**: 纯 OpenCV 操作, 无需额外标注数据和训练; 对弯曲导线天然适配
- **权衡**: 对导线交错重叠场景, 最大连通域可能混入其他导线

### 决策 4: IC 用 OCR + 查表而非训练分类模型
- **原因**: IC 型号有限 (竞赛常用 LM324/LM358), 查表 100% 准确; 无需额外训练数据
- **权衡**: 不支持数据库外的 IC (后期可扩展)

---

## 六、下一步计划 (截止 3.25 提交)

### 紧急 (P0) — 必须在 3.10 前完成

| 任务 | 说明 | 预计工时 |
|------|------|----------|
| **YOLO 模型重训练** | 用新拍摄的9类数据集 (含三极管/IC/变阻器) 重训练 YOLOv8-OBB, 验证mAP | 2-3 天 |
| **端到端集成测试** | 用真实面包板照片跑完整管线: 上传→检测→校准→分析→报告, 修复断链 | 1 天 |
| **离线依赖全量预下载** | 创建 `scripts/prepare_offline.py`, 确保 text2vec / PaddleOCR / ChromaDB 全部离线可用 | 0.5 天 |
| **校准器行号对齐** | 用实际面包板校准结果确认 `rail_track_rows` 的行号 (1/2/64/65 是否正确) | 0.5 天 |

### 重要 (P1) — 3.15 前完成

| 任务 | 说明 | 预计工时 |
|------|------|----------|
| **Ubuntu 20.04 部署脚本** | `scripts/setup_ubuntu.sh` + OpenVINO/NPU 驱动安装, Python 3.10 venv | 1 天 |
| **RAGEngine 离线保障** | embedding_model 默认从 `models/text2vec_chinese/` 加载, 失败再在线 | 0.5 天 |
| **knowledge_base 扩展** | 补充 LM324/LM358 运放电路知识文档 (RAG 可检索) | 0.5 天 |
| **benchmark 更新** | `tools/benchmark.py` 适配 Image-only 架构, 测试单张/多图分析延迟 | 0.5 天 |

### 优化 (P2) — 3.20 前完成

| 任务 | 说明 | 预计工时 |
|------|------|----------|
| **WireAnalyzer 调优** | 用 50+ 张真实导线图片测试骨架化效果, 调整 HSV 阈值和形态学参数 | 1 天 |
| **IC OCR 鲁棒性** | 测试 LM324/LM358 丝印在不同光照/角度下的识别率, 增加错字修正规则 | 0.5 天 |
| **多图融合效果验证** | 测试遮挡场景: 1张 vs 2张 vs 3张的检测准确率对比 | 0.5 天 |
| **界面美化** | UploadPage / CircuitPage 电源轨 UI 细节打磨, 适配 DK-2500 屏幕分辨率 | 0.5 天 |

### 收尾 (P3) — 3.22 前完成

| 任务 | 说明 | 预计工时 |
|------|------|----------|
| **设计报告更新** | 补充 v5.0 架构图、导线骨架算法、IC 多引脚模型到设计报告 | 1 天 |
| **演示用例准备** | 准备 3-5 个经典电路 (LED 限流、运放反相、三极管开关) 的黄金参考 + 各种错误变体 | 1 天 |
| **DK-2500 实机部署** | 在竞赛平台上完整部署, 确认 NPU 驱动 + OpenVINO + 全部离线依赖 | 1 天 |

### 冻结 — 3.23 代码冻结, 3.25 提交

---

## 七、已知限制和风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| YOLO 9 类模型未训练 | 三极管/IC/变阻器无法检测 | **P0: 立即开始标注 + 训练** |
| 校准器行号未验证 | 电源轨映射可能错误 | 用真实面包板校准后调整 config |
| DK-2500 NPU 驱动 | Ubuntu 20.04 可能需要特定内核版本 | 提前在 DK-2500 上测试 OpenVINO NPU |
| PaddleOCR 首次下载 | 离线环境无法下载模型 | prepare_offline.py 预下载 |
| 导线骨架化对交叉导线 | 最大连通域可能混入其他导线 | 结合 wire_color 区分, 必要时请求第二张照片 |
| IC 引脚数据库有限 | 仅支持 LM324/LM358/NE5532 | 竞赛范围内应已足够, 可按需扩展 |

---

## 八、关键里程碑

```
3.01  ✅  v5.0 代码完成 (面包板/导线/架构/元件库)
3.02~3.05  拍摄训练数据 + 标注
3.05~3.08  YOLO 9 类模型训练 + 验证
3.08~3.10  端到端集成测试 + 修复
3.10~3.12  Ubuntu 20.04 移植 + 离线打包
3.12~3.15  DK-2500 实机部署 + NPU 调试
3.15~3.18  WireAnalyzer / OCR 调优
3.18~3.20  多图融合测试 + UI 美化
3.20~3.22  设计报告更新 + 演示准备
3.23       代码冻结
3.25       提交设计报告
