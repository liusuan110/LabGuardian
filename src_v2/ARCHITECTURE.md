# LabGuardian v3 -- 系统架构文档

> **版本**: v3.2 (电源轨智能推断 + 三极管引脚视觉检测)
> **更新日期**: 2026-02-24
> **总代码量**: ~9,500 行 / 41 个 Python 源文件
> **竞赛平台**: Intel Core Ultra 5 225U (DK-2500)

---

## 一、系统定位

LabGuardian 是一个**边缘端全离线**的电子实验智能助教系统。核心链路:

```
USB 摄像头
   |
   v
[YOLO 元件检测]  ------>  [面包板校准 + 坐标映射]
   |                              |
   v                              v
[多帧稳定化]              [电路拓扑图 (NetworkX)]
   |                              |
   v                              v
[OCR 芯片识读]            [VF2++ 电路验证]
   |                              |
   v                              v
[RAG 知识检索]            [极性诊断 + Ghost Wire]
   |                              |
   v                              v
[LLM 智能问答]  <------- [自然语言网表上下文]
   |
   v
[PySide6 GUI 实时展示]
```

---

## 二、四层架构

```
+------------------------------------------------------------------+
|                         GUI 层 (gui_qt/)                          |
|  PySide6 无边框窗口 / PyDracula 暗色主题 / 5 页路由               |
|  VideoPanel / Dashboard / ChatPanel / CircuitPage / Settings      |
|  QThread Workers: VideoWorker / ModelLoaderWorker / LLMWorker     |
+------------------------------------------------------------------+
|                        认知层 (ai/)                                |
|  LLMEngine: Cloud(DeepSeek) -> Local(OpenVINO NPU) -> Rules      |
|  RAGEngine: ChromaDB + text2vec-base-chinese (49 文本块)          |
|  OCREngine: PaddleOCR(主) -> EasyOCR(备) -> 无OCR(兜底)          |
+------------------------------------------------------------------+
|                        推理层 (logic/)                             |
|  CircuitAnalyzer: NetworkX 图 + 自动命名 + 电源轨总线 + VCC/GND 推断|
|  CircuitValidator: VF2++ 图同构 + 4 级诊断 + 6 项拓扑检查          |
|  PolarityResolver: OBB 极性推断 + pin3 视觉优先 + TO-92 引脚约定   |
+------------------------------------------------------------------+
|                        感知层 (vision/)                            |
|  ComponentDetector: YOLOv8 HBB+OBB 双模检测 + 引脚延伸先验        |
|  DetectionStabilizer: 5 帧滑窗 IoU 投票 (>=3 票确认)             |
|  BreadboardCalibrator: 4 级孔洞检测 + Grid RANSAC + Top-K 候选   |
+------------------------------------------------------------------+
```

### 层间依赖规则

```
GUI 层  --->  认知层 (LLM 问答)
  |           推理层 (电路验证)
  |           感知层 (YOLO 检测)
  |
  +-- 全部通过 AppContext 间接访问, 不直接构造底层模块

认知层  --->  推理层 (RAG 需要电路上下文)
推理层  --->  感知层 (电路分析需要检测结果 + 校准坐标)
感知层  --->  无下层依赖 (仅依赖 OpenCV / YOLO)
```

**关键设计**: 低层模块不依赖高层, 高层通过 AppContext 获取低层服务。
任何一层可以独立替换或测试。

---

## 三、AppContext -- 服务注册中心

```
                     AppContext (app_context.py)
                    /      |       |       \
                   /       |       |        \
          感知层服务    推理层服务   认知层服务   线程安全设施
          ---------    ---------   ---------   -----------
          detector     analyzer    llm         ReadWriteLock
          stabilizer   polarity    ocr         _ocr_lock
          calibrator   validator              _ghost_lock
                                               _desc_lock
```

### 3.1 为什么引入 AppContext

| 问题 | AppContext 之前 | AppContext 之后 |
|------|----------------|-----------------|
| 模块构造 | MainWindow.__init__ 直接 new 全部模块 | AppContext 统一构造, MainWindow 只接收 ctx |
| 可测试性 | 必须启动 GUI 才能测试业务逻辑 | `ctx = AppContext()` 即可独立测试 |
| 线程安全 | 无任何保护, 竞态条件 | ReadWriteLock + 互斥锁 + 快照机制 |
| 替换实现 | 改 MainWindow 代码 | 改 AppContext 构造即可 |
| Benchmark | 无法脱离 GUI 运行性能测试 | `tools/benchmark.py` 直接使用 AppContext |

### 3.2 生命周期

```python
# gui_qt/run.py
ctx = AppContext()           # 1. 创建全部服务实例
window = MainWindow(ctx=ctx) # 2. 注入到 GUI
window.show()                # 3. GUI 启动, 触发模型加载
# ... 运行 ...
# closeEvent -> video_worker.stop()  # 4. 关闭时清理
```

### 3.3 线程安全模型

```
                    +-------------------------+
                    |    VideoWorker 线程      |
                    |                         |
                    |  detector.detect()      |  <-- 只读, 无需锁
                    |        |                |
                    |  with write_lock():     |  <-- 写锁保护
                    |    stabilizer.update()  |
                    |    analyzer.reset()     |
                    |    analyzer.add_comp()  |
                    |    update_snapshot()    |  <-- 生成文本快照
                    |        |                |
                    |  ocr_cache_set()       |  <-- 互斥锁
                    |  get_missing_links()   |  <-- 互斥锁, 返回副本
                    +-------------------------+
                              |
                    (frame_ready Signal)
                              |
                    +-------------------------+
                    |      主线程 (UI)         |
                    |                         |
                    |  with read_lock():      |  <-- 读锁保护
                    |    validator.compare()  |
                    |    get_description()    |
                    |    set_reference()      |
                    |                         |
                    |  set_missing_links()    |  <-- 互斥锁
                    |  get_circuit_snapshot() |  <-- 互斥锁, 无阻塞
                    +-------------------------+
                              |
                    +-------------------------+
                    |    LLMWorker 线程        |
                    |                         |
                    |  llm.ask(question,      |
                    |    context=snapshot)     |  <-- 用快照, 完全无锁
                    +-------------------------+
```

**ReadWriteLock 保证**:
- 多个读者可以并发 (UI 线程的 validate / netlist / set_ref)
- 写者独占 (VideoWorker 每帧 reset+rebuild)
- 写者优先 (防止写饥饿导致视频帧卡顿)

**快照机制**:
`update_circuit_snapshot()` 在写锁内生成 `analyzer.get_circuit_description()` 的字符串副本,
并附加 `CircuitValidator.diagnose()` 的独立诊断结果 (v3.2 新增)。
LLM 问答通过 `get_circuit_snapshot()` 读取, 完全不需要获取读写锁, 避免 LLM 推理 (可能数秒) 期间阻塞视频线程。

---

## 四、各层详细设计

### 4.1 感知层 (vision/)

#### ComponentDetector (`detector.py`, ~250 行)

```
输入: BGR 帧 (np.ndarray)
   |
   v
YOLO predict(frame, conf, iou, imgsz)
   |
   +-- HBB 模式: bbox = (x1, y1, x2, y2)
   |   pin_pixel = bbox 边缘向外延伸 (元件类型先验补偿)
   |
   +-- OBB 模式: obb_corners = 4 个角点
       pin_pixel = 短边中点沿长轴向外延伸 (遮挡补偿)
   |
   v
输出: List[Detection]
```

**OBB vs HBB 自动切换**: 检测结果是否包含 `obb` 属性决定使用哪种模式。
OBB 提供旋转矩形, 更适合面包板上斜插的元件, 同时为极性推断提供方向信息。

**引脚延伸先验** (v3.1 新增):
元件本体遮挡导致 OBB/HBB 边缘 ≠ 真实引脚插入点。按元件类型沿长轴向外延伸:

| 元件类型 | 延伸比例 (× 长边) | 物理原因 |
|----------|-------------------|----------|
| RESISTOR | 0.10 | 引线细长, 略超出色环体 |
| LED | 0.08 | 顶视时引脚被圆顶遮挡 |
| DIODE | 0.10 | 类似电阻 |
| CAPACITOR | 0.08 | 引脚较短 |
| Wire | 0.02 | 端点几乎就是连接点 |
| Push_Button | 0.06 | 引脚在底部 |

#### DetectionStabilizer (`stabilizer.py`, ~120 行)

```
滑窗 = deque(maxlen=5)

每帧:
  1. 将当前帧检测结果加入滑窗
  2. 对当前帧每个检测 d:
     a. 在 5 帧历史中查找 class_name 相同且 IoU >= 0.3 的检测
     b. 统计命中次数 hit_count
     c. hit_count >= 3 --> 通过投票, 加入 stable_results
  3. 返回 List[StableDetection] (附加 stability 分数)
```

**设计意图**: 消除单帧误检/漏检抖动。在竞赛现场光照不稳定时尤为关键。
5 帧窗口 + 3 票阈值在实测中平衡了响应速度和稳定性。

#### BreadboardCalibrator (`calibrator.py`, ~740 行)

四级检测管线:

```
原图 -> CLAHE 预处理 -> 多阈值
  |
  +-- Level 1: cv2.findCirclesGrid()      快速, 标准面包板
  +-- Level 2: Multi-Blob Ensemble        鲁棒, 覆盖面广
  +-- Level 3: cv2.HoughCircles()         补漏
  +-- Level 4: Contour Analysis           兜底
  |
  v
合并去重 (NMS by distance)
  |
  v
Grid RANSAC 拟合 (剔离群 + 补缺)
  |
  v
坐标映射:
  nearest_hole()                 → 单一最近孔洞 (向后兼容)
  top_k_holes(k=3)              → K 个最近孔洞 + 距离分数 (v3.1 新增)
  frame_pixel_to_logic_candidates(k) → K 个候选逻辑坐标 (v3.1 新增)
```

**Top-K 候选孔洞** (v3.1 新增):
当引脚位置因遮挡不精确时, 单一 nearest_hole 可能选错。
`top_k_holes()` 用 `np.argpartition` O(N) 返回 K 个最近孔洞及距离。
`frame_pixel_to_logic_candidates()` 完成完整管线: 透视变换 → Top-K 孔洞 → 逻辑坐标去重。
无候选时自动回退线性映射。

### 4.2 推理层 (logic/)

#### CircuitAnalyzer (`circuit.py`, ~770 行)

**核心数据结构: NetworkX 图**

```
节点 = 面包板导通组:
  Row{n}_L    — 第 n 行左侧 (a-e 五孔导通)
  Row{n}_R    — 第 n 行右侧 (f-j 五孔导通)
  RAIL_TOP    — 顶部电源轨总线 (v3.2: 整行导通, 不分左右)
  RAIL_BOTTOM — 底部电源轨总线 (v3.2: 整行导通, 不分左右)
  PWR_PLUS    — 电源正轨 (旧格式兼容: +/plus/P 列)
  PWR_MINUS   — 电源负轨 (旧格式兼容: -/minus/N/GND 列)

边 = 元件连接:
  属性: component(名称), type(类型), polarity, confidence,
        pin1_role, pin2_role, junction("EB"/"BC" 仅三端)
```

**v3.2 新增功能**:

| 功能 | 说明 |
|------|------|
| `__init__(rail_top_rows, rail_bottom_rows)` | 构造器接受电源轨行号, 由 AppContext 从 CircuitConfig 注入 |
| `RAIL_TOP` / `RAIL_BOTTOM` 节点 | 电源轨行映射为总线节点 (整行导通, 不区分 L/R), 替代旧的 Row{n}_L/R |
| `_infer_power_rail_polarity()` | 投票机制推断 VCC/GND: LED 极性 (+2), 三极管 E/C (+1), 间接导线 (+1), 默认上红下蓝 |
| `_identify_power_nets()` | 增强: 支持 RAIL_TOP/RAIL_BOTTOM + PWR_PLUS/PWR_MINUS + 元件引脚角色 三种来源 |

**v3.1 已有功能**:

| 功能 | 说明 |
|------|------|
| `norm_component_type()` | 模块级统一类型归一化, 支持 12+ 种元件类型, CircuitAnalyzer 和 PolarityResolver 共用 |
| `_auto_name()` | 元件自动命名 R1/R2/LED1/Q1..., 基于 `_TYPE_PREFIX` 字典 + 计数器 |
| 三端边建模 | 三极管建模为 E-B + B-C 两条结边 (而非错误的 pin1→pin3 直连) |
| 电源轨节点 | `+`/`-` 列映射为 `PWR_PLUS`/`PWR_MINUS` 共享总线 |
| 结构化描述 | `get_circuit_description()` 输出 5 段中文: 元件统计→引脚位置→网络→电源→问题检测 |
| 快速检查 | `_quick_check_issues()` 自动发现短路/开路/极性未知等异常 |

**元件自动命名规则**:
```python
_TYPE_PREFIX = {
    "RESISTOR": "R", "LED": "LED", "DIODE": "D",
    "CAPACITOR": "C", "WIRE": "W", "PUSH_BUTTON": "SW",
    "NPN": "Q", "PNP": "Q", "TRANSISTOR": "Q",
    "OPAMP": "U", "IC_555": "U", "POWER": "V",
}
# 名称 = 前缀 + 递增序号: R1, R2, LED1, Q1, Q2, ...
```

**三端元件边建模** (v3.1 修正):
```
原来 (错误):  E ────────────── C    (pin1 直连 pin2, 忽略 B 极)

现在 (正确):  E ──[EB结]── B ──[BC结]── C
              node1       node3       node2
```

**电源轨 VCC/GND 推断** (v3.2 新增):

```
输入: 当前电路中所有元件的拓扑连接
  |
  v
投票计算 (vote > 0 → 顶部=VCC, vote < 0 → 顶部=GND):
  +-----------------------------+-----------+
  | 证据                        |  投票权重 |
  +-----------------------------+-----------+
  | LED 阴极直连 RAIL_TOP       |    -2     |
  | LED 阴极直连 RAIL_BOTTOM    |    +2     |
  | LED 阳极直连 RAIL_TOP       |    +2     |
  | LED 阳极直连 RAIL_BOTTOM    |    -2     |
  | LED 阴极经导线到 RAIL_TOP   |    -1     |
  | LED 阴极经导线到 RAIL_BOTTOM|    +1     |
  | 三极管发射极连 RAIL_TOP     |    -1     |
  | 三极管发射极连 RAIL_BOTTOM  |    +1     |
  | 三极管集电极连 RAIL_TOP     |    +1     |
  | 三极管集电极连 RAIL_BOTTOM  |    -1     |
  +-----------------------------+-----------+
  |
  v
vote >= 0 → (RAIL_TOP="VCC", RAIL_BOTTOM="GND")   [面包板约定: 上红下蓝]
vote <  0 → (RAIL_TOP="GND", RAIL_BOTTOM="VCC")   [反向供电]
```

#### CircuitValidator (`validator.py`, ~850 行)

**四级诊断管线**:

```
L0  快速预检
    - 元件类型计数比较 (O(n))
    - 不一致则直接报告 "缺少 2x RESISTOR" 等

L1  全图同构 (VF2++)
    - 度序列签名预拒 (O(n log n)) -- 不一致则跳过 VF2++
    - 节点匹配: kind + ctype + pins + polarity
    - 边匹配: pin_role
    - 通过 --> "拓扑完全匹配"

L2  子图同构
    - 检查 curr 是否是 ref 的子图
    - 场景: 学生只搭了部分电路
    - 输出: progress = matched / total

L2.5 极性专项诊断
    - 用不带极性的 VF2++ 重新匹配
    - 通过但 L1 未通过 --> 仅极性错误
    - 输出: "LED_3 接反了, 请对调阳极和阴极"

L3  GED (图编辑距离)
    - 小图 (<=50 节点): 精确解
    - 大图: 三维度近似 (类型余弦 + 度序列 + 边比值)
    - 输出: similarity = 0.0 ~ 1.0
```

**度序列预拒的价值**: VF2++ 时间复杂度为 O(n! * n) 最坏情况。
通过 O(n log n) 的度序列比较提前排除不可能的匹配, 避免在竞赛现场卡顿。

**`diagnose()` 静态方法** (v3.1 新增):
独立于 Golden Reference 的拓扑健康检查, 直接分析当前电路:

| 检查项 | 检测方法 | 严重程度 |
|--------|----------|----------|
| LED 无限流电阻 | 图邻居遍历: LED 边的对端节点是否连接 RESISTOR | 高 (烧毁风险) |
| 极性未知 | 极化元件 polarity == UNKNOWN | 中 |
| 同组短路 | 两引脚映射到同一导通组 (Row{n}_L 或 Row{n}_R) | 高 |
| 三极管缺引脚 | pin3_loc is None | 高 |
| 悬空节点 | 图中度数 = 1 的节点 (开路风险) | 中 |
| 断路子图 | 连通分量 > 1 (电路不连通) | 低 |

#### PolarityResolver (`polarity.py`, ~280 行)

```
                  元件类型
                 /   |   \
                /    |    \
POLARIZED     NON_POLAR   THREE_PIN
(DIODE/LED)  (R/Wire/Btn) (NPN/PNP)
    |             |            |
OBB 长轴       NONE       视觉 pin3 优先 (v3.2)
方向推断     (无极性)     → 校准孔洞搜索
    |                    → 回退: 行号中点插值
  FORWARD                E/B/C 分配
```

**v3.2 变更**: 三极管引脚推断采用**视觉优先**策略:
1. 优先使用 `_find_transistor_pin3()` 视觉检测结果 (comp.pin3_loc 已设置)
2. 回退: 仅当视觉检测未找到 pin3 且 row_span >= 2 时, 使用行号中点插值

**v3.1 变更**: `_norm_type()` 方法改为委托给模块级 `norm_component_type()`,
消除与 CircuitAnalyzer 之间的重复代码和命名不一致问题。

### 4.3 认知层 (ai/)

#### LLMEngine (`llm_engine.py`, ~600 行)

**三级降级策略**:

```
try Cloud (DeepSeek API):
  - 在线时最优体验
  - DEEPSEEK_BASE_URL + API_KEY
  |
  v (失败 / 离线)
try Local (OpenVINO GenAI on NPU):
  - Qwen2.5-1.5B INT4 或 MiniCPM-1B INT4
  - openvino_genai.LLMPipeline(model_dir, device="NPU")
  - 完全离线, 无网络依赖
  |
  v (失败 / 未安装)
Rule-based Fallback:
  - 9 套元件知识规则 (电阻色环 / LED 方向 / 三极管引脚 ...)
  - 关键词匹配 -> 预置回答
  - 零依赖, 100% 可用
```

**系统提示词设计**:
```
你是 LabGuardian 助教, 专门辅导电子实验。
当前电路状态: {circuit_context}
请基于上述电路信息回答学生的问题。
```

`circuit_context` 来自 `AppContext.get_circuit_snapshot()`, 包含元件列表、网表、极性信息。

**`RuleBasedBackend._summarize_circuit()`** (v3.1 新增):
当用户询问概况类问题时, 从 circuit_context 中解析元件统计和网络数量,
生成中文概要回复, 不依赖 LLM。

#### RAGEngine (`rag_engine.py`, 530 行)

```
知识文档 (markdown)
  |
  v
text2vec-base-chinese 编码为向量
  |
  v
ChromaDB 存储 (49 个文本块, overlap=100 字符)
  |
  v
查询时:
  1. 问题编码为向量
  2. 余弦相似度 Top-K 检索
  3. 过滤 min_score < 0.35
  4. 拼接到 LLM 提示词中
```

#### OCREngine (`ocr_engine.py`, 611 行)

```
YOLO 检测到 IC/CHIP/TRANSISTOR 类别
  |
  v
裁剪 bbox 区域 + padding
  |
  v
原图识别 -> 失败? -> 预处理 (CLAHE + 二值化) 再识别
  |
  v
13 种正则匹配芯片型号 (NE555, LM358, S8050, 74HC...)
  |
  v
OCR 错字修正表 (NES55 -> NE555, LM3S8 -> LM358 ...)
  |
  v
触发 RAG 自动知识检索
```

### 4.4 GUI 层 (gui_qt/)

#### 页面路由

```
Sidebar (导航)
  |
  +-- home     ->  Dashboard (左: VideoPanel, 右: 状态卡)
  +-- video    ->  VideoPanel + ChatPanel (视频+侧栏聊天)
  +-- chat     ->  ChatPanel (全屏聊天)
  +-- circuit  ->  CircuitPage (验证/模板/网表/原理图)
  +-- settings ->  SettingsPage (摄像头/YOLO/LLM 配置)
```

#### 信号槽架构

```
VideoWorker (QThread)
  |
  +-- frame_ready(QImage)  ----------->  VideoPanel.update_frame()
  +-- fps_updated(float)   ----------->  Dashboard.update_fps()
  +-- _process_callback(frame) -------->  MainWindow._process_frame()
                                          (在 VideoWorker 线程执行)

ChatPanel
  |
  +-- message_sent(str) ------>  MainWindow._ask_ai()
                                   |
                                   v
                            LLMWorker (QThread)
                                   |
                                   +-- response_ready(str) -> ChatPanel.add_message()

CircuitPage
  |
  +-- validate_requested() -->  MainWindow._validate_circuit()
  +-- golden_ref_requested -> MainWindow._set_golden_ref()
  +-- show_netlist_requested -> MainWindow._show_netlist()
```

---

## 五、数据流 -- 一帧的完整旅程

```
+----------------------------------------------------------------+
|                     VideoWorker 线程                            |
|                                                                |
|  1. cap.read() -> frame (BGR np.ndarray)                       |
|  2. callback = _process_frame(frame)                           |
|     |                                                          |
|     +-- detector.detect(frame, conf=0.25)                      |
|     |   -> List[Detection] (6 类: R/LED/C/Wire/Btn/Diode)      |
|     |   引脚位置已含延伸先验补偿 (v3.1)                          |
|     |                                                          |
|     +-- [write_lock]                                           |
|     |   stabilizer.update(detections)                          |
|     |   -> List[StableDetection] (过滤抖动)                    |
|     |                                                          |
|     |   annotate_frame(frame, stable_dets)                     |
|     |   -> annotated (带检测框的帧)                             |
|     |                                                          |
|     |   if calibrated:                                         |
|     |     analyzer.reset()                                     |
|     |     for det in stable_dets:                              |
|     |       cands1 = calibrator.frame_pixel_to_logic_candidates(pin1, k=3)
|     |       cands2 = calibrator.frame_pixel_to_logic_candidates(pin2, k=3)
|     |       loc1, loc2 = _pick_best_pin_pair(cands1, cands2, type)
|     |       comp = CircuitComponent(loc1, loc2, auto_name)     |
|     |       polarity_resolver.enrich(comp, obb_corners, angle) |
|     |       analyzer.add_component(comp)                       |
|     |     update_circuit_snapshot()                             |
|     |   [/write_lock]                                          |
|     |                                                          |
|     +-- ghost_wires = get_missing_links()                      |
|     |   draw_ghost_wires(annotated, ghost_wires)               |
|     |                                                          |
|     +-- every 30 frames:                                       |
|     |   ocr.recognize_chip(frame, bbox)                        |
|     |   -> cache_set(key, chip_model)                          |
|     |   -> auto_rag_lookup(chip_model) -> RAG 查询             |
|     |                                                          |
|     +-- draw_ocr_labels(annotated, stable_dets)                |
|     |                                                          |
|     +-- return annotated                                       |
|                                                                |
|  3. frame_ready.emit(QImage(annotated))                        |
+----------------------------------------------------------------+
           |
           v
+----------------------------------------------------------------+
|                       主线程 (GUI)                              |
|                                                                |
|  VideoPanel.update_frame(qimage)  -> 显示到屏幕                |
|  Dashboard.update_fps(fps)        -> 更新 FPS 指示             |
|                                                                |
|  用户点击 "验证":                                              |
|    [read_lock]                                                 |
|    validator.compare(analyzer) -> 生成验证报告                  |
|    [/read_lock]                                                |
|    set_missing_links(results['missing_links'])                 |
|    CircuitPage.set_result(report)                              |
|                                                                |
|  用户输入问题:                                                 |
|    context = get_circuit_snapshot()    -- 无锁, 用快照          |
|    LLMWorker(llm, question, context).start()                   |
|    -> response_ready -> ChatPanel.add_message()                |
+----------------------------------------------------------------+
```

---

## 六、文件清单与职责

### 核心架构文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `app_context.py` | ~180 | 服务注册中心 + ReadWriteLock + 线程安全接口 |
| `config.py` | 422 | 8 个 dataclass 配置 + 环境变量覆盖 |
| `launcher.py` | 448 | 生产级启动器: 诊断 + 看门狗 + 字体检测 |

### 感知层 (vision/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `detector.py` | ~250 | YOLOv8 HBB+OBB 双模检测 + 引脚延伸先验 |
| `stabilizer.py` | ~120 | 5 帧滑窗 IoU 投票稳定器 |
| `calibrator.py` | ~740 | 4 级孔洞检测 + RANSAC 网格 + Top-K 候选 |

### 推理层 (logic/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `circuit.py` | ~560 | NetworkX 电路图 + 自动命名 + 电源轨 + 三端建模 |
| `validator.py` | ~850 | VF2++ 图同构 + 4 级诊断 + 6 项拓扑检查 |
| `polarity.py` | ~320 | OBB 几何极性推断 + TO-92 引脚 |

### 认知层 (ai/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `llm_engine.py` | ~600 | Cloud + Local + Rule 三级 LLM |
| `rag_engine.py` | 530 | ChromaDB + text2vec 检索增强 |
| `ocr_engine.py` | 611 | PaddleOCR + EasyOCR + 13 种正则 |

### GUI 层 (gui_qt/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `main_window.py` | ~960 | 主窗口: 路由 + 帧处理 + 引脚约束选择 + 信号连接 |
| `workers.py` | 221 | Video / ModelLoader / LLM 三 QThread |
| `dashboard.py` | 238 | 仪表盘: 状态卡 + 模块健康指示 |
| `chat_panel.py` | 250 | 聊天面板: 气泡 + 快捷按钮 |
| `circuit_page.py` | ~120 | 电路验证页: 7 个操作按钮 |
| `video_panel.py` | 200 | 视频面板: FPS 叠加 + 置信度滑块 |
| `settings_page.py` | 203 | 设置页: 摄像头 / YOLO / LLM |
| `sidebar.py` | 181 | 可折叠侧边导航栏 |
| `styles.py` | 339 | PyDracula QSS 暗色主题 |
| `run.py` | ~76 | GUI 入口 (创建 AppContext) |

### 工具 (tools/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `benchmark.py` | ~430 | 5 项性能基准测试 + JSON 报告导出 |
| `annotate_helper.py` | -- | OBB 标注辅助 |
| `train_obb.py` | -- | 训练脚本 |

---

## 七、Intel 异构计算部署方案

```
+--------------------------------------------------------------+
|               Intel Core Ultra 5 225U (DK-2500)               |
|                                                               |
|  +------------+  +------------+  +------------------------+  |
|  |    CPU      |  |   iGPU     |  |        NPU             |  |
|  |            |  |            |  |                        |  |
|  | PaddleOCR  |  | YOLOv8     |  | Qwen2.5-1.5B INT4     |  |
|  | ChromaDB   |  | OpenVINO   |  | OpenVINO GenAI         |  |
|  | NetworkX   |  | IR 格式    |  | LLM 推理               |  |
|  | PySide6    |  | 960x960    |  |                        |  |
|  | RAG 检索   |  | FP16       |  |                        |  |
|  +------------+  +------------+  +------------------------+  |
|                                                               |
|  OpenVINO Runtime 2024.x 统一调度                             |
+--------------------------------------------------------------+
```

**离线约束**: 竞赛评审环境不能联网, 所有模型和依赖必须预装:
- YOLO 权重: `models/yolov8s-obb_openvino/`
- LLM 权重: `models/qwen2.5_1.5b_ov/` (INT4 量化)
- Embedding: `models/text2vec_chinese/` (预下载)
- PaddleOCR: 预下载模型文件到本地
- ChromaDB: 预构建向量索引到 `knowledge_base/chroma_db/`

---

## 八、配置系统

```python
# config.py 中的 8 个 dataclass

@dataclass RAGConfig         # RAG 知识检索参数
@dataclass VisionConfig      # YOLO 检测参数 (conf, iou, imgsz)
@dataclass CalibrationConfig # 面包板校准参数 (行列数, Blob 参数)
@dataclass CameraConfig      # 摄像头参数 (device_id, backend)
@dataclass LLMConfig         # LLM 参数 (model_path, device, use_cloud)
@dataclass OCRConfig         # OCR 参数 (未来扩展)
@dataclass GUIConfig         # GUI 参数 (窗口大小, 标题)
@dataclass CircuitConfig     # 电路分析 + 引脚遮挡补偿参数
```

**CircuitConfig v3.1 新增字段**:
```python
pin_candidate_k: int = 3              # 每个引脚返回的候选孔洞数量
pin_same_group_penalty: float = 100.0  # 两引脚同导通组惩罚分 (短路)
pin_same_row_penalty: float = 50.0     # 非Wire同行惩罚分
pin_large_span_threshold: int = 10     # 行跨度异常阈值
```

**环境变量覆盖** (优先级: 环境变量 > .env 文件 > 默认值):
```bash
LG_API_KEY=sk-xxx         # DeepSeek API Key
LG_CAMERA_ID=0             # 摄像头编号
LG_OV_DEVICE=GPU           # OpenVINO 设备 (CPU/GPU/NPU)
LG_MODEL_PATH=models/best.pt  # YOLO 模型路径
LG_RAG_ENABLED=true        # 是否启用 RAG
```

---

## 九、可借鉴的开源项目与优化方向

基于与 LabGuardian 架构相似的开源项目分析, 以下是值得学习的模式和具体优化建议。

### 9.1 Ultralytics YOLOv8 -- 推理管线设计

**项目**: `ultralytics/ultralytics` (GitHub ~35K stars)

**可借鉴点 -- Results 封装模式**:

YOLOv8 的 `Results` 类将检测结果封装为统一对象, 支持 `.plot()`, `.to_json()`, `.pandas()` 等链式操作。

当前 LabGuardian 的 `Detection` 是简单 dataclass, 缺少这种富接口。

**优化建议**: 给 `Detection` / `StableDetection` 增加序列化方法:

```python
@dataclass
class Detection:
    # ... 现有字段 ...

    def to_dict(self) -> dict:
        """JSON 序列化 (基准测试 / 日志 / 模板保存)"""

    def draw_on(self, frame: np.ndarray) -> np.ndarray:
        """把标注逻辑从 detector.annotate_frame 移到 Detection 自身"""
```

### 9.2 Supervision (Roboflow) -- 帧处理管线

**项目**: `roboflow/supervision` (GitHub ~25K stars)

**可借鉴点 -- Annotator 装饰器链**:

```python
# Supervision 的设计:
annotated = frame.copy()
annotated = box_annotator.annotate(annotated, detections)
annotated = label_annotator.annotate(annotated, detections)
annotated = trace_annotator.annotate(annotated, detections)
```

每个 Annotator 是独立的、可组合的。LabGuardian 的 `_process_frame` 把检测、分析、标注全混在一起。

**优化建议**: 抽取 `FramePipeline`, 每个阶段是独立的 Stage:

```python
class FramePipeline:
    stages = [
        DetectionStage(detector),
        StabilizationStage(stabilizer),
        CalibrationStage(calibrator),
        AnalysisStage(analyzer, polarity),
        OCRStage(ocr, interval=30),
        AnnotationStage(),  # 画框、画标签、画幽灵线
    ]

    def process(self, frame) -> AnnotatedFrame:
        ctx = FrameContext(frame)
        for stage in self.stages:
            stage.execute(ctx)
        return ctx.annotated
```

好处: 竞赛现场可以按需禁用某个 Stage (如 OCR 太慢就跳过), 也方便 benchmark 单独测量每个 Stage 的耗时。

### 9.3 X-AnyLabeling -- PySide6 + YOLO 集成

**项目**: `CVHub520/X-AnyLabeling` (GitHub ~5K stars)

**可借鉴点 -- 多后端模型管理**:

X-AnyLabeling 用 `ModelManager` 统一管理 ONNX / OpenVINO / TensorRT 三种推理后端, 运行时根据环境自动选择。类似 LabGuardian 的三级 LLM 降级。

**优化建议**: 将 YOLO 检测器也做成多后端:

```python
class ComponentDetector:
    def load(self):
        if openvino_available and has_gpu:
            self._backend = OpenVINOBackend(model_xml)
        elif onnx_available:
            self._backend = ONNXBackend(model_onnx)
        else:
            self._backend = UltralyticsBackend(model_pt)
```

当前 `detector.py` 虽然支持 OpenVINO 格式, 但切换逻辑混在 YOLO 库内部。
显式的后端选择对竞赛调试更友好。

### 9.4 LangChain / LlamaIndex -- RAG 架构

**项目**: `langchain-ai/langchain` (GitHub ~100K stars)

**可借鉴点 -- Retriever + Prompt Template 分离**:

LangChain 的 RAG 链路:
```
Retriever (文档检索) -> ContextFormatter (上下文格式化) -> LLM (生成)
```

当前 LabGuardian 的 `rag_engine.py` 把检索和格式化混在一起。

**优化建议**: 分离 Retriever 和 PromptBuilder:

```python
class PromptBuilder:
    """将电路上下文 + RAG 检索结果 + 用户问题组合为 LLM 提示词"""

    def build(self, question, circuit_ctx, rag_results):
        return f"""你是 LabGuardian 助教。

相关知识:
{self._format_rag(rag_results)}

当前电路:
{circuit_ctx}

学生问题: {question}
"""
```

好处: 可以针对不同场景 (通用问答 / 纠错引导 / 芯片查询) 用不同模板, 提升回答针对性。

### 9.5 OpenVINO Model Server -- 模型生命周期管理

**项目**: `openvinotoolkit/model_server` (GitHub ~600 stars)

**可借鉴点 -- 模型热加载 + 健康检查**:

OVMS 为每个模型维护 `ModelStatus` (LOADING / AVAILABLE / FAILED / UNLOADING),
并通过健康检查接口暴露状态。

**优化建议**: 给 AppContext 加模型状态追踪:

```python
class AppContext:
    def get_health(self) -> dict:
        return {
            "detector": "ok" if self.detector.model else "not_loaded",
            "llm": self.llm.backend_name,
            "ocr": self.ocr.backend_name,
            "calibrator": "calibrated" if self.calibrator.is_calibrated else "pending",
            "rag": f"{self.llm.rag.doc_count} chunks" if self.llm.rag_ready else "disabled",
        }
```

Dashboard 可以直接调用 `ctx.get_health()` 更新模块指示灯。

### 9.6 Frigate NVR -- 事件驱动检测架构

**项目**: `blakeblackshear/frigate` (GitHub ~20K stars)

**可借鉴点 -- 检测区域 + 事件触发**:

Frigate 不是每帧全图检测, 而是定义 "检测区域" (zones), 只在关注区域运行 YOLO。
同时用 "motion detection" 低计算量方式触发高计算量的 YOLO。

**优化建议**: 面包板场景可以做 ROI 优化:

```
1. 校准后已知面包板区域, 只在面包板区域内运行 YOLO
2. 帧间差分判断: 如果面包板区域没有变化, 跳过 YOLO (复用上帧结果)
3. OCR 只在新检测到 IC 类别时触发 (已实现)
```

这对 DK-2500 的算力优化价值巨大。

### 9.7 综合优化路线图

根据以上分析, 按优先级排列的优化方向:

| 优先级 | 优化 | 参考项目 | 预期收益 |
|--------|------|----------|----------|
| P0 | ~~ROI 裁剪~~ (已实现) | Frigate | YOLO 推理速度提升 30-50% |
| P0 | ~~帧间差分跳帧~~ (已实现) | Frigate | 平均 FPS 提升 2-3x |
| P0 | ~~引脚遮挡补偿~~ (已实现) | 自研 | 引脚定位准确率显著提升 |
| P1 | FramePipeline 拆分 | Supervision | 代码可维护性 + 可调试性 |
| P1 | 模型健康检查 API | OVMS | Dashboard 实时反馈 |
| P2 | Detection 富接口 (to_dict/draw_on) | Ultralytics | 代码整洁度 |
| P2 | PromptBuilder 分离 | LangChain | LLM 回答质量 |
| P3 | 多后端检测器 | X-AnyLabeling | DK-2500 部署灵活性 |

---

## 十、目录结构 (最新)

```
LabGuardian/
+-- src_v2/                          主代码目录
|   +-- app_context.py               服务注册中心 + 线程安全
|   +-- config.py                    8 dataclass 配置
|   +-- launcher.py                  生产级启动器
|   +-- run_qt.py                    PySide6 入口
|   +-- build_knowledge_base.py      RAG 知识库 CLI
|   |
|   +-- ai/                          认知层
|   |   +-- llm_engine.py            三级 LLM 引擎
|   |   +-- rag_engine.py            ChromaDB RAG
|   |   +-- ocr_engine.py            PaddleOCR + EasyOCR
|   |
|   +-- vision/                      感知层
|   |   +-- detector.py              YOLOv8 HBB+OBB
|   |   +-- stabilizer.py            多帧投票稳定器
|   |   +-- calibrator.py            面包板 4 级校准
|   |
|   +-- logic/                       推理层
|   |   +-- circuit.py               NetworkX 电路图 + 自动命名
|   |   +-- validator.py             VF2++ 验证 + 拓扑诊断
|   |   +-- polarity.py              极性解析器
|   |
|   +-- gui_qt/                      GUI 层 (PySide6)
|   |   +-- main_window.py           主窗口 (AppContext 驱动)
|   |   +-- workers.py               3 个 QThread 工作线程
|   |   +-- dashboard.py             仪表盘页
|   |   +-- chat_panel.py            聊天页
|   |   +-- circuit_page.py          电路验证页
|   |   +-- video_panel.py           视频页
|   |   +-- settings_page.py         设置页
|   |   +-- sidebar.py               侧边栏
|   |   +-- styles.py                PyDracula 主题
|   |   +-- resources.py             图标常量
|   |   +-- run.py                   GUI 入口
|   |
|   +-- tools/                       辅助工具
|       +-- benchmark.py             性能基准测试 (5 项)
|       +-- annotate_helper.py       OBB 标注
|       +-- train_obb.py             训练脚本
|       +-- fix_dataset.py           数据集修正
|       +-- replicate_labels.py      标签复制
|
+-- models/                          模型资产 (~2.1G)
+-- dataset/                         训练数据
+-- knowledge_base/                  RAG 知识库
|   +-- builtin/                     6 篇内置文档
|   +-- chroma_db/                   向量索引
|   +-- user_docs/                   用户文档
+-- scripts/                         训练/部署脚本
+-- logs/                            运行日志
+-- start.sh                         Linux 启动脚本
+-- launch.bat                       Windows 启动脚本
```

---

## 十一、术语表

| 术语 | 含义 |
|------|------|
| HBB | Horizontal Bounding Box, 水平矩形检测框 |
| OBB | Oriented Bounding Box, 旋转矩形检测框 |
| VF2++ | 图同构匹配算法 (Juttner & Madarasi, 2018) |
| GED | Graph Edit Distance, 图编辑距离 |
| TTFT | Time To First Token, 首 token 延迟 |
| Ghost Wire | 幽灵线, 缺失连接的 AR 标注引导 |
| Golden Reference | 金标准电路, 教师预设的正确电路模板 |
| Net 节点 | 面包板上同一行/列的电气网络 (等电位) |
| Comp 节点 | 图中代表一个电子元件的节点 |
| 导通组 | 面包板上物理导通的 5 孔组: Row{n}_L (a-e) 或 Row{n}_R (f-j) |
| 引脚延伸先验 | 从 OBB/HBB 边缘沿长轴向外延伸, 补偿元件遮挡的引脚位置偏差 |
| Top-K 候选 | 返回 K 个最近孔洞而非单一最近, 提供候选冗余 |
| 约束排序 | 基于面包板导通规则 (短路惩罚/跨度/导通组) 从多候选中选最优引脚对 |

---

## 十二、版本变更记录

### v3.1 (2026-02-24) — 逻辑层强化 + 引脚遮挡补偿

**推理层**:
- 移除 `schematic.py` (schemdraw 依赖不必要)
- `circuit.py`: 统一类型归一化 `norm_component_type()`, 元件自动命名, 三端元件 E-B/B-C 结边建模, 电源轨 PWR_PLUS/PWR_MINUS 节点, 结构化中文电路描述, 自动问题检测
- `validator.py`: 新增 `diagnose()` 静态方法 (6 项拓扑检查: LED无电阻/极性未知/短路/缺引脚/悬空/断路)
- `polarity.py`: `_norm_type()` 委托给共用的 `norm_component_type()`

**感知层**:
- `detector.py`: 引脚延伸先验 (按元件类型沿长轴向外补偿 OBB 遮挡)
- `calibrator.py`: 新增 `top_k_holes()` 和 `frame_pixel_to_logic_candidates()` 多候选映射

**GUI 层**:
- `main_window.py`: 新增 `_compute_obb_angle()` (OBB 旋转角度计算), `_pick_best_pin_pair()` (约束排序选择), 电路构建循环改为候选+约束管线

**认知层**:
- `llm_engine.py`: 实现 `_summarize_circuit()` (从电路上下文生成中文概要)

**配置**:
- `config.py`: 新增 `pin_candidate_k`, `pin_same_group_penalty`, `pin_same_row_penalty`, `pin_large_span_threshold`

### v3.0 (2026-02-22) — AppContext + 线程安全重构

- 引入 AppContext 服务注册中心
- ReadWriteLock + 快照机制
- ROI 裁剪 + 帧间差分跳帧

---

*文档版本: v3.1 | 逻辑层强化 + 引脚遮挡补偿 | 2026-02-24*
