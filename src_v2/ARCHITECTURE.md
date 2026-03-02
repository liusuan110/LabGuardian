# LabGuardian v5 -- 系统架构文档

> **版本**: v5.0 (Image-only + 4 轨电源轨 + 导线骨架 + IC 多引脚)
> **更新日期**: 2026-03-01
> **总代码量**: ~10,900 行 / 41 个 Python 源文件
> **竞赛平台**: Intel Core Ultra 5 225U (DK-2500)

---

## 一、系统定位

LabGuardian 是一个**边缘端全离线**的电子实验智能助教系统。核心链路:

```
手机俯拍 (1-3 张高清图片)
   |
   v
[YOLO 元件检测 (9 类)]  ------>  [面包板校准 + 坐标映射]
   |                                     |
   v                                     v
[Wire 骨架端点精炼]              [电路拓扑图 (NetworkX)]
[+ 导线颜色分类]                 [+ IC hub 多引脚模型]
   |                                     |
   v                                     v
[多图 IoU 融合]                  [VF2++ 电路验证]
   |                                     |
   v                                     v
[OCR IC 丝印 → 引脚数据库]      [极性诊断]
   |                                     |
   v                                     v
[RAG 知识检索]  <---------  [自然语言网表上下文]
   |
   v
[LLM 智能问答]                  [学生标注电源轨]
   |                                     |
   v                                     v
[PySide6 GUI 展示]              [结构化分析报告]
```

---

## 二、四层架构

```
+------------------------------------------------------------------+
|                         GUI 层 (gui_qt/)                          |
|  PySide6 无边框窗口 / PyDracula 暗色主题 / 5 页路由               |
|  UploadPage / Dashboard / ChatPanel / CircuitPage / Settings      |
|  QThread Workers: ImageAnalysisWorker / ModelLoaderWorker / LLM   |
+------------------------------------------------------------------+
|                        认知层 (ai/)                                |
|  LLMEngine: Cloud(DeepSeek) -> Local(OpenVINO NPU) -> Rules      |
|  RAGEngine: ChromaDB + text2vec-base-chinese                      |
|  OCREngine: PaddleOCR(主) -> EasyOCR(备) -> 无OCR(兜底)          |
+------------------------------------------------------------------+
|                        推理层 (logic/)                             |
|  CircuitAnalyzer: NetworkX 图 + IC hub 多引脚 + 4 轨电源轨       |
|  CircuitValidator: VF2++ 图同构 + 4 级诊断 + 6 项拓扑检查         |
|  PolarityResolver: OBB 极性 + TO-92 + 电位器 + IC 引脚           |
|  ICPinoutDB: LM324 / LM358 / NE5532 引脚查表                     |
+------------------------------------------------------------------+
|                        感知层 (vision/)                            |
|  ComponentDetector: YOLOv8 HBB+OBB 双模 (9 类, 含 IC/变阻器)     |
|  ImageAnalyzer: 多图分析 + IoU 融合 + 结构化报告                  |
|  WireAnalyzer: 骨架化端点检测 + HSV 颜色分类                     |
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
                     AppContext (app_context.py, 241 行)
                    /      |       |       \
                   /       |       |        \
          感知层服务    推理层服务   认知层服务   线程安全设施
          ---------    ---------   ---------   -----------
          detector     analyzer    llm         ReadWriteLock
          calibrator   polarity    ocr         _ocr_lock
                       validator               _desc_lock
                       ic_pinout
```

### 3.1 为什么引入 AppContext

| 问题 | AppContext 之前 | AppContext 之后 |
|------|----------------|-----------------|
| 模块构造 | MainWindow.__init__ 直接 new 全部模块 | AppContext 统一构造, MainWindow 只接收 ctx |
| 可测试性 | 必须启动 GUI 才能测试业务逻辑 | `ctx = AppContext()` 即可独立测试 |
| 线程安全 | 无任何保护, 竞态条件 | ReadWriteLock + 互斥锁 + 快照机制 |
| 替换实现 | 改 MainWindow 代码 | 改 AppContext 构造即可 |
| Benchmark | 无法脱离 GUI 运行性能测试 | `tools/benchmark.py` 直接使用 AppContext |

### 3.2 生命周期 (v5.0 Image-only)

```python
# gui_qt/run.py
ctx = AppContext()              # 1. 创建全部服务实例
window = MainWindow(ctx=ctx)    # 2. 注入到 GUI
window.show()                   # 3. GUI 启动, 触发模型加载
# ... 学生上传图片 ...
# ImageAnalysisWorker 在后台线程分析, 完成后信号更新 UI
# closeEvent -> 清理
```

### 3.3 线程安全模型 (v5.0)

```
                    +-------------------------+
                    |  ImageAnalysisWorker     |
                    |        (QThread)         |
                    |                         |
                    |  ImageAnalyzer.analyze() |
                    |    detector.detect()     |  <-- 只读, 无需锁
                    |    wire_analyzer.analyze |
                    |    calibrator.map()      |
                    |    analyzer.reset()      |
                    |    analyzer.add_comp()   |  <-- 内部无竞争
                    |    ocr.recognize_chip()  |
                    |    polarity_resolver()   |
                    |    --> AnalysisResult     |
                    +-------------------------+
                              |
                    (finished Signal)
                              |
                    +-------------------------+
                    |      主线程 (GUI)         |
                    |                         |
                    |  UploadPage.show_result()|
                    |  update_circuit_snapshot()|
                    |                         |
                    |  用户点击 "验证":         |
                    |    validator.compare()   |
                    |                         |
                    |  用户标注电源轨:          |
                    |    ctx.set_rail_assign() |
                    +-------------------------+
                              |
                    +-------------------------+
                    |    LLMWorker 线程        |
                    |                         |
                    |  llm.ask(question,      |
                    |    context=snapshot)     |  <-- 用快照, 完全无锁
                    +-------------------------+
```

**快照机制**:
`update_circuit_snapshot()` 生成 `analyzer.get_circuit_description()` 的字符串副本,
并附加 `CircuitValidator.diagnose()` 的独立诊断结果。
LLM 问答通过 `get_circuit_snapshot()` 读取, 完全不需要获取读写锁。

### 3.4 电源轨管理 API (v5.0 新增)

```python
# 学生在 CircuitPage 标注电源轨
ctx.set_rail_assignment("RAIL_TOP_1", "VCC +5V")
ctx.set_rail_assignment("RAIL_BOTTOM_2", "GND")

# 查询未标注的活跃轨道 (有导线连接但未标注)
unassigned = ctx.get_unassigned_active_rails()
# → ["RAIL_TOP_2", "RAIL_BOTTOM_1"]

# 清除所有标注
ctx.clear_rail_assignments()
```

---

## 四、各层详细设计

### 4.1 感知层 (vision/)

#### ComponentDetector (`detector.py`, 272 行)

```
输入: BGR 图片 (np.ndarray)
   |
   v
YOLO predict(image, conf, iou, imgsz)
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

**9 类元件检测**:
```python
COMPONENT_CLASSES = [
    "CAPACITOR", "DIODE", "LED", "RESISTOR", "Push_Button", "Wire",
    "TRANSISTOR", "IC_DIP", "POTENTIOMETER"
]
```

**引脚延伸先验**:

| 元件类型 | 延伸比例 (× 长边) | 物理原因 |
|----------|-------------------|----------|
| RESISTOR | 0.10 | 引线细长, 略超出色环体 |
| LED | 0.08 | 顶视时引脚被圆顶遮挡 |
| DIODE | 0.10 | 类似电阻 |
| CAPACITOR | 0.08 | 引脚较短 |
| Wire | 0.02 | 端点几乎就是连接点 |
| Push_Button | 0.06 | 引脚在底部 |
| TRANSISTOR | 0.10 | TO-92 引脚间距 |
| IC_DIP | 0.02 | DIP 引脚紧贴芯片体 |
| POTENTIOMETER | 0.08 | 电位器引脚 |

**Detection 数据类 (v5.0)**:
```python
@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    pin1_pixel: Tuple[int, int]       # 引脚1 像素坐标
    pin2_pixel: Tuple[int, int]       # 引脚2 像素坐标
    is_obb: bool = False
    obb_corners: Optional[np.ndarray] = None
    wire_color: Optional[str] = None  # v5.0 新增: 导线颜色
```

#### WireAnalyzer (`wire_analyzer.py`, 249 行) — v5.0 新增

**导线骨架分析器**: 对 YOLO 检测到的 Wire 区域执行端点精确定位和颜色分类。

```
输入: frame (全图) + wire bbox
  |
  v
1. Safe-crop (bbox + 20% padding)
  |
  v
2. HSV 颜色分割
   - 排除背景: 低饱和度 + 高亮度 (面包板白色)
   - 识别彩色: 饱和度 > 40
   - 识别黑色: 亮度 < 60
   - 导线 mask = 非背景 ∩ (彩色 ∪ 黑色)
  |
  v
3. 形态学清理
   - 闭运算 (填断裂) → 开运算 (去噪)
   - 保留最大连通域
  |
  v
4. 骨架化
   - 优先: cv2.ximgproc.thinning (Zhang-Suen)
   - 回退: 迭代形态学腐蚀
  |
  v
5. 端点检测
   - 扫描骨架像素, 8-邻域连接度 = 1 → 端点
   - 端点 > 2: 凸包加速选最远一对
   - 端点 < 2: 回退选最远骨架像素对
  |
  v
6. 颜色分类
   - 对导线 mask 区域的 HSV 值分布投票
   - 支持: red / blue / green / yellow / orange / black / white
   - 阈值: 该颜色像素占比 > 15%
  |
  v
输出: ((端点1_px, 端点2_px), "red")
```

**安全降级**: 任何步骤失败 → 保留原始 OBB 端点, wire_color = None。

#### ImageAnalyzer (`image_analyzer.py`, 579 行) — v5.0 新增

**图片分析引擎 — 替代 FramePipeline, 用于 Image-only 架构**。

```python
@dataclass
class AnalysisResult:
    annotated_image: np.ndarray    # 标注后的图片
    detections: list               # 融合后的检测列表
    report: str                    # 结构化文本报告
    component_count: int           # 元件数量
    net_count: int                 # 电气网络数量
    ocr_results: Dict[str, str]    # IC 型号识别结果
    issues: List[str]              # 检测到的问题
```

**分析管线**:

```
输入: 1-3 张 BGR 图片 + conf + imgsz
  |
  +-- 每张图片:
  |     detector.detect(img, conf, imgsz)
  |     → List[Detection]
  |     wire_analyzer.analyze_wire() (仅 Wire 类型)
  |     → 替换 pin1/pin2, 填充 wire_color
  |
  +-- 多图融合 (_fuse_detections):
  |     图1 = 基准
  |     图2,3 与基准做 IoU 匹配 (阈值 0.3)
  |     匹配到: 取置信度更高的
  |     未匹配: 加入结果 (新元件, 被遮挡的)
  |
  +-- 校准 + 坐标映射:
  |     calibrator.frame_pixel_to_logic_candidates(pin, k=3)
  |     → _pick_best_pin_pair() 约束选择
  |
  +-- 电路建模 (_build_circuit):
  |     2-pin 元件: analyzer.add_component(loc1, loc2)
  |     3-pin 元件: _find_3pin_middle() → pin3
  |     IC_DIP: _build_ic_component() → OCR → 引脚数据库 → 多引脚映射
  |
  +-- 极性解析 + 电路验证:
  |     polarity_resolver.enrich()
  |     CircuitValidator.diagnose()
  |
  +-- 报告生成 + 标注绘制:
        _generate_report() → 结构化文本
        _annotate() → 带框/标签的标注图
  |
  v
输出: AnalysisResult
```

**_pick_best_pin_pair (引脚约束选择)**:
从每引脚 K 个候选孔洞中选最佳组合:
- 短路惩罚: 两引脚在同一导通组 → +100 分
- 同行惩罚: 非 Wire 元件两引脚同行 → +50 分
- 大跨度惩罚: 行跨度 > 10 → 额外距离
- Wire 豁免: 不受同组/同行惩罚

#### BreadboardCalibrator (`calibrator.py`, 671 行)

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
  nearest_hole()                 → 单一最近孔洞
  top_k_holes(k=3)              → K 个最近孔洞 + 距离分数
  frame_pixel_to_logic_candidates(k) → K 个候选逻辑坐标
```

### 4.2 推理层 (logic/)

#### CircuitAnalyzer (`circuit.py`, 758 行)

**核心数据结构: NetworkX 图**

```
节点 = 面包板导通组:
  Row{n}_L      — 第 n 行左侧 (a-e 五孔导通)
  Row{n}_R      — 第 n 行右侧 (f-j 五孔导通)
  RAIL_TOP_1    — 顶部外侧电源轨 (v5.0: 4 条独立轨道)
  RAIL_TOP_2    — 顶部内侧电源轨
  RAIL_BOTTOM_1 — 底部内侧电源轨
  RAIL_BOTTOM_2 — 底部外侧电源轨
  {name}_hub    — IC 元件中心节点 (v5.0: IC 多引脚)

边 = 元件连接:
  属性: component(名称), type(类型), polarity, confidence,
        pin1_role, pin2_role, junction("EB"/"BC" 仅三端),
        ic_pin_number(仅 IC)
```

**元件数据类 (v5.0)**:
```python
@dataclass
class CircuitComponent:
    name: str              # "R1", "LED1", "U1"
    type: str              # "RESISTOR", "IC_DIP"
    pin1_loc: (str, str)   # 引脚1 坐标 (Row, Col)
    pin2_loc: (str, str)   # 引脚2 坐标
    polarity: Polarity
    pin_roles: Dict[int, PinRole]
    confidence: float
    pin3_loc: (str, str)   # 第三引脚 (三极管/电位器)
    pin_locs: List[...]    # IC 全部引脚 [(row,col,pin_name,PinRole), ...]
    ic_model: str          # IC 型号 ("LM324")
```

**引脚角色 (PinRole) 枚举**:
```python
class PinRole(Enum):
    GENERIC          # 通用 (无特殊含义)
    ANODE / CATHODE  # 阳极 / 阴极 (二极管/LED)
    BASE / COLLECTOR / EMITTER   # 三极管 B/C/E
    VCC / GND        # 电源
    POSITIVE / NEGATIVE   # 正极 / 负极 (电容)
    WIPER            # 滑动触点 (电位器)
    TERMINAL_A / TERMINAL_B  # 端子 (电位器)
    IC_PIN           # 通用 IC 引脚
    IC_VCC / IC_GND  # IC 电源
    IC_OUTPUT        # IC 输出
    IC_INPUT_INV     # IC 反相输入
    IC_INPUT_NON     # IC 同相输入
```

**IC Hub 节点模型 (v5.0 新增)**:

对于 DIP 封装 IC (如 LM324 14 脚), 使用 hub 节点连接所有引脚:

```
  Row5_L ──[pin1]── U1_hub ──[pin14]── Row5_R
  Row6_L ──[pin2]── U1_hub ──[pin13]── Row6_R
  Row7_L ──[pin3]── U1_hub ──[pin12]── Row7_R
    ...              ...                 ...
  Row11_L ──[pin7]── U1_hub ──[pin8]── Row11_R
```

**4 轨道电源轨模型 (v5.0 新增)**:

```python
rail_track_rows = {
    "RAIL_TOP_1":    (1,),    # 顶部外侧
    "RAIL_TOP_2":    (2,),    # 顶部内侧
    "RAIL_BOTTOM_1": (64,),   # 底部内侧
    "RAIL_BOTTOM_2": (65,),   # 底部外侧
}
```

每条轨道由学生独立标注用途:
```python
analyzer.set_rail_assignment("RAIL_TOP_1", "VCC +5V")
analyzer.set_rail_assignment("RAIL_BOTTOM_2", "GND")
# _identify_power_nets() 读取标注 → 建立电源网络
```

**三端元件边建模**:
```
  E ──[EB结]── B ──[BC结]── C
  node1       node3       node2
```
适用于: 三极管 (NPN/PNP), 电位器 (Terminal_A / Wiper / Terminal_B)

**元件自动命名**:
```python
_TYPE_PREFIX = {
    "RESISTOR": "R", "LED": "LED", "DIODE": "D",
    "CAPACITOR": "C", "WIRE": "W", "PUSH_BUTTON": "SW",
    "NPN": "Q", "PNP": "Q", "TRANSISTOR": "Q",
    "OPAMP": "U", "IC_555": "U", "IC_DIP": "U",
    "POTENTIOMETER": "VR", "POWER": "V",
}
```

#### CircuitValidator (`validator.py`, 726 行)

**四级诊断管线**:

```
L0  快速预检: 元件类型计数比较 (O(n))
L1  全图同构: VF2++ (度序列预拒 → 节点/边匹配)
L2  子图同构: 检查进度 = matched / total
L2.5 极性专项: 无极性 VF2++ → 仅极性错误诊断
L3  GED: 小图精确解 / 大图三维度近似
```

**`diagnose()` — 6 项拓扑检查**:

| 检查项 | 严重程度 |
|--------|----------|
| LED 无限流电阻 | 高 (烧毁风险) |
| 极性未知 | 中 |
| 同组短路 | 高 |
| 三极管缺引脚 | 高 |
| 悬空节点 | 中 |
| 断路子图 | 低 |

#### PolarityResolver (`polarity.py`, 295 行)

```
                  元件类型
                /    |    \      \        \
POLARIZED   NON_POLAR  THREE_PIN  IC_DIP  POTENTIOMETER
    |           |         |          |        |
OBB 长轴     NONE     视觉 pin3    检查     与三极管
方向推断   (无极性)   → E/B/C    pin_locs  同逻辑
```

#### IC 引脚数据库 (`ic_pinout_db.py`, 198 行) — v5.0 新增

| IC 型号 | 封装 | 引脚数 | 别名 |
|---------|------|--------|------|
| LM324 | DIP-14 | 14 | LM324N, LM324AN, LM2902 |
| LM358 | DIP-8 | 8 | LM358N, LM358P, LM358D, LM358AN |
| NE5532 | DIP-8 | 8 | NE5532N, NE5532P, SA5532 |

查找: 精确匹配 → 别名 → 前缀匹配 → 反向前缀

### 4.3 认知层 (ai/)

#### LLMEngine (`llm_engine.py`, 640 行)

三级降级: Cloud (DeepSeek) → Local (OpenVINO GenAI, NPU) → Rule-based

#### RAGEngine (`rag_engine.py`, 455 行)

text2vec-base-chinese → ChromaDB → Top-K 检索 → LLM 提示词

#### OCREngine (`ocr_engine.py`, 499 行)

PaddleOCR → EasyOCR → 13 种正则 + 错字修正 → RAG 查询 + 引脚数据库

### 4.4 GUI 层 (gui_qt/)

#### 页面路由 (v5.0)

```
Sidebar (导航)
  |
  +-- home     →  UploadPage (图片上传 + 分析控制) + Dashboard
  +-- results  →  标注结果图 + ChatPanel (侧栏聊天)
  +-- chat     →  ChatPanel (全屏聊天)
  +-- circuit  →  CircuitPage (验证/模板/网表 + 电源轨标注)
  +-- settings →  SettingsPage (YOLO/LLM 配置)
```

#### UploadPage (`upload_page.py`, 296 行)

```
+-------------------------------------------------------+
|  [图片1]      [图片2]      [图片3]                     |
|  (缩略图)     (缩略图)     (空槽位)                    |
|                                                       |
|  置信度: ====●=================== 0.25               |
|  分辨率: [1280 ▼]                                     |
|                                                       |
|  [📐 校准] [🔍 开始分析] [🗑️ 清空]                    |
|  ■■■■■■■■■■■■□□□□□□ 分析中...                         |
|                                                       |
|  +-------------------+  +---------------------------+ |
|  |   标注结果图      |  |  === 分析报告 ===          | |
|  |   (bbox + 坐标)   |  |  元件: 8 (R×3, LED×2...) | |
|  +-------------------+  +---------------------------+ |
+-------------------------------------------------------+
```

#### CircuitPage 电源轨标注 UI (v5.0 新增)

```
+-- 电源轨配置 ----------------------------------+
|  顶部外侧轨:  ● 未连接  [VCC +5V ▼] [      ] |
|  顶部内侧轨:  ● 已标注  [GND     ▼] [      ] |
|  底部内侧轨:  ● 未连接  [选择... ▼] [      ] |
|  底部外侧轨:  ● 已标注  [VCC +5V ▼] [      ] |
|              [清除所有轨道标注]                  |
+------------------------------------------------+
```

#### 信号槽架构 (v5.0)

```
UploadPage.analyze_requested → MainWindow._start_analysis()
  → ImageAnalysisWorker → finished(AnalysisResult) → show_result()

CircuitPage.rail_assigned(str,str) → ctx.set_rail_assignment()
CircuitPage.rail_cleared → ctx.clear_rail_assignments()
CircuitPage.validate_requested → MainWindow._validate_circuit()

ChatPanel.message_sent(str) → LLMWorker → response_ready(str)
```

---

## 五、数据流 -- 一次图片分析的完整旅程

```
+----------------------------------------------------------------+
|                  ImageAnalysisWorker 线程                       |
|                                                                |
|  1. images = upload_page.get_images()  (1-3 张 BGR ndarray)    |
|  2. ImageAnalyzer.analyze(images, conf=0.25, imgsz=1280)      |
|     |                                                          |
|     +-- 每张图: YOLO 检测 → Wire 骨架精炼                      |
|     +-- 多图 IoU 融合                                          |
|     +-- 校准 + 坐标映射 (Top-K 候选 → 约束选择)                |
|     +-- 电路建模 (2-pin / 3-pin / IC 多引脚)                   |
|     +-- OCR → ic_pinout_db 查表                                |
|     +-- 极性解析 + CircuitValidator.diagnose()                  |
|     +-- 标注绘制 + 结构化报告生成                               |
|     +-- return AnalysisResult                                  |
|                                                                |
|  3. finished.emit(result)                                      |
+----------------------------------------------------------------+
           |
           v
+----------------------------------------------------------------+
|                       主线程 (GUI)                              |
|                                                                |
|  upload_page.show_result(result)  → 标注图 + 报告              |
|  update_circuit_snapshot()        → LLM 问答可用               |
|  检查未标注电源轨 → 提示学生标注                                |
|                                                                |
|  用户标注电源轨 → ctx.set_rail_assignment()                    |
|  用户验证电路 → validator.compare() → 差异报告                |
|  用户提问 → LLMWorker(snapshot) → ChatPanel                    |
+----------------------------------------------------------------+
```

---

## 六、文件清单与职责

### 核心架构文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `app_context.py` | 241 | 服务注册中心 + 线程安全 + 电源轨管理 |
| `config.py` | 425 | 9 个 dataclass 配置 + 9 类元件 + 4 轨道模型 |
| `launcher.py` | 439 | 生产级启动器: 诊断 + 看门狗 + 字体检测 |

### 感知层 (vision/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `detector.py` | 272 | YOLOv8 HBB+OBB 双模 (9 类) |
| `image_analyzer.py` | 579 | 图片分析引擎 (多图融合 + IC 多引脚) |
| `wire_analyzer.py` | 249 | 导线骨架端点 + 颜色分类 |
| `calibrator.py` | 671 | 4 级孔洞检测 + RANSAC 网格 |
| `stabilizer.py` | 103 | 多帧投票稳定器 (保留兼容) |

### 推理层 (logic/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `circuit.py` | 758 | NetworkX 电路图 + IC hub + 4 轨电源轨 |
| `validator.py` | 726 | VF2++ 验证 + 拓扑诊断 |
| `polarity.py` | 295 | 极性解析 (含电位器/IC) |
| `ic_pinout_db.py` | 198 | IC 引脚数据库 |

### 认知层 (ai/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `llm_engine.py` | 640 | 三级 LLM 引擎 |
| `rag_engine.py` | 455 | ChromaDB RAG |
| `ocr_engine.py` | 499 | PaddleOCR + EasyOCR |

### GUI 层 (gui_qt/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `main_window.py` | 617 | 主窗口 (v5.0 Image-only) |
| `upload_page.py` | 296 | 图片上传 + 分析页 |
| `workers.py` | 430 | ImageAnalysis + ModelLoader + LLM |
| `circuit_page.py` | 214 | 电路验证 + 电源轨标注 |
| `dashboard.py` | 192 | 仪表盘 |
| `chat_panel.py` | 208 | 聊天面板 |
| `settings_page.py` | 163 | 设置页 |
| `sidebar.py` | 148 | 侧边栏 |
| `styles.py` | 319 | PyDracula 主题 |
| `calibration.py` | 102 | 校准辅助 |
| `toast.py` | 109 | Toast 通知 |
| `resources.py` | 46 | 图标常量 |
| `run.py` | 61 | GUI 入口 |

### 工具 (tools/)

| 文件 | 行数 | 职责 |
|------|------|------|
| `benchmark.py` | 513 | 5 项性能基准 |
| `annotate_helper.py` | 92 | OBB 标注辅助 |
| `train_obb.py` | 45 | 训练脚本 |

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
|  | PySide6    |  | 1280x1280  |  |                        |  |
|  | WireAnalyzer|  | FP16       |  |                        |  |
|  +------------+  +------------+  +------------------------+  |
|                                                               |
|  OpenVINO Runtime 2024.x 统一调度                             |
+--------------------------------------------------------------+
```

**离线约束**: 所有模型和依赖必须预装:
- YOLO 权重: `models/` (9 类, OpenVINO IR)
- LLM 权重: `models/qwen2.5_1.5b_ov/` (INT4)
- Embedding: `models/text2vec_chinese/`
- PaddleOCR: 预下载模型
- ChromaDB: `knowledge_base/chroma_db/`
- IC 引脚数据库: 内置于代码 (无外部依赖)

---

## 八、配置系统

```python
@dataclass RAGConfig         # RAG 检索参数
@dataclass VisionConfig      # YOLO 参数 (conf, iou, imgsz)
@dataclass CalibrationConfig # 面包板校准参数
@dataclass CameraConfig      # 摄像头参数
@dataclass LLMConfig         # LLM 降级策略
@dataclass OCRConfig         # OCR 参数
@dataclass GUIConfig         # GUI 参数
@dataclass ClassroomConfig   # 课堂模式
@dataclass CircuitConfig     # 电路 + 引脚 + 4 轨电源轨 + IC
```

**环境变量覆盖**:
```bash
LG_API_KEY=sk-xxx           LG_CAMERA_ID=0
LG_OV_DEVICE=GPU            LG_MODEL_PATH=models/best.pt
LG_RAG_ENABLED=true         LG_COMPETITION_MODE=true
```

---

## 九、目录结构 (v5.0)

```
LabGuardian/
├── src_v2/
│   ├── app_context.py
│   ├── config.py
│   ├── launcher.py
│   ├── ARCHITECTURE.md
│   ├── CHANGELOG_2026-03-01.md
│   │
│   ├── ai/
│   │   ├── llm_engine.py
│   │   ├── rag_engine.py
│   │   └── ocr_engine.py
│   │
│   ├── vision/
│   │   ├── detector.py
│   │   ├── image_analyzer.py      ← NEW
│   │   ├── wire_analyzer.py       ← NEW
│   │   ├── calibrator.py
│   │   └── stabilizer.py
│   │
│   ├── logic/
│   │   ├── circuit.py
│   │   ├── validator.py
│   │   ├── polarity.py
│   │   └── ic_pinout_db.py        ← NEW
│   │
│   ├── gui_qt/
│   │   ├── main_window.py         ← REWRITTEN
│   │   ├── upload_page.py         ← NEW
│   │   ├── workers.py
│   │   ├── circuit_page.py
│   │   ├── dashboard.py
│   │   ├── chat_panel.py
│   │   ├── settings_page.py
│   │   ├── sidebar.py
│   │   ├── calibration.py
│   │   ├── toast.py
│   │   ├── styles.py
│   │   ├── resources.py
│   │   ├── run.py
│   │   ├── frame_pipeline.py      (弃用)
│   │   └── video_panel.py         (弃用)
│   │
│   └── tools/
│       ├── benchmark.py
│       ├── annotate_helper.py
│       ├── train_obb.py
│       ├── fix_dataset.py
│       └── replicate_labels.py
│
├── models/
├── dataset/
├── knowledge_base/
├── scripts/
├── logs/
├── start.sh
└── launch.bat
```

---

## 十、术语表

| 术语 | 含义 |
|------|------|
| HBB / OBB | Horizontal / Oriented Bounding Box |
| VF2++ | 图同构匹配算法 |
| GED | Graph Edit Distance, 图编辑距离 |
| Hub 节点 | IC 多引脚的中心虚拟节点 |
| Golden Reference | 教师预设的正确电路模板 |
| 导通组 | 面包板上 5 孔组: Row{n}_L (a-e) / Row{n}_R (f-j) |
| 骨架化 | 二值图迭代腐蚀至单像素宽 (Zhang-Suen) |
| IoU 融合 | 多图检测结果 Intersection over Union 合并 |
| 电源轨标注 | 学生手动指定轨道用途 (VCC/GND/电压) |
| DIP | Dual In-line Package, 双列直插封装 |

---

## 十一、版本变更记录

### v5.0 (2026-03-01) — Image-only + 导线骨架 + IC 多引脚 + 元件扩展

- Image-only 架构迁移 (UploadPage + ImageAnalyzer)
- WireAnalyzer: 骨架端点 + 颜色分类
- 9 类元件 (+TRANSISTOR, IC_DIP, POTENTIOMETER)
- IC 引脚数据库 (LM324/LM358/NE5532) + hub 多引脚模型
- 4 轨道电源轨 + 学生主动标注 (删除投票推断)
- PinRole 扩展 (电位器 + IC)

### v3.2 (2026-02-24) — 电源轨推断 + 三极管视觉检测

### v3.1 (2026-02-24) — 逻辑层强化 + 引脚遮挡补偿

### v3.0 (2026-02-22) — AppContext + 线程安全重构

---

*文档版本: v5.0 | 2026-03-01*
