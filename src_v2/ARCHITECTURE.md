# LabGuardian v5.1 -- 系统架构与项目介绍

> **版本**: v5.1 (Image-only + 视觉引脚占用检测 + 4 轨电源轨 + 导线骨架 + IC 多引脚)
> **更新日期**: 2026-03-03
> **总代码量**: ~11,500 行 / 43 个 Python 源文件
> **竞赛平台**: Intel Core Ultra 5 225U (DK-2500)
> **团队**: 蓝桥杯 Intel Cup 参赛队

---

## 一、系统定位与项目背景

LabGuardian 是一个**边缘端全离线**的电子实验智能助教系统，面向高校电子工程实验教学场景。系统通过手机俯拍面包板照片，自动识别电子元件、追踪引脚插入孔位、建立电路拓扑并与标准电路对比验证，实现**无人值守的电路搭建智能检查与指导**。

### 1.1 解决的核心问题

| 痛点 | LabGuardian 解决方案 |
|------|---------------------|
| 实验教师无法逐一检查每位同学的面包板接线 | YOLO 目标检测自动识别 9 类元件 + 网格校准精确定位引脚 |
| 学生不确定自己的电路哪里接错 | NetworkX 图结构建模 + VF2++ 同构验证 + 自然语言诊断报告 |
| 远程/大班教学场景下实验指导困难 | 拍照即分析 + LLM 智能问答 + 教师端仪表盘实时监控 |
| 竞赛要求全离线运行 | OpenVINO 异构部署 (CPU + iGPU + NPU), 全部模型本地推理 |

### 1.2 核心技术链路

```
手机俯拍 (1-3 张高清图片)
   |
   v
[YOLO 元件检测 (9 类, HBB+OBB)]  ------>  [面包板校准 + 网格坐标映射]
   |                                              |
   v                                              v
[Wire 骨架端点精炼]                       [视觉引脚-孔洞占用检测]
[+ 导线颜色分类]                          [+ 几何候选约束匹配]
   |                                              |
   v                                              v
[多图 IoU 融合]                           [电路拓扑图 (NetworkX)]
   |                                       [+ IC hub 多引脚模型]
   v                                              |
[OCR IC 丝印 → 引脚数据库]               [VF2++ 电路验证]
   |                                       [+ 极性诊断 + Ghost Wire]
   v                                              |
[RAG 知识检索]  <-------------------  [自然语言网表上下文]
   |
   v
[LLM 智能问答]                           [学生标注电源轨]
   |                                              |
   v                                              v
[PySide6 GUI 展示]                        [结构化分析报告]
```

---

## 二、四层架构设计

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
|  PinHoleDetector: 视觉孔洞占用检测 + 自适应特征统计              |
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
                       validator               _ghost_lock
                       ic_pinout               _desc_lock
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

### 3.3 线程安全模型

```
                    +-------------------------+
                    |  ImageAnalysisWorker     |
                    |        (QThread)         |
                    |                         |
                    |  ImageAnalyzer.analyze() |
                    |    detector.detect()     |  <-- 只读, 无需锁
                    |    wire_analyzer.analyze |
                    |    calibrator.map()      |
                    |    pin_hole_detector     |  <-- v5.1 新增
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
                    |  UploadPage.show_result()
                    |  update_circuit_snapshot()
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

### 3.4 电源轨管理 API

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

#### 4.1.1 ComponentDetector (`detector.py`, 272 行)

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

**Detection 数据类**:
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
    wire_color: Optional[str] = None  # 导线颜色
```

#### 4.1.2 PinHoleDetector (`pin_hole_detector.py`, ~420 行) — v5.1 核心新增

**视觉引脚-孔洞占用检测器** — 通过图像分析直接判断哪些面包板孔洞被元件引脚占用。

**核心思路**:
面包板上大多数孔洞是空的，呈现为深色小圆洞。当元件引脚插入后，孔洞的视觉特征发生显著变化：
- 中心亮度升高（金属反光取代暗空洞）
- 中心-周围对比度降低（不再是暗洞+亮面）
- 梯度能量升高（引脚边缘产生强梯度）
- 颜色多样性增加（金属色 vs 塑料）

**算法管线**:

```
校准后的面包板图像
  |
  v
1. 逆透视变换: 将孔洞 warp 坐标映射回原始帧高分辨率坐标
  |
  v
2. 局部 Patch 提取: 每个孔洞中心 ±R 像素 (R = 孔间距×0.8)
  |
  v
3. 多维特征提取 (HoleFeature):
   - center_intensity:  中心 5×5 区域平均灰度
   - contrast:          中心灰度 - 边缘环灰度 (越负 → 越像空洞)
   - gradient_energy:   Sobel 梯度在中心区域的能量
   - color_std:         RGB 3 通道标准差均值
  |
  v
4. 鲁棒统计建模 (Median + MAD):
   - 对所有孔洞的每维特征计算 Median 和 MAD (中位绝对偏差)
   - 建立空孔洞基线模型 (大多数孔洞是空的假设)
  |
  v
5. 占用评分 (改进 Z-score → Sigmoid):
   - 对每维计算修正 Z-score: z = |feature - median| / (MAD × 1.4826)
   - 绝对特征门控: 当特征值仍在空孔洞正常范围 → 压低 Z-score
   - 加权融合: raw = contrast×0.40 + center×0.25 + gradient×0.20 + color×0.15
   - Sigmoid 映射: score = 1 / (1 + exp(-(raw - 3.0)))
  |
  v
6. 导线阴影线性聚簇过滤:
   - 检测同行/同列 ≥4 个连续高分占用孔洞
   - 判定为导线投影阴影干扰
   - 抑制聚簇内部节点 (仅保留两端)
  |
  v
输出: Dict[(hx, hy), float] — 每个孔洞的占用概率
```

**元件引脚定位** (`find_component_pins`):

```
元件检测框 (OBB/HBB)
  |
  v
1. 计算元件主轴方向 + 中心点 (warp 坐标)
  |
  v
2. 自适应搜索半径 (按元件类型调整):
   - Resistor: max(孔间距×5, 长轴×0.6)   ← 长引线
   - LED/Diode: max(孔间距×4, 长轴×0.5)
   - 其他: max(孔间距×3.5, 长轴×0.5)
  |
  v
3. 轴对齐约束过滤:
   - 候选孔洞距元件主轴的垂直距离 < 孔间距×2
   - 按占用分数×距离综合排序
  |
  v
4. 最佳引脚对选择 (_select_best_pair):
   - 避免同导通组短路 (+0.5 惩罚)
   - 避免 span=0 或 span=1 长体元件 (+0.2 惩罚)
   - 奖励合理跨度 span=2~5 (-0.1)
   - 最小引脚间距约束
  |
  v
输出: (逻辑坐标1, 逻辑坐标2) 如 ("17", "a"), ("20", "a")
```

#### 4.1.3 WireAnalyzer (`wire_analyzer.py`, 249 行)

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

#### 4.1.4 ImageAnalyzer (`image_analyzer.py`, ~812 行)

**图片分析引擎 — 替代 FramePipeline, 用于 Image-only 架构**。

```python
@dataclass
class AnalysisResult:
    annotated_image: np.ndarray    # 标注后的图片
    detections: list               # 融合后的检测列表
    report: str                    # 结构化文本报告
    component_count: int           # 元件数量
    net_count: int                 # 电气网络数量
    ocr_results: Dict[str, str]   # IC 型号识别结果
    issues: List[str]              # 检测到的问题
```

**分析管线**:

```
输入: 1-3 张 BGR 图片 + conf + imgsz
  |
  +-- 1. 校准 (auto_calibrate + 4 级孔洞检测)
  |
  +-- 2. 每张图片:
  |     detector.detect(img, conf, imgsz)
  |     → List[Detection]
  |     wire_analyzer.analyze_wire() (仅 Wire 类型)
  |     → 替换 pin1/pin2, 填充 wire_color
  |
  +-- 3. 多图融合 (_fuse_detections):
  |     图1 = 基准
  |     图2,3 与基准做 IoU 匹配 (阈值 0.3)
  |     匹配到: 取置信度更高的
  |     未匹配: 加入结果 (新元件, 被遮挡的)
  |
  +-- 3.5 视觉引脚-孔洞占用检测 (v5.1):
  |     PinHoleDetector.detect_occupied_holes()
  |     → occupancy_map {(hx,hy): score}
  |
  +-- 4. 引脚定位 + 电路建模 (_build_circuit):
  |     策略1 (优先): 视觉占用 → find_component_pins()
  |     策略2 (回退): 几何 Top-K 候选 → _pick_best_pin_pair()
  |     2-pin 元件: analyzer.add_component(loc1, loc2)
  |     3-pin 元件: _find_3pin_middle() → pin3
  |     IC_DIP: OCR → 引脚数据库 → 多引脚映射
  |
  +-- 5. OCR IC 芯片识别 + 极性解析 + 电路验证
  |
  +-- 6. 报告生成 + 标注绘制 (含 Ghost Wire)
  |
  v
输出: AnalysisResult
```

**引脚定位双策略** (v5.1 核心创新):

```
                 [元件检测框 + pin_pixel]
                          |
              +-----------+-----------+
              |                       |
       [视觉占用检测]          [几何候选匹配]
       (优先, 非 Wire)         (回退/Wire)
              |                       |
   PinHoleDetector              Top-K 近邻
   .find_component_pins()      _pick_best_pin_pair()
              |                       |
              +----------++-----------+
                         ||
                   [电路建模]
```

**_pick_best_pin_pair (几何约束选择)**:
从每引脚 K 个候选孔洞中选最佳组合:
- 短路惩罚: 两引脚在同一导通组 → +100 分
- 同行惩罚: 非 Wire 元件两引脚同行 → +50 分
- 大跨度惩罚: 行跨度 > 10 → 额外距离
- Wire 豁免: 不受同组/同行惩罚
- 合理跨度奖励: span 1-5 → -3 分

#### 4.1.5 BreadboardCalibrator (`calibrator.py`, ~700 行)

**四级检测管线**:

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
中缝感知列中心精炼 (v5.1 新增):
  - 检测 e-f 列间的中央凹槽间隙
  - 凹槽两侧独立计算 5+5 列中心
  - 消除凹槽宽度对 a-e / f-j 列映射的干扰
  |
  v
坐标映射:
  nearest_hole()                 → 单一最近孔洞
  top_k_holes(k=3)              → K 个最近孔洞 + 距离分数
  frame_pixel_to_logic_candidates(k) → K 个候选逻辑坐标
```

**中缝感知列中心精炼** (v5.1 新增):

面包板中央有一条贯穿的凹槽 (groove), 将 a-e 列和 f-j 列物理分隔。
旧版将所有孔洞 x 坐标一起聚类为 10 列, 凹槽的额外宽度可能导致 e/f 列偏移。

```
改进: 检测 col_centers[4]-[5] 间距 > 平均列间距×1.3
  → 以凹槽中线分割
  → 左侧 (a-e) 和右侧 (f-j) 独立用 _robust_bin_centers() 拟合 5 列
  → 结果更精确, 消除列偏差
```

### 4.2 推理层 (logic/)

#### 4.2.1 CircuitAnalyzer (`circuit.py`, 758 行)

**核心数据结构: NetworkX 图**

```
节点 = 面包板导通组:
  Row{n}_L      — 第 n 行左侧 (a-e 五孔导通)
  Row{n}_R      — 第 n 行右侧 (f-j 五孔导通)
  RAIL_TOP_1    — 顶部外侧电源轨
  RAIL_TOP_2    — 顶部内侧电源轨
  RAIL_BOTTOM_1 — 底部内侧电源轨
  RAIL_BOTTOM_2 — 底部外侧电源轨
  {name}_hub    — IC 元件中心节点

边 = 元件连接:
  属性: component(名称), type(类型), polarity, confidence,
        pin1_role, pin2_role, junction("EB"/"BC" 仅三端),
        ic_pin_number(仅 IC)
```

**元件数据类**:
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
    pin_locs: List[...]    # IC 全部引脚 [(row,col), ...]
    ic_model: str          # IC 型号 ("LM324")
```

**引脚角色 (PinRole) 枚举**:
```python
class PinRole(Enum):
    GENERIC                        # 通用
    ANODE / CATHODE                # 阳极 / 阴极
    BASE / COLLECTOR / EMITTER     # 三极管 B/C/E
    VCC / GND                      # 电源
    POSITIVE / NEGATIVE            # 正极 / 负极 (电容)
    WIPER                          # 滑动触点 (电位器)
    TERMINAL_A / TERMINAL_B        # 端子 (电位器)
    IC_PIN                         # 通用 IC 引脚
    IC_VCC / IC_GND                # IC 电源引脚
    IC_OUTPUT                      # IC 输出引脚
    IC_INPUT_INV / IC_INPUT_NON    # IC 反相/同相输入
```

**IC Hub 节点模型**:

```
  Row5_L ──[pin1]── U1_hub ──[pin14]── Row5_R
  Row6_L ──[pin2]── U1_hub ──[pin13]── Row6_R
  Row7_L ──[pin3]── U1_hub ──[pin12]── Row7_R
    ...              ...                 ...
  Row11_L ──[pin7]── U1_hub ──[pin8]── Row11_R
```

**4 轨道电源轨模型**:

```python
rail_track_rows = {
    "RAIL_TOP_1":    (1,),    # 顶部外侧
    "RAIL_TOP_2":    (2,),    # 顶部内侧
    "RAIL_BOTTOM_1": (64,),   # 底部内侧
    "RAIL_BOTTOM_2": (65,),   # 底部外侧
}
```

**三端元件边建模**:
```
  E ──[EB结]── B ──[BC结]── C
  node1       node3       node2
```

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

#### 4.2.2 CircuitValidator (`validator.py`, 726 行)

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

#### 4.2.3 PolarityResolver (`polarity.py`, 295 行)

```
                  元件类型
                /    |    \      \        \
POLARIZED   NON_POLAR  THREE_PIN  IC_DIP  POTENTIOMETER
    |           |         |          |        |
OBB 长轴     NONE     视觉 pin3    检查     与三极管
方向推断   (无极性)   → E/B/C    pin_locs  同逻辑
```

#### 4.2.4 IC 引脚数据库 (`ic_pinout_db.py`, 198 行)

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

#### 页面路由

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
|  [校准] [开始分析] [清空]                              |
|  ■■■■■■■■■■■■□□□□□□ 分析中...                         |
|                                                       |
|  +-------------------+  +---------------------------+ |
|  |   标注结果图      |  |  === 分析报告 ===          | |
|  |   (bbox + 坐标)   |  |  元件: 8 (R×3, LED×2...) | |
|  +-------------------+  +---------------------------+ |
+-------------------------------------------------------+
```

#### CircuitPage 电源轨标注 UI

```
+-- 电源轨配置 ----------------------------------+
|  顶部外侧轨:  ● 未连接  [VCC +5V ▼] [      ] |
|  顶部内侧轨:  ● 已标注  [GND     ▼] [      ] |
|  底部内侧轨:  ● 未连接  [选择... ▼] [      ] |
|  底部外侧轨:  ● 已标注  [VCC +5V ▼] [      ] |
|              [清除所有轨道标注]                  |
+------------------------------------------------+
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
|     +-- 视觉引脚-孔洞占用检测 (PinHoleDetector)                |
|     +-- 引脚定位: 视觉占用优先 → 几何候选回退                   |
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
|  用户验证电路 → validator.compare() → ghost_wires              |
|  用户提问 → LLMWorker(snapshot) → ChatPanel                    |
+----------------------------------------------------------------+
```

---

## 六、Intel 异构计算部署方案

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
|  | PinHoleDetector|           |  |                        |  |
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

## 七、配置系统

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

## 八、目录结构 (v5.1)

```
LabGuardian/
├── src_v2/
│   ├── app_context.py           # 服务注册中心 + 线程安全 + 电源轨管理
│   ├── config.py                # 9 个 dataclass 配置 + 9 类元件 + 4 轨道模型
│   ├── launcher.py              # 生产级启动器: 诊断 + 看门狗 + 字体检测
│   ├── build_knowledge_base.py  # 知识库构建脚本
│   ├── ARCHITECTURE.md          # 本文档
│   ├── CHANGELOG_2026-03-01.md  # v3.2→v5.0 变更日志
│   │
│   ├── ai/                      # 认知层
│   │   ├── llm_engine.py        # 三级 LLM 引擎 (Cloud→Local→Rules)
│   │   ├── rag_engine.py        # ChromaDB RAG 知识检索
│   │   └── ocr_engine.py        # PaddleOCR + EasyOCR IC 识别
│   │
│   ├── vision/                  # 感知层
│   │   ├── detector.py          # YOLOv8 HBB+OBB 双模检测 (9 类)
│   │   ├── image_analyzer.py    # 图片分析引擎 (多图融合 + IC 多引脚)
│   │   ├── pin_hole_detector.py # 视觉引脚-孔洞占用检测 (v5.1 新增)
│   │   ├── wire_analyzer.py     # 导线骨架端点 + 颜色分类
│   │   ├── calibrator.py        # 4 级孔洞检测 + RANSAC 网格 + 中缝精炼
│   │   └── stabilizer.py        # 多帧投票稳定器 (保留兼容)
│   │
│   ├── logic/                   # 推理层
│   │   ├── circuit.py           # NetworkX 电路图 + IC hub + 4 轨电源轨
│   │   ├── validator.py         # VF2++ 验证 + 拓扑诊断
│   │   ├── polarity.py          # 极性解析 (含电位器/IC)
│   │   └── ic_pinout_db.py      # IC 引脚数据库 (LM324/LM358/NE5532)
│   │
│   ├── gui_qt/                  # GUI 层 (PySide6)
│   │   ├── main_window.py       # 主窗口 (Image-only 架构)
│   │   ├── upload_page.py       # 图片上传与分析页
│   │   ├── workers.py           # QThread Workers
│   │   ├── circuit_page.py      # 电路验证 + 电源轨标注
│   │   ├── dashboard.py         # 仪表盘
│   │   ├── chat_panel.py        # 聊天面板
│   │   ├── settings_page.py     # 设置页
│   │   ├── sidebar.py           # 侧边栏导航
│   │   ├── styles.py            # PyDracula 暗色主题
│   │   ├── calibration.py       # 校准辅助
│   │   ├── toast.py             # Toast 通知
│   │   ├── resources.py         # 图标常量
│   │   └── run.py               # GUI 入口
│   │
│   └── tools/                   # 开发/测试工具
│       ├── benchmark.py         # 5 项性能基准
│       ├── test_pipeline.py     # 完整流水线测试
│       ├── test_batch.py        # 批量测试 (多场景)
│       ├── test_pin_detection.py # 引脚检测对比测试 (几何/视觉/YOLO)
│       ├── annotate_helper.py   # OBB 标注辅助
│       ├── train_obb.py         # 训练脚本
│       ├── fix_dataset.py       # 数据集修复
│       └── replicate_labels.py  # 标签复制工具
│
├── models/                      # AI 模型权重文件
│   ├── yolo26n.pt               # YOLOv2.6 nano
│   ├── yolov8n.pt               # YOLOv8 nano
│   ├── yolov8n-obb.pt           # YOLOv8 OBB nano
│   ├── lab_guardian_obb_openvino/  # OBB OpenVINO IR
│   └── tinyllama_openvino/      # LLM OpenVINO IR
│
├── dataset/                     # 训练数据集
│   ├── data.yaml                # 数据集配置
│   ├── train/                   # 训练集 (342 images + labels)
│   └── val/                     # 验证集
│
├── knowledge_base/              # RAG 知识库
│   ├── builtin/                 # 内置技术文档
│   │   ├── analog_lab_guide.md
│   │   ├── chip_pinout_guide.md
│   │   ├── ne555_timer.md
│   │   ├── op_amp_basics.md
│   │   └── transistor_805x.md
│   └── chroma_db/               # ChromaDB 向量数据库 (自动生成)
│
├── scripts/                     # 独立工具脚本
│   ├── export_llm_openvino.py   # LLM 模型导出
│   ├── prepare_dataset.py       # 数据集准备
│   ├── setup_roboflow_dataset.py # Roboflow 数据集配置
│   └── train_pipeline.py        # 训练编排脚本
│
├── shared/                      # 共享模块
│   ├── models.py                # 共享数据模型
│   └── risk.py                  # 风险评估逻辑
│
├── teacher/                     # 教师端 (Vue.js + Flask)
│   ├── server.py / classroom.py # 后端
│   └── frontend/                # Vue.js 前端 (Nav/Station/Alert/Ranking)
│
├── runs/                        # YOLO 训练结果 (gitignored)
├── logs/                        # 运行日志 (gitignored)
├── test_results/                # 测试输出 (gitignored)
├── .gitignore
├── README.md
├── start.sh                     # Linux 启动脚本
└── start_classroom.bat          # Windows 课堂模式启动
```

---

## 九、版本变更历史

### v5.1 (2026-03-03) — 视觉引脚-孔洞占用检测 + 校准精炼

**核心新增**: PinHoleDetector — 通过图像特征分析直接判断孔洞占用状态

- **新建** `vision/pin_hole_detector.py` (~420 行): 视觉占用检测器
  - Median + MAD 鲁棒统计建模
  - 改进 Sigmoid 评分 (3σ 阈值 + 绝对特征门控)
  - 元件类型自适应搜索半径 (Resistor: 5×spacing)
  - 主轴对齐约束过滤 (perp_tolerance = 2×spacing)
  - 导线阴影线性聚簇过滤 (≥4 连续孔 → 抑制中间节点)
  - 最佳引脚对选择 (span/导通组约束 + 最小间距)

- **修改** `vision/calibrator.py`: 中缝感知列中心精炼
  - 检测 e-f 列间凹槽间隙
  - 凹槽两侧独立拟合 5+5 列中心
  - 消除列坐标偏差

- **修改** `vision/image_analyzer.py`: 集成视觉引脚检测
  - 步骤 3.5: 调用 PinHoleDetector.detect_occupied_holes()
  - _build_circuit: 视觉占用优先 → 几何候选回退

- **新增** 测试工具:
  - `tools/test_pipeline.py`: 完整流水线测试 (含视觉 vs 几何对比)
  - `tools/test_batch.py`: 批量场景测试
  - `tools/test_pin_detection.py`: 三种引脚检测方法对比

- **验证结果** (3 个真实场景):
  - 假阳性降低 17~76% (Scene1: 75→18, Scene3: 214→93, Scene5: 87→72)
  - 长引线元件 (电阻) 定位从 N/A 恢复到可用

### v5.0 (2026-03-01) — Image-only + 导线骨架 + IC 多引脚 + 元件扩展

- **Image-only 架构迁移**: UploadPage + ImageAnalyzer 替代视频流 (FramePipeline)
- **WireAnalyzer**: 骨架端点 + HSV 颜色分类 (red/blue/green/yellow/orange/black/white)
- **9 类元件**: +TRANSISTOR, IC_DIP, POTENTIOMETER
- **IC 引脚数据库**: LM324/LM358/NE5532 + hub 多引脚模型
- **4 轨道电源轨**: 学生主动标注 (删除投票推断)
- **PinRole 扩展**: 电位器 (Wiper/Terminal) + IC (VCC/GND/Output/Input)

### v3.2 (2026-02-24) — 电源轨推断 + 三极管视觉检测
### v3.1 (2026-02-24) — 逻辑层强化 + 引脚遮挡补偿
### v3.0 (2026-02-22) — AppContext + 线程安全重构

---

## 十、技术决策记录

### 决策 1: 放弃视频流, 采用 Image-only
- **原因**: 手机 12MP 俯拍远优于 USB 摄像头 640×480; 代码复杂度降低 40%
- **权衡**: 失去实时反馈, 但竞赛场景以结果准确性为首要

### 决策 2: 电源轨由学生标注而非系统推断
- **原因**: 投票推断需足够多极性元件, 简单电路可能失败
- **权衡**: 增加一步操作, 消除推断失败风险

### 决策 3: 导线用骨架化而非训练端点检测模型
- **原因**: 纯 OpenCV, 无需标注数据; 对弯曲导线天然适配
- **权衡**: 交叉重叠场景最大连通域可能混入其他导线

### 决策 4: IC 用 OCR + 查表而非训练分类模型
- **原因**: IC 型号有限, 查表 100% 准确; 无需训练数据
- **权衡**: 不支持数据库外的 IC (可按需扩展)

### 决策 5: 引脚定位采用视觉占用检测 + 几何回退
- **原因**: YOLO OBB 边界估计误差 1-3 行; 纯几何方法假阳性高
- **权衡**: 视觉检测依赖图片质量和光照; 几何方法作为可靠回退

---

## 十一、术语表

| 术语 | 含义 |
|------|------|
| HBB / OBB | Horizontal / Oriented Bounding Box |
| VF2++ | 图同构匹配算法 |
| GED | Graph Edit Distance, 图编辑距离 |
| Hub 节点 | IC 多引脚的中心虚拟节点 |
| Ghost Wire | 幽灵线, 缺失连接的标注引导 |
| Golden Reference | 教师预设的正确电路模板 |
| 导通组 | 面包板上 5 孔组: Row{n}_L (a-e) / Row{n}_R (f-j) |
| 骨架化 | 二值图迭代腐蚀至单像素宽 (Zhang-Suen) |
| IoU 融合 | 多图检测结果 Intersection over Union 合并 |
| 电源轨标注 | 学生手动指定轨道用途 (VCC/GND/电压) |
| DIP | Dual In-line Package, 双列直插封装 |
| MAD | Median Absolute Deviation, 中位绝对偏差 |
| 占用概率 | 孔洞被引脚插入的视觉评分 (0~1) |
| 中缝 / Groove | 面包板中央贯穿凹槽, 分隔 a-e 和 f-j 列 |

---

*文档版本: v5.1 | 2026-03-03*
