"""电路拓扑建模模块

职责
----
将检测到的元件构建为电气连接图，生成网表描述。

核心能力
--------
1. 元件极性 / 引脚角色感知 (polarity, pin_roles)
2. 电源网络识别 (VCC / GND 标记)
3. 置信度加权连接
4. 结构化网表导出 (供 LLM 理解和 LVS 比较)

参考
----
- SKiDL (devbisme/skidl): Net/Part/Pin 数据模型
- schematic-o-matic (nickbild): 面包板→网表映射
- NetworkX VF2++ (Jüttner & Madarasi, 2018): 带标签图同构
"""

import logging
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# 元件类型归一化 (全局统一)
# ============================================================

def norm_component_type(t: str) -> str:
    """归一化元件类型名。
    全模块统一使用此函数, 避免 circuit.py / polarity.py 行为不一致。
    """
    if not t:
        return "UNKNOWN"
    u = str(t).strip().upper()
    if "RESIST" in u:
        return "RESISTOR"
    if "WIRE" in u:
        return "WIRE"
    if "LED" in u:
        return "LED"
    if "DIODE" in u:
        return "DIODE"
    if "BUTTON" in u or "SWITCH" in u:
        return "Push_Button"
    if "CAP" in u:
        return "CAPACITOR"
    if "NPN" in u:
        return "NPN"
    if "PNP" in u:
        return "PNP"
    if "TRANSISTOR" in u or "BJT" in u:
        return "TRANSISTOR"
    if "BATTERY" in u or "POWER" in u:
        return "POWER"
    if "OPAMP" in u or "OP-AMP" in u or "OP_AMP" in u:
        return "OPAMP"
    if "555" in u:
        return "IC_555"
    return u


# ============================================================
# 枚举定义
# ============================================================

class Polarity(Enum):
    """元件极性 / 方向"""
    NONE = "none"            # 无极性（电阻、导线）
    FORWARD = "forward"      # 正向: pin1=阳极, pin2=阴极
    REVERSE = "reverse"      # 反向: pin1=阴极, pin2=阳极
    UNKNOWN = "unknown"      # 有极性但未确定方向


class PinRole(Enum):
    """引脚角色"""
    GENERIC = "generic"      # 通用 (无特殊含义)
    ANODE = "anode"          # 阳极 (二极管/LED)
    CATHODE = "cathode"      # 阴极
    BASE = "base"            # 基极 (三极管)
    COLLECTOR = "collector"  # 集电极
    EMITTER = "emitter"      # 发射极
    VCC = "vcc"              # 电源正极
    GND = "gnd"              # 电源负极
    POSITIVE = "positive"    # 正极 (电容)
    NEGATIVE = "negative"    # 负极


@dataclass
class CircuitComponent:
    """电路元件数据类。

    Attributes:
        name: 元件名称 (e.g. "R1", "LED1")
        type: 元件类型 (e.g. "RESISTOR", "LED")
        pin1_loc: 引脚1 坐标 (Row, Col)
        pin2_loc: 引脚2 坐标 (可为空)
        polarity: 极性方向 (由 PolarityResolver 填充)
        pin_roles: 各引脚角色 (e.g. anode/cathode, B/C/E)
        confidence: 检测置信度 (来自 YOLO)
        orientation_deg: OBB 旋转角度
        pin3_loc: 三极管第三引脚坐标
    """
    name: str          # e.g. "R1", "LED1"
    type: str          # e.g. "RESISTOR", "Wire", "LED"
    pin1_loc: Tuple[str, str]  # (行, 列) e.g. ("15", "a")
    pin2_loc: Optional[Tuple[str, str]] = None

    polarity: Polarity = Polarity.NONE
    pin_roles: Tuple[PinRole, ...] = (PinRole.GENERIC, PinRole.GENERIC)
    confidence: float = 1.0
    orientation_deg: float = 0.0

    # 三极管第三引脚
    pin3_loc: Optional[Tuple[str, str]] = None

    def __repr__(self):
        pol = f" [{self.polarity.value}]" if self.polarity != Polarity.NONE else ""
        return f"{self.name}({self.pin1_loc}-{self.pin2_loc}{pol})"

    @property
    def is_polarized(self) -> bool:
        """该元件是否有极性"""
        return self.polarity not in (Polarity.NONE,)

    @property
    def has_known_polarity(self) -> bool:
        """极性是否已确定"""
        return self.polarity in (Polarity.FORWARD, Polarity.REVERSE)


# ============================================================
# 元件类型分类 (用于极性判断)
# ============================================================

POLARIZED_TYPES = {"DIODE", "LED"}                  # 有极性的二端元件
THREE_PIN_TYPES = {"TRANSISTOR", "NPN", "PNP"}       # 三端元件
CAPACITOR_TYPES = {"CAPACITOR"}                       # 可能有极性 (电解) 也可能无极性 (瓷片)
NON_POLAR_TYPES = {"RESISTOR", "WIRE", "Push_Button"} # 无极性元件
IC_TYPES = {"IC_555", "OPAMP"}                        # 多引脚 IC
POWER_KEYWORDS = {"VCC", "GND", "POWER", "BATTERY"}  # 电源相关关键词


class CircuitAnalyzer:
    """电路拓扑分析器。

    基于 NetworkX 图论，将面包板上检测到的元件建模为电气连接图。
    面包板规则: 同一行 a-e 导通 (Left节点), f-j 导通 (Right节点)。
    电源轨: 最上/最下行为总线, 通过拓扑分析推断 VCC/GND。
    """

    def __init__(self,
                 rail_top_rows: Tuple[int, ...] = (),
                 rail_bottom_rows: Tuple[int, ...] = ()):
        """
        Args:
            rail_top_rows:    面包板顶部电源轨行号 (e.g. (1, 2))
            rail_bottom_rows: 面包板底部电源轨行号 (e.g. (29, 30))
        """
        self.graph = nx.Graph()
        self.components: List[CircuitComponent] = []
        self.power_nets: Dict[str, str] = {}  # node_name → "VCC" | "GND"
        self._name_counters: Dict[str, int] = {}  # 元件类型 → 自增计数

        # 电源轨行号 (总线节点: 同一轨所有行共用一个节点)
        self._rail_top_rows = set(rail_top_rows)
        self._rail_bottom_rows = set(rail_bottom_rows)
        self._all_rail_rows = self._rail_top_rows | self._rail_bottom_rows

    def reset(self):
        """清空所有元件和连接"""
        self.graph.clear()
        self.components = []
        self.power_nets = {}
        self._name_counters = {}

    # ---- 元件命名 ----
    _TYPE_PREFIX = {
        "RESISTOR": "R", "LED": "LED", "DIODE": "D",
        "CAPACITOR": "C", "WIRE": "W", "Push_Button": "SW",
        "TRANSISTOR": "Q", "NPN": "Q", "PNP": "Q",
        "IC_555": "U", "OPAMP": "U", "POWER": "PWR",
    }

    def _auto_name(self, comp_type: str) -> str:
        """生成唯一的元件名 (R1, R2, LED1, Q1, ...)"""
        norm = self._norm_type(comp_type)
        prefix = self._TYPE_PREFIX.get(norm, norm[:3])
        self._name_counters[norm] = self._name_counters.get(norm, 0) + 1
        return f"{prefix}{self._name_counters[norm]}"

    def add_component(self, comp: CircuitComponent):
        """添加元件到电路图，自动根据面包板导通规则生成电气节点。

        如果 comp.name == comp.type (未手动命名), 自动分配唯一名称。
        边属性携带极性 / 置信度 / 引脚角色信息。
        """
        # 自动命名: name 仍为原始 class_name 时,分配 R1/LED2 等
        if comp.name == comp.type or comp.name in ("UNKNOWN", ""):
            comp.name = self._auto_name(comp.type)

        self.components.append(comp)

        node1 = self._get_node_name(comp.pin1_loc)

        if comp.pin3_loc is not None:
            # 三端元件 (三极管): pin1=E, pin3=B(中间), pin2=C
            # 建模为两条边: E-B 和 B-C (不是 E-C 直连)
            node2 = self._get_node_name(comp.pin2_loc) if comp.pin2_loc else None
            node3 = self._get_node_name(comp.pin3_loc)

            role1 = comp.pin_roles[0].value if comp.pin_roles else 'generic'
            role3 = comp.pin_roles[2].value if len(comp.pin_roles) > 2 else 'generic'
            role2 = comp.pin_roles[1].value if len(comp.pin_roles) > 1 else 'generic'

            base_attrs = {
                'type': comp.type,
                'polarity': comp.polarity.value,
                'confidence': comp.confidence,
            }

            # Edge: pin1(E) — pin3(B)
            self.graph.add_edge(node1, node3,
                                component=comp.name,
                                pin1_role=role1, pin2_role=role3,
                                junction="EB", **base_attrs)
            # Edge: pin3(B) — pin2(C)
            if node2 is not None:
                self.graph.add_edge(node3, node2,
                                    component=comp.name,
                                    pin1_role=role3, pin2_role=role2,
                                    junction="BC", **base_attrs)

        elif comp.pin2_loc:
            # 二端元件 (电阻/LED/二极管/电容/导线/按钮)
            node2 = self._get_node_name(comp.pin2_loc)
            edge_attrs = {
                'component': comp.name,
                'type': comp.type,
                'polarity': comp.polarity.value,
                'confidence': comp.confidence,
                'pin1_role': comp.pin_roles[0].value if comp.pin_roles else 'generic',
                'pin2_role': comp.pin_roles[1].value if len(comp.pin_roles) > 1 else 'generic',
            }
            self.graph.add_edge(node1, node2, **edge_attrs)
        else:
            # 单引脚元件
            self.graph.add_node(node1, component=comp.name)

    def _get_node_name(self, loc: Tuple[str, str]) -> str:
        """根据面包板规则将 (Row, Col) 映射为电气节点名。

        面包板导通规则:
          主区域: a-e → Row{n}_L (左侧5孔导通), f-j → Row{n}_R (右侧5孔导通)
          电源轨: 顶部行 → RAIL_TOP (总线), 底部行 → RAIL_BOTTOM (总线)
                  '+'/'-' → PWR_PLUS / PWR_MINUS (旧格式兼容)

        电源轨总线: 同一轨的所有行(如 row 1,2)是横向全部导通的,
        不区分左右侧, 映射为同一个总线节点。
        """
        row, col = loc
        if col in ('+', 'plus', 'P'):
            return "PWR_PLUS"
        if col in ('-', 'minus', 'N', 'GND'):
            return "PWR_MINUS"

        # 电源轨行: 整行导通, 映射为总线节点
        try:
            row_int = int(row)
        except (ValueError, TypeError):
            row_int = -1

        if row_int in self._rail_top_rows:
            return "RAIL_TOP"
        if row_int in self._rail_bottom_rows:
            return "RAIL_BOTTOM"

        # 普通行: 左右分侧导通
        if col in ('a', 'b', 'c', 'd', 'e'):
            return f"Row{row}_L"
        else:
            return f"Row{row}_R"

    @staticmethod
    def _norm_type(t: str) -> str:
        """归一化元件类型名 (委托给模块级函数)"""
        return norm_component_type(t)

    def build_topology_graph(self) -> nx.Graph:
        """构建布局无关的拓扑图（双部图）。

        节点类型:
          - Net节点 (kind='net'): 电气网络
          - 元件节点 (kind='comp'): 携带 ctype / polarity 标签

        Wire 被视为理想导体，合并到网络中，不生成元件节点。
        """
        # 构建仅含 Wire 的子图，用于确定电气网络连通性
        conductor = nx.Graph()
        conductor.add_nodes_from(self.graph.nodes())

        for u, v, data in self.graph.edges(data=True):
            if self._norm_type(data.get("type", "")) == "WIRE":
                conductor.add_edge(u, v)

        # 每个 Wire 连通分量代表一个电气网络
        net_groups = list(nx.connected_components(conductor))
        node_to_net = {}
        for i, group in enumerate(net_groups):
            for n in group:
                node_to_net[n] = f"N{i}"

        topo = nx.Graph()

        # 添加网络节点（标记电源类型）
        self._identify_power_nets()
        for i in range(len(net_groups)):
            net_id = f"N{i}"
            attrs = {"kind": "net"}
            # 检查是否是电源网络
            for n in net_groups[i]:
                if n in self.power_nets:
                    attrs["power"] = self.power_nets[n]
                    break
            topo.add_node(net_id, **attrs)

        # 添加元件节点（跳过 Wire，携带极性标签）
        comp_idx = 0
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "WIRE":
                continue

            try:
                n1 = node_to_net.get(self._get_node_name(comp.pin1_loc))
                n2 = node_to_net.get(self._get_node_name(comp.pin2_loc)) if comp.pin2_loc else None
            except Exception:
                n1, n2 = None, None

            if n1 is None:
                continue

            cid = f"C{comp_idx}"
            comp_idx += 1

            node_attrs = {
                "kind": "comp",
                "ctype": ctype,
                "polarity": comp.polarity.value,
            }

            if n2 is None:
                node_attrs["pins"] = 1
                topo.add_node(cid, **node_attrs)
                topo.add_edge(cid, n1)
            elif n1 == n2:
                node_attrs["pins"] = 2
                node_attrs["same_net"] = True
                topo.add_node(cid, **node_attrs)
                topo.add_edge(cid, n1)
            else:
                node_attrs["pins"] = 2
                topo.add_node(cid, **node_attrs)
                # 对有极性元件: 边携带引脚角色
                if comp.polarity != Polarity.NONE and len(comp.pin_roles) >= 2:
                    topo.add_edge(cid, n1, pin_role=comp.pin_roles[0].value)
                    topo.add_edge(cid, n2, pin_role=comp.pin_roles[1].value)
                else:
                    topo.add_edge(cid, n1)
                    topo.add_edge(cid, n2)

            # 三极管第三引脚
            if comp.pin3_loc:
                n3 = node_to_net.get(self._get_node_name(comp.pin3_loc))
                if n3 is not None:
                    node_attrs["pins"] = 3
                    pin3_role = comp.pin_roles[2].value if len(comp.pin_roles) > 2 else 'generic'
                    topo.add_edge(cid, n3, pin_role=pin3_role)

        return topo

    def get_circuit_description(self) -> str:
        """生成结构化电路网表描述，供 LLM 理解电路连接关系。

        输出格式:
          1. 元件统计摘要
          2. 各元件引脚位置 + 极性 + 所在网络
          3. 电气网络 (哪些节点导通)
          4. 电源网络摘要
          5. 潜在问题提示
        """
        if not self.components:
            return "当前未检测到电路元件。"

        self._identify_power_nets()
        connected_groups = list(nx.connected_components(self.graph))

        # 构建 node → net_id 映射
        node_to_net = {}
        for idx, group in enumerate(connected_groups):
            net_id = f"Net_{idx + 1}"
            for n in group:
                node_to_net[n] = net_id

        # ---- 1. 元件统计摘要 ----
        type_counts = Counter(self._norm_type(c.type) for c in self.components)
        total = len(self.components)
        counts_str = ", ".join(f"{t}×{c}" for t, c in sorted(type_counts.items()))
        desc = f"电路概况: 共 {total} 个元件 ({counts_str}), {len(connected_groups)} 个电气网络\n\n"

        # ---- 2. 元件详情 ----
        desc += "元件连接:\n"
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            node1 = self._get_node_name(comp.pin1_loc)
            net1 = node_to_net.get(node1, "?")

            # 极性/引脚角色描述
            role_info = ""
            if comp.polarity == Polarity.FORWARD:
                r1 = comp.pin_roles[0].value if comp.pin_roles else '?'
                r2 = comp.pin_roles[1].value if len(comp.pin_roles) > 1 else '?'
                role_info = f" [{r1}/{r2}]"
            elif comp.polarity == Polarity.REVERSE:
                r1 = comp.pin_roles[0].value if comp.pin_roles else '?'
                r2 = comp.pin_roles[1].value if len(comp.pin_roles) > 1 else '?'
                role_info = f" [{r1}/{r2} 反向]"
            elif comp.polarity == Polarity.UNKNOWN:
                role_info = " [极性未知]"

            if comp.pin3_loc:
                # 三端元件 (三极管)
                node2 = self._get_node_name(comp.pin2_loc) if comp.pin2_loc else "?"
                node3 = self._get_node_name(comp.pin3_loc)
                net2 = node_to_net.get(node2, "?")
                net3 = node_to_net.get(node3, "?")

                r1 = comp.pin_roles[0].value if comp.pin_roles else 'pin1'
                r2 = comp.pin_roles[1].value if len(comp.pin_roles) > 1 else 'pin2'
                r3 = comp.pin_roles[2].value if len(comp.pin_roles) > 2 else 'pin3'

                desc += (f"  {comp.name} ({ctype}): "
                         f"{r1}=Row{comp.pin1_loc[0]}{comp.pin1_loc[1]}({net1}), "
                         f"{r3}=Row{comp.pin3_loc[0]}{comp.pin3_loc[1]}({net3}), "
                         f"{r2}=Row{comp.pin2_loc[0]}{comp.pin2_loc[1]}({net2})\n")
            elif comp.pin2_loc:
                # 二端元件
                node2 = self._get_node_name(comp.pin2_loc)
                net2 = node_to_net.get(node2, "?")
                desc += (f"  {comp.name} ({ctype}{role_info}): "
                         f"Row{comp.pin1_loc[0]}{comp.pin1_loc[1]}({net1}) — "
                         f"Row{comp.pin2_loc[0]}{comp.pin2_loc[1]}({net2})\n")
            else:
                desc += (f"  {comp.name} ({ctype}{role_info}): "
                         f"Row{comp.pin1_loc[0]}{comp.pin1_loc[1]}({net1})\n")

        # ---- 3. 电气网络 ----
        desc += "\n电气网络:\n"
        for idx, group in enumerate(connected_groups):
            net_id = f"Net_{idx + 1}"
            nodes = sorted(list(group))

            # 收集该网络上的元件
            comps_on_net = []
            for comp in self.components:
                comp_nodes = set()
                comp_nodes.add(self._get_node_name(comp.pin1_loc))
                if comp.pin2_loc:
                    comp_nodes.add(self._get_node_name(comp.pin2_loc))
                if comp.pin3_loc:
                    comp_nodes.add(self._get_node_name(comp.pin3_loc))
                if comp_nodes & group:
                    comps_on_net.append(comp.name)

            # 电源标记
            power_tag = ""
            for n in nodes:
                if n in self.power_nets:
                    power_tag = f" [{self.power_nets[n]}]"
                    break

            comps_str = ", ".join(sorted(set(comps_on_net)))
            desc += f"  {net_id}{power_tag}: {', '.join(nodes)} → 元件: {comps_str}\n"

        # ---- 4. 电源网络 ----
        if self.power_nets:
            desc += "\n电源:\n"
            for node, ptype in sorted(self.power_nets.items()):
                desc += f"  {node} → {ptype}\n"

        # ---- 5. 潜在问题 ----
        issues = self._quick_check_issues()
        if issues:
            desc += "\n⚠ 潜在问题:\n"
            for issue in issues:
                desc += f"  - {issue}\n"

        return desc

    def _quick_check_issues(self) -> List[str]:
        """快速检查常见电路问题 (用于网表描述)"""
        issues = []
        has_led = False
        has_resistor_near_led = False

        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "LED":
                has_led = True
                # 检查同网络是否有电阻
                led_node1 = self._get_node_name(comp.pin1_loc)
                led_node2 = self._get_node_name(comp.pin2_loc) if comp.pin2_loc else None
                for other in self.components:
                    if self._norm_type(other.type) == "RESISTOR":
                        r_node1 = self._get_node_name(other.pin1_loc)
                        r_node2 = self._get_node_name(other.pin2_loc) if other.pin2_loc else None
                        if (r_node1 in (led_node1, led_node2) or
                                r_node2 in (led_node1, led_node2)):
                            has_resistor_near_led = True
                            break

            if ctype in POLARIZED_TYPES and comp.polarity == Polarity.UNKNOWN:
                issues.append(f"{comp.name} ({ctype}) 极性未确定, 请检查安装方向")

            # 检查同一 net 两个引脚 (短路)
            if comp.pin2_loc:
                n1 = self._get_node_name(comp.pin1_loc)
                n2 = self._get_node_name(comp.pin2_loc)
                if n1 == n2 and ctype != "WIRE":
                    issues.append(f"{comp.name} ({ctype}) 两引脚在同一导通组, 可能短路或未跨行")

        if has_led and not has_resistor_near_led:
            issues.append("LED 未检测到相邻限流电阻, 可能缺少限流保护")

        return issues

    # ========================================================
    # 电源网络识别
    # ========================================================

    def _identify_power_nets(self):
        """启发式识别电源网络：

        1. 电源轨总线节点 (RAIL_TOP / RAIL_BOTTOM) 通过拓扑推断 VCC/GND
        2. 旧格式电源轨 (PWR_PLUS / PWR_MINUS) 直接标记
        3. 标记为 POWER/BATTERY 的元件引脚
        4. 引脚角色为 VCC/GND 的元件引脚
        """
        # 1. 电源轨总线: 通过拓扑推断 VCC vs GND
        has_top = "RAIL_TOP" in self.graph
        has_bottom = "RAIL_BOTTOM" in self.graph

        if has_top or has_bottom:
            top_role, bottom_role = self._infer_power_rail_polarity()
            if has_top:
                self.power_nets["RAIL_TOP"] = top_role
            if has_bottom:
                self.power_nets["RAIL_BOTTOM"] = bottom_role

        # 2. 旧格式电源轨直接标记
        if "PWR_PLUS" in self.graph:
            self.power_nets["PWR_PLUS"] = "VCC"
        if "PWR_MINUS" in self.graph:
            self.power_nets["PWR_MINUS"] = "GND"

        # 3. 元件引脚角色标记
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype in POWER_KEYWORDS:
                if comp.pin_roles[0] == PinRole.VCC:
                    node = self._get_node_name(comp.pin1_loc)
                    self.power_nets[node] = "VCC"
                if len(comp.pin_roles) > 1 and comp.pin_roles[1] == PinRole.GND:
                    if comp.pin2_loc:
                        node = self._get_node_name(comp.pin2_loc)
                        self.power_nets[node] = "GND"
            # 其他元件如果引脚被标记为 VCC/GND, 也记录
            for i, role in enumerate(comp.pin_roles):
                if role == PinRole.VCC:
                    loc = [comp.pin1_loc, comp.pin2_loc, comp.pin3_loc][i] if i < 3 else None
                    if loc:
                        self.power_nets[self._get_node_name(loc)] = "VCC"
                elif role == PinRole.GND:
                    loc = [comp.pin1_loc, comp.pin2_loc, comp.pin3_loc][i] if i < 3 else None
                    if loc:
                        self.power_nets[self._get_node_name(loc)] = "GND"

    def _infer_power_rail_polarity(self) -> Tuple[str, str]:
        """从电路拓扑推断顶部轨和底部轨分别是 VCC 还是 GND。

        推断策略 (投票机制):
          1. LED/二极管阴极侧连接的轨 → GND 票
          2. LED/二极管阳极链方向连接的轨 → VCC 票
          3. 三极管发射极连接的轨 → 通常是 GND (NPN 共射电路)
          4. 三极管集电极经电阻到轨 → VCC 票
          5. 默认: 顶部=VCC, 底部=GND (面包板约定:上红下蓝)

        Returns:
            (top_role, bottom_role): 各为 "VCC" 或 "GND"
        """
        # 投票: positive → top=VCC, negative → top=GND
        vote = 0

        for comp in self.components:
            ctype = self._norm_type(comp.type)

            if ctype in POLARIZED_TYPES and comp.pin1_loc and comp.pin2_loc:
                # LED / 二极管: pin1=阳极, pin2=阴极
                node_anode = self._get_node_name(comp.pin1_loc)
                node_cathode = self._get_node_name(comp.pin2_loc)

                # 阴极连到哪个轨 → 那个轨是 GND
                if node_cathode == "RAIL_TOP":
                    vote -= 2  # top=GND
                elif node_cathode == "RAIL_BOTTOM":
                    vote += 2  # bottom=GND → top=VCC

                # 阳极连到哪个轨 → 那个轨是 VCC
                if node_anode == "RAIL_TOP":
                    vote += 2  # top=VCC
                elif node_anode == "RAIL_BOTTOM":
                    vote -= 2  # bottom=VCC → top=GND

                # 间接连接: 阴极经导线到轨
                if node_cathode not in ("RAIL_TOP", "RAIL_BOTTOM"):
                    for neighbor in self.graph.neighbors(node_cathode):
                        edge = self.graph.edges[node_cathode, neighbor]
                        if self._norm_type(edge.get("type", "")) == "WIRE":
                            if neighbor == "RAIL_TOP":
                                vote -= 1
                            elif neighbor == "RAIL_BOTTOM":
                                vote += 1

            if ctype in THREE_PIN_TYPES:
                # 三极管: 发射极通常接 GND (NPN 共射)
                if comp.pin_roles and len(comp.pin_roles) >= 2:
                    for i, role in enumerate(comp.pin_roles):
                        loc = [comp.pin1_loc, comp.pin2_loc, comp.pin3_loc][i] if i < 3 else None
                        if loc is None:
                            continue
                        node = self._get_node_name(loc)
                        if role == PinRole.EMITTER:
                            if node == "RAIL_TOP":
                                vote -= 1  # E→TOP → TOP=GND
                            elif node == "RAIL_BOTTOM":
                                vote += 1  # E→BOTTOM → BOTTOM=GND → TOP=VCC
                        elif role == PinRole.COLLECTOR:
                            if node == "RAIL_TOP":
                                vote += 1  # C→TOP → TOP=VCC
                            elif node == "RAIL_BOTTOM":
                                vote -= 1

        # 默认: 顶部=VCC, 底部=GND (面包板约定上红下蓝)
        if vote >= 0:
            return ("VCC", "GND")
        else:
            return ("GND", "VCC")

    def get_net_count(self) -> int:
        """获取电气网络数量"""
        return len(list(nx.connected_components(self.graph)))

    def validate_connection(self, pin1_row: str, pin2_row: str) -> bool:
        """检查两行是否导通"""
        node1 = f"Row{pin1_row}_L"
        node2 = f"Row{pin2_row}_L"

        if node1 not in self.graph:
            node1 = f"Row{pin1_row}_R"
        if node2 not in self.graph:
            node2 = f"Row{pin2_row}_R"

        if node1 not in self.graph or node2 not in self.graph:
            return False

        return nx.has_path(self.graph, node1, node2)

    def find_missing_link(self, target_netlist):
        """对比目标网表，找出缺失的连接"""
        missing = []
        for start, end in target_netlist:
            if not (self.graph.has_node(start) and
                    self.graph.has_node(end) and
                    nx.has_path(self.graph, start, end)):
                missing.append(f"{start} -> {end}")
        return sorted(missing)

    # ========================================================
    # 网表导出
    # ========================================================

    def export_netlist(self) -> Dict:
        """导出结构化网表（类 SPICE 格式）。

        Returns:
            dict 包含::

                nets        网络分组 {net_id: [node_names]}
                components  元件列表 [{name, type, polarity, pins}]
                power       电源映射 {net_id: 'VCC'|'GND'}
        """
        self._identify_power_nets()

        # 构建网络
        connected = list(nx.connected_components(self.graph))
        node_to_net_id = {}
        nets = {}
        for i, group in enumerate(connected):
            net_id = f"N{i}"
            nets[net_id] = sorted(list(group))
            for n in group:
                node_to_net_id[n] = net_id

        # 导出元件
        comp_list = []
        for comp in self.components:
            entry = {
                'name': comp.name,
                'type': self._norm_type(comp.type),
                'polarity': comp.polarity.value,
                'confidence': comp.confidence,
                'pins': [],
            }

            # Pin 1
            n1 = self._get_node_name(comp.pin1_loc)
            role1 = comp.pin_roles[0].value if comp.pin_roles else 'generic'
            entry['pins'].append({
                'loc': comp.pin1_loc,
                'role': role1,
                'net': node_to_net_id.get(n1, 'floating'),
            })

            # Pin 2
            if comp.pin2_loc:
                n2 = self._get_node_name(comp.pin2_loc)
                role2 = comp.pin_roles[1].value if len(comp.pin_roles) > 1 else 'generic'
                entry['pins'].append({
                    'loc': comp.pin2_loc,
                    'role': role2,
                    'net': node_to_net_id.get(n2, 'floating'),
                })

            # Pin 3 (transistor)
            if comp.pin3_loc:
                n3 = self._get_node_name(comp.pin3_loc)
                role3 = comp.pin_roles[2].value if len(comp.pin_roles) > 2 else 'generic'
                entry['pins'].append({
                    'loc': comp.pin3_loc,
                    'role': role3,
                    'net': node_to_net_id.get(n3, 'floating'),
                })

            comp_list.append(entry)

        # 电源网络映射
        power = {}
        for node, ptype in self.power_nets.items():
            net_id = node_to_net_id.get(node)
            if net_id:
                power[net_id] = ptype

        return {
            'nets': nets,
            'components': comp_list,
            'power': power,
        }
