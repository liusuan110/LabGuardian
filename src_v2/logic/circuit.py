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
NON_POLAR_TYPES = {"RESISTOR", "WIRE", "Push_Button"} # 无极性元件
POWER_KEYWORDS = {"VCC", "GND", "POWER", "BATTERY"}  # 电源相关关键词


class CircuitAnalyzer:
    """电路拓扑分析器。

    基于 NetworkX 图论，将面包板上检测到的元件建模为电气连接图。
    面包板规则: 同一行 a-e 导通 (Left节点), f-j 导通 (Right节点)。
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.components: List[CircuitComponent] = []
        self.power_nets: Dict[str, str] = {}  # node_name → "VCC" | "GND"

    def reset(self):
        """清空所有元件和连接"""
        self.graph.clear()
        self.components = []
        self.power_nets = {}

    def add_component(self, comp: CircuitComponent):
        """添加元件到电路图，自动根据面包板导通规则生成电气节点。

        边属性携带极性 / 置信度 / 引脚角色信息。
        """
        self.components.append(comp)

        node1 = self._get_node_name(comp.pin1_loc)

        if comp.pin2_loc:
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
            self.graph.add_node(node1, component=comp.name)

        # 三极管第三引脚
        if comp.pin3_loc is not None:
            node3 = self._get_node_name(comp.pin3_loc)
            self.graph.add_edge(node1, node3,
                                component=comp.name + "_p13",
                                type=comp.type,
                                polarity=comp.polarity.value,
                                confidence=comp.confidence)

    @staticmethod
    def _get_node_name(loc: Tuple[str, str]) -> str:
        """根据面包板规则将 (Row, Col) 映射为电气节点名。

        a-e → Row{n}_L, f-j → Row{n}_R
        """
        row, col = loc
        if col in ('a', 'b', 'c', 'd', 'e'):
            return f"Row{row}_L"
        else:
            return f"Row{row}_R"

    @staticmethod
    def _norm_type(t: str) -> str:
        """归一化元件类型名"""
        if not t:
            return "UNKNOWN"
        u = str(t).strip().upper()
        if "RESIST" in u:
            return "RESISTOR"
        if "WIRE" in u:
            return "WIRE"
        if "LED" in u:
            return "LED"
        if "BUTTON" in u or "SWITCH" in u:
            return "SWITCH"
        if "DIODE" in u:
            return "DIODE"
        if "CAP" in u:
            return "CAPACITOR"
        return u

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
        """生成自然语言描述的电路网表，供 LLM 理解电路连接关系。"""
        if not self.components:
            return "No circuit components detected."

        desc = "Current Detected Circuit Topology:\n"
        connected_groups = list(nx.connected_components(self.graph))

        # 标注电源网络
        self._identify_power_nets()

        for idx, group in enumerate(connected_groups):
            nodes = sorted(list(group))
            rows = [n for n in nodes if n.startswith("Row")]
            net_label = f"Net {idx + 1}"

            # 检查是否是电源网络
            for n in nodes:
                if n in self.power_nets:
                    net_label += f" ({self.power_nets[n]})"
                    break

            desc += f"- {net_label} connects: {', '.join(rows)}\n"

        desc += "\nComponent Connections:\n"
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            side1 = "Left" if comp.pin1_loc[1] <= 'e' else "Right"

            # 极性信息
            pol_info = ""
            if comp.polarity == Polarity.FORWARD:
                pol_info = f" [pin1={comp.pin_roles[0].value}, pin2={comp.pin_roles[1].value if len(comp.pin_roles)>1 else '?'}]"
            elif comp.polarity == Polarity.UNKNOWN:
                pol_info = " [polarity: UNKNOWN]"

            if comp.pin2_loc:
                side2 = "Left" if comp.pin2_loc[1] <= 'e' else "Right"
                desc += (f"- {comp.name} ({ctype}{pol_info}) connects "
                         f"Row{comp.pin1_loc[0]} ({side1}) to "
                         f"Row{comp.pin2_loc[0]} ({side2}) "
                         f"[conf={comp.confidence:.2f}]\n")
            else:
                desc += f"- {comp.name} ({ctype}{pol_info}) at Row{comp.pin1_loc[0]} ({side1})\n"

            # 三极管第三引脚
            if comp.pin3_loc:
                side3 = "Left" if comp.pin3_loc[1] <= 'e' else "Right"
                desc += f"  └ pin3 at Row{comp.pin3_loc[0]} ({side3})\n"

        # 电源网络摘要
        if self.power_nets:
            desc += "\nPower Networks:\n"
            for node, ptype in sorted(self.power_nets.items()):
                desc += f"  {node} → {ptype}\n"

        return desc

    # ========================================================
    # 电源网络识别
    # ========================================================

    def _identify_power_nets(self):
        """启发式识别电源网络：根据元件类型和引脚角色标记 VCC / GND。"""
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            # 电池/电源模块的引脚直接标记
            if ctype in POWER_KEYWORDS:
                if comp.pin_roles[0] == PinRole.VCC:
                    node = self._get_node_name(comp.pin1_loc)
                    self.power_nets[node] = "VCC"
                if len(comp.pin_roles) > 1 and comp.pin_roles[1] == PinRole.GND:
                    if comp.pin2_loc:
                        node = self._get_node_name(comp.pin2_loc)
                        self.power_nets[node] = "GND"

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
