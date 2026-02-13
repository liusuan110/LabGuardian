"""
电路验证器模块 (v2)
职责：保存/加载标准电路模板，与当前检测的电路进行对比

v2 增强:
  - 极性感知的拓扑同构比较 (Polarity-aware VF2++)
  - 极性错误精确诊断 ("LED/二极管接反", "三极管 B/C/E 错位")
  - 电源网络匹配检查 (VCC/GND 正确性)
  - 保存/加载增强: 持久化极性和引脚角色信息

参考:
  - NetworkX VF2++ (Jüttner & Madarasi, 2018): 带标签子图同构
  - EDA LVS (Layout Versus Schematic) 方法论
"""

import logging
import json
import networkx as nx
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from networkx.readwrite import json_graph

from .circuit import (
    CircuitAnalyzer, CircuitComponent,
    Polarity, PinRole,
    POLARIZED_TYPES, THREE_PIN_TYPES,
)

logger = logging.getLogger(__name__)


class CircuitValidator:
    """
    电路验证器
    
    支持两种对比模式:
    1. 拓扑同构比较 (布局无关): 只看元件种类和连接关系
    2. 启发式位置比较 (布局相关): 检查元件在面包板上的具体位置
    """

    def __init__(self):
        self.ref_graph: Optional[nx.Graph] = None
        self.ref_components: List[CircuitComponent] = []
        self.ref_topology: Optional[nx.Graph] = None

    @property
    def has_reference(self) -> bool:
        return len(self.ref_components) > 0

    def set_reference(self, analyzer: CircuitAnalyzer):
        """将当前电路状态设为 Golden Reference (保留极性信息)"""
        self.ref_graph = analyzer.graph.copy()
        self.ref_components = [
            CircuitComponent(
                name=c.name, type=c.type,
                pin1_loc=c.pin1_loc, pin2_loc=c.pin2_loc,
                polarity=c.polarity,
                pin_roles=c.pin_roles,
                confidence=c.confidence,
                orientation_deg=c.orientation_deg,
                pin3_loc=c.pin3_loc,
            )
            for c in analyzer.components
        ]
        try:
            self.ref_topology = analyzer.build_topology_graph()
        except Exception:
            self.ref_topology = None
        logger.info(f"[Validator] Reference set with {len(self.ref_components)} components.")

    def save_reference(self, file_path: str):
        """将 Golden Reference 保存为 JSON 文件"""
        if not self.ref_components:
            raise ValueError("No reference circuit set.")

        topo_payload = None
        if self.ref_topology is not None:
            topo_payload = json_graph.node_link_data(self.ref_topology)

        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "format": "labguardian_ref_v3",
            },
            "components": [
                {
                    "name": c.name,
                    "type": c.type,
                    "pin1_loc": list(c.pin1_loc) if c.pin1_loc else None,
                    "pin2_loc": list(c.pin2_loc) if c.pin2_loc else None,
                    "polarity": c.polarity.value,
                    "pin_roles": [r.value for r in c.pin_roles],
                    "pin3_loc": list(c.pin3_loc) if c.pin3_loc else None,
                }
                for c in self.ref_components
            ],
            "topology": topo_payload,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_reference(self, file_path: str):
        """从 JSON 文件加载 Golden Reference"""
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        comps = []
        for item in payload.get("components", []):
            pin1 = tuple(item["pin1_loc"]) if item.get("pin1_loc") else None
            pin2 = tuple(item["pin2_loc"]) if item.get("pin2_loc") else None
            if pin1 is None:
                continue

            # 还原极性信息 (兼容 v2 格式: 无极性字段则默认 NONE)
            pol_str = item.get("polarity", "none")
            try:
                polarity = Polarity(pol_str)
            except ValueError:
                polarity = Polarity.NONE

            roles_raw = item.get("pin_roles", ["generic", "generic"])
            pin_roles = tuple(
                PinRole(r) if r in [e.value for e in PinRole] else PinRole.GENERIC
                for r in roles_raw
            )

            pin3 = tuple(item["pin3_loc"]) if item.get("pin3_loc") else None

            comps.append(CircuitComponent(
                name=item.get("name", "UNKNOWN"),
                type=item.get("type", "UNKNOWN"),
                pin1_loc=pin1,
                pin2_loc=pin2,
                polarity=polarity,
                pin_roles=pin_roles,
                pin3_loc=pin3,
            ))

        self.ref_components = comps

        # 重建参考图
        tmp = CircuitAnalyzer()
        for c in self.ref_components:
            tmp.add_component(c)
        self.ref_graph = tmp.graph.copy()

        # 重建拓扑图
        topo_data = payload.get("topology")
        if topo_data:
            try:
                self.ref_topology = json_graph.node_link_graph(topo_data)
            except Exception:
                self.ref_topology = None
        else:
            try:
                self.ref_topology = tmp.build_topology_graph()
            except Exception:
                self.ref_topology = None

    def compare(self, curr_analyzer: CircuitAnalyzer) -> Dict:
        """
        对比当前电路与参考电路
        
        Returns:
            {
                'errors': List[str],                   # 文字描述的差异
                'missing_links': List[(loc1, loc2)],   # 缺失连接 (用于幽灵线绘制)
                'is_match': bool,                      # 是否完全匹配
            }
        """
        result = {
            'errors': [],
            'missing_links': [],
            'is_match': False,
        }

        if not self.has_reference:
            result['errors'].append("No reference circuit set. Cannot validate.")
            return result

        # ---- 拓扑同构检测 (布局无关) ----
        try:
            if self.ref_topology is not None:
                curr_topo = curr_analyzer.build_topology_graph()

                def _node_match(a, b):
                    """v2: 带极性标签的节点匹配"""
                    if a.get('kind') != b.get('kind'):
                        return False
                    if a.get('kind') == 'comp':
                        if a.get('ctype') != b.get('ctype'):
                            return False
                        if a.get('pins', 2) != b.get('pins', 2):
                            return False
                        # 极性匹配 (仅在参考有已知极性时检查)
                        ref_pol = a.get('polarity', 'none')
                        cur_pol = b.get('polarity', 'none')
                        if ref_pol in ('forward', 'reverse') and cur_pol in ('forward', 'reverse'):
                            if ref_pol != cur_pol:
                                return False  # 极性方向不匹配
                        return True
                    if a.get('kind') == 'net':
                        # 电源网络: 参考有 power 标记时, 当前也应匹配
                        ref_power = a.get('power')
                        cur_power = b.get('power')
                        if ref_power and cur_power:
                            return ref_power == cur_power
                    return True

                from networkx.algorithms.isomorphism import GraphMatcher
                gm = GraphMatcher(self.ref_topology, curr_topo, node_match=_node_match)

                if gm.is_isomorphic():
                    result['errors'].append("✅ Topology matches lesson template (layout-independent).")
                    result['is_match'] = True
                    return result
                else:
                    # 先尝试不带极性的匹配, 区分"拓扑错误"和"极性错误"
                    self._check_polarity_errors(result, curr_topo)
                    self._append_topology_diff(result, curr_topo)
        except Exception as e:
            result['errors'].append(f"⚠️ Topology check failed (fallback): {e}")

        # ---- 元件数量检查 ----
        ref_counts = Counter(c.type for c in self.ref_components)
        curr_counts = Counter(c.type for c in curr_analyzer.components)

        for t in sorted(set(ref_counts.keys()) | set(curr_counts.keys())):
            r_c, c_c = ref_counts[t], curr_counts[t]
            if c_c < r_c:
                result['errors'].append(f"❌ Missing {r_c - c_c} x {t}")
            elif c_c > r_c:
                result['errors'].append(f"⚠️ Extra {c_c - r_c} x {t}")

        # ---- 位置启发式匹配 ----
        matched = set()
        for ref_c in self.ref_components:
            best_idx, min_dist = None, 999
            ref_row = int(ref_c.pin1_loc[0])

            for idx, curr_c in enumerate(curr_analyzer.components):
                if idx in matched or curr_c.type != ref_c.type:
                    continue
                try:
                    dist = abs(int(curr_c.pin1_loc[0]) - ref_row)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                except (ValueError, TypeError):
                    continue

            if best_idx is not None:
                matched.add(best_idx)
                if min_dist > 2:
                    curr_c = curr_analyzer.components[best_idx]
                    result['errors'].append(
                        f"⚠️ {ref_c.type} position mismatch: "
                        f"expected Row~{ref_row}, found Row{curr_c.pin1_loc[0]}"
                    )
            else:
                result['missing_links'].append((ref_c.pin1_loc, ref_c.pin2_loc))

        if not result['errors']:
            result['errors'].append("✅ Circuit matches Reference!")
            result['is_match'] = True

        return result

    def _check_polarity_errors(self, result: Dict, curr_topo: nx.Graph):
        """
        极性专项诊断

        策略: 先用不检查极性的 _node_match 做VF2++
        如果拓扑匹配但极性不配 → 说明电路连接对了但元件方向反了
        """
        if self.ref_topology is None:
            return

        def _node_match_no_polarity(a, b):
            if a.get('kind') != b.get('kind'):
                return False
            if a.get('kind') == 'comp':
                return (a.get('ctype') == b.get('ctype') and
                        a.get('pins', 2) == b.get('pins', 2))
            return True

        from networkx.algorithms.isomorphism import GraphMatcher
        gm = GraphMatcher(self.ref_topology, curr_topo,
                          node_match=_node_match_no_polarity)

        if gm.is_isomorphic():
            # 拓扑正确, 但带极性匹配失败 → 极性问题
            mapping = gm.mapping  # ref_node → curr_node
            for ref_node, curr_node in mapping.items():
                ref_data = self.ref_topology.nodes[ref_node]
                cur_data = curr_topo.nodes[curr_node]

                if ref_data.get('kind') != 'comp':
                    continue

                ref_pol = ref_data.get('polarity', 'none')
                cur_pol = cur_data.get('polarity', 'none')
                ctype = ref_data.get('ctype', '?')

                if ref_pol in ('forward', 'reverse') and cur_pol in ('forward', 'reverse'):
                    if ref_pol != cur_pol:
                        if ctype in {'LED', 'DIODE'}:
                            result['errors'].append(
                                f"🔴 {ctype} ({curr_node}) 接反了！"
                                f"请将阳极(+)和阴极(-)对调")
                        elif ctype in {'TRANSISTOR', 'NPN', 'PNP'}:
                            result['errors'].append(
                                f"🔴 {ctype} ({curr_node}) 引脚方向错误！"
                                f"请检查 B/C/E 引脚接法")
                        else:
                            result['errors'].append(
                                f"🔴 {ctype} ({curr_node}) 极性反接")

                elif ref_pol in ('forward', 'reverse') and cur_pol == 'unknown':
                    result['errors'].append(
                        f"⚠️ {ctype} ({curr_node}) 极性无法判断,"
                        f" 请确认安装方向")

            # 检查电源网络
            for ref_node, curr_node in mapping.items():
                ref_data = self.ref_topology.nodes[ref_node]
                cur_data = curr_topo.nodes[curr_node]
                if ref_data.get('kind') == 'net':
                    ref_pwr = ref_data.get('power')
                    cur_pwr = cur_data.get('power')
                    if ref_pwr and cur_pwr and ref_pwr != cur_pwr:
                        result['errors'].append(
                            f"🔴 电源网络极性错误: 期望 {ref_pwr}, 实际 {cur_pwr}")

    def _append_topology_diff(self, result: Dict, curr_topo: nx.Graph):
        """添加拓扑差异的详细诊断信息"""
        def _counts(g):
            comps = [d.get('ctype') for _, d in g.nodes(data=True) if d.get('kind') == 'comp']
            nets = sum(1 for _, d in g.nodes(data=True) if d.get('kind') == 'net')
            return nets, comps

        ref_nets, ref_comps = _counts(self.ref_topology)
        cur_nets, cur_comps = _counts(curr_topo)
        rc, cc = Counter(ref_comps), Counter(cur_comps)

        result['errors'].append("❌ Topology mismatch vs lesson template.")
        result['errors'].append(f"   Nets: expected {ref_nets}, found {cur_nets}")
        for t in sorted(set(rc.keys()) | set(cc.keys())):
            if rc[t] != cc[t]:
                result['errors'].append(f"   {t}: expected {rc[t]}, found {cc[t]}")


# 全局单例
validator = CircuitValidator()
