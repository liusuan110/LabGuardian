"""电路验证器模块

职责
----
保存 / 加载标准电路模板 (Golden Reference)，与学生当前搭建的
电路进行多级对比，输出结构化诊断结果。

核心能力
--------
1. 极性感知拓扑同构 (Polarity-aware VF2++)
2. 子图同构进度评估 — 学生只搭了部分电路时给出完成率
3. 图编辑距离 (GED) — 量化相似度 + 精确差异定位
4. 边属性感知 — pin_role 级别连接验证
5. 快速预拒 — 度序列 + 元件签名 O(1) 排除不可能匹配
6. 分级诊断管线: L0 预检 → L1 全图同构 → L2 子图 → L3 GED

参考文献
--------
- Jüttner & Madarasi (2018): VF2++ 带标签子图同构
- Dong et al. (2021): 子图同构子电路识别, IEEE ICTA
- Li et al. (2024): 两阶段子图匹配框架, ACM ICCAD
- Xu et al. (2024): 图注意力对称约束提取 — GED 度量
- Guo (2022): 容错图匹配 / 近似图同构
- EDA LVS (Layout Versus Schematic) 方法论
"""

import logging
import json
import math
import networkx as nx
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from itertools import combinations

from networkx.readwrite import json_graph

from .circuit import (
    CircuitAnalyzer, CircuitComponent,
    Polarity, PinRole,
    POLARIZED_TYPES, THREE_PIN_TYPES,
    norm_component_type,
)

logger = logging.getLogger(__name__)


def _graph_signature(g: nx.Graph) -> Tuple:
    """计算图的结构不变量签名，用于 O(1) 快速排除不可能的同构匹配。

    同构图必然共享相同的度序列与节点标签分布 (Abiad 2020)。

    Returns:
        (节点数, 边数, 排序度序列, 元件类型计数, 网络节点数)
    """
    degrees = sorted([d for _, d in g.degree()], reverse=True)
    comp_types = Counter(
        d.get('ctype', 'NET')
        for _, d in g.nodes(data=True)
        if d.get('kind') == 'comp'
    )
    net_count = sum(1 for _, d in g.nodes(data=True) if d.get('kind') == 'net')
    return (g.number_of_nodes(), g.number_of_edges(),
            tuple(degrees), tuple(sorted(comp_types.items())), net_count)


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
        """将 Golden Reference 序列化为 JSON 文件（含极性与引脚角色）。"""
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
        """从 JSON 文件反序列化 Golden Reference（兼容早期格式）。"""
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        comps = []
        for item in payload.get("components", []):
            pin1 = tuple(item["pin1_loc"]) if item.get("pin1_loc") else None
            pin2 = tuple(item["pin2_loc"]) if item.get("pin2_loc") else None
            if pin1 is None:
                continue

            # 还原极性（兼容旧格式：缺失字段时默认 NONE）
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

        # 从元件列表重建连接图与拓扑图
        tmp = CircuitAnalyzer()
        for c in self.ref_components:
            tmp.add_component(c)
        self.ref_graph = tmp.graph.copy()

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
        """对比当前电路与参考电路，按分级管线逐层诊断。

        诊断管线::

            L0  快速预检  ─ 元件数量统计
            L1  全图同构  ─ 带极性 + 边属性的 VF2++
            L2  子图同构  ─ 学生搭了部分电路时的进度评估
            L2.5 极性诊断 ─ 拓扑对但极性错的专项检查
            L3  GED      ─ 图编辑距离 + 精确差异定位

        Returns:
            dict 包含以下键::

                errors             文字描述的差异列表
                missing_links      缺失连接坐标
                extra_links        多余连接坐标
                is_match           是否完全匹配
                similarity         相似度 0.0 ~ 1.0
                progress           搭建进度 0.0 ~ 1.0
                matched_components 已正确匹配的元件名
                missing_components 缺失的元件类型
                extra_components   多余的元件类型
                polarity_errors    极性错误描述列表
        """
        result = {
            'errors': [],
            'missing_links': [],
            'extra_links': [],
            'is_match': False,
            'similarity': 0.0,
            'progress': 0.0,
            'matched_components': [],
            'missing_components': [],
            'extra_components': [],
            'polarity_errors': [],
        }

        if not self.has_reference:
            result['errors'].append("No reference circuit set. Cannot validate.")
            return result

        # ---- L0: 元件数量统计 ----
        ref_counts = Counter(c.type for c in self.ref_components)
        curr_counts = Counter(c.type for c in curr_analyzer.components)

        for t in sorted(set(ref_counts.keys()) | set(curr_counts.keys())):
            r_c, c_c = ref_counts[t], curr_counts[t]
            if c_c < r_c:
                result['errors'].append(f"❌ Missing {r_c - c_c} x {t}")
                result['missing_components'].extend([t] * (r_c - c_c))
            elif c_c > r_c:
                result['errors'].append(f"⚠️ Extra {c_c - r_c} x {t}")
                result['extra_components'].extend([t] * (c_c - r_c))

        # ---- L1: 全图同构 (带极性 + 边属性 VF2++) ----
        try:
            if self.ref_topology is not None:
                curr_topo = curr_analyzer.build_topology_graph()

                # L1a: 度序列预拒 — 不一致则跳过开销较大的 VF2++
                ref_sig = _graph_signature(self.ref_topology)
                cur_sig = _graph_signature(curr_topo)

                if ref_sig == cur_sig:
                    # 度序列匹配，执行完整 VF2++
                    from networkx.algorithms.isomorphism import GraphMatcher
                    gm = GraphMatcher(
                        self.ref_topology, curr_topo,
                        node_match=self._node_match_full,
                        edge_match=self._edge_match,
                    )

                    if gm.is_isomorphic():
                        result['errors'] = ["✅ Topology matches lesson template (layout-independent)."]
                        result['is_match'] = True
                        result['similarity'] = 1.0
                        result['progress'] = 1.0
                        result['matched_components'] = [
                            c.name for c in self.ref_components
                        ]
                        return result

                # ---- L2: 子图同构 — 学生可能只搭了一部分 ----
                self._check_subgraph_match(result, curr_topo)

                # ---- L2.5: 极性专项诊断 ----
                self._check_polarity_errors(result, curr_topo)

                # ---- L3: GED 相似度 + 精确差异定位 ----
                self._compute_ged_similarity(result, curr_topo)
                self._localize_errors(result, curr_analyzer, curr_topo)

        except Exception as e:
            result['errors'].append(f"⚠️ Topology check failed (fallback): {e}")
            logger.exception("Topology check failed")

        # 位置启发式匹配（兆底）
        if not result['is_match']:
            self._heuristic_position_match(result, curr_analyzer)

        if not result['errors']:
            result['errors'].append("✅ Circuit matches Reference!")
            result['is_match'] = True

        return result

    # ----------------------------------------------------------------
    # 节点 / 边匹配函数 (VF2++ 回调)
    # ----------------------------------------------------------------

    @staticmethod
    def _node_match_full(a: dict, b: dict) -> bool:
        """带极性的严格节点匹配。

        同时检查 kind / ctype / pins / polarity，
        仅当参考与当前都有确定极性时才比较极性。
        """
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
                    return False
            return True
        if a.get('kind') == 'net':
            ref_power = a.get('power')
            cur_power = b.get('power')
            if ref_power and cur_power:
                return ref_power == cur_power
        return True

    @staticmethod
    def _node_match_no_polarity(a: dict, b: dict) -> bool:
        """不检查极性的节点匹配，用于区分拓扑错误 vs 极性错误。"""
        if a.get('kind') != b.get('kind'):
            return False
        if a.get('kind') == 'comp':
            return (a.get('ctype') == b.get('ctype') and
                    a.get('pins', 2) == b.get('pins', 2))
        return True

    @staticmethod
    def _node_match_type_only(a: dict, b: dict) -> bool:
        """仅比较 kind + ctype 的宽松匹配，用于子图同构检测。"""
        if a.get('kind') != b.get('kind'):
            return False
        if a.get('kind') == 'comp':
            return a.get('ctype') == b.get('ctype')
        return True

    @staticmethod
    def _edge_match(a: dict, b: dict) -> bool:
        """边属性匹配: 检查 pin_role 一致性。

        确保阳极连到阳极侧网络、阴极连到阴极侧网络，
        而非仅仅拓扑连通。任一侧缺少 pin_role 时视为匹配。
        """
        ref_role = a.get('pin_role')
        cur_role = b.get('pin_role')
        # 如果任一侧没有标记 pin_role, 视为匹配 (宽容)
        if ref_role is None or cur_role is None:
            return True
        return ref_role == cur_role

    # ----------------------------------------------------------------
    # 子图同构检测
    # ----------------------------------------------------------------

    def _check_subgraph_match(self, result: Dict, curr_topo: nx.Graph):
        """检查当前电路是否是参考电路的子图，评估搭建进度。

        场景: 学生只搭了参考电路的一部分 (progress < 100%)。
        策略: 用宽松的类型匹配检查 curr ⊆ ref。
        """
        if self.ref_topology is None:
            return

        from networkx.algorithms.isomorphism import GraphMatcher

        # 检查 curr_topo 是否是 ref_topology 的子图
        gm = GraphMatcher(
            self.ref_topology, curr_topo,
            node_match=self._node_match_type_only,
        )

        if gm.subgraph_is_isomorphic():
            mapping = gm.mapping  # ref_node → curr_node
            matched_ref_comps = set()
            for ref_node in mapping:
                data = self.ref_topology.nodes[ref_node]
                if data.get('kind') == 'comp':
                    matched_ref_comps.add(ref_node)

            total_ref_comps = sum(
                1 for _, d in self.ref_topology.nodes(data=True)
                if d.get('kind') == 'comp'
            )

            if total_ref_comps > 0:
                progress = len(matched_ref_comps) / total_ref_comps
            else:
                progress = 0.0

            result['progress'] = progress

            # 找出参考中未被匹配的元件 (学生还没搭的部分)
            all_ref_comps = set(
                n for n, d in self.ref_topology.nodes(data=True)
                if d.get('kind') == 'comp'
            )
            unmatched = all_ref_comps - matched_ref_comps
            for comp_node in unmatched:
                ctype = self.ref_topology.nodes[comp_node].get('ctype', '?')
                result['missing_components'].append(ctype)

            if progress < 1.0:
                result['errors'].append(
                    f"📊 Circuit is a valid subset of reference "
                    f"(progress: {progress:.0%}, "
                    f"{len(matched_ref_comps)}/{total_ref_comps} components matched)")

            result['matched_components'] = [
                self.ref_topology.nodes[n].get('ctype', '?')
                for n in matched_ref_comps
            ]

            logger.info(f"[Validator] Subgraph match: {progress:.0%} progress, "
                        f"{len(unmatched)} components remaining")

    # ----------------------------------------------------------------
    # 图编辑距离 (GED) 相似度计算
    # ----------------------------------------------------------------

    def _compute_ged_similarity(self, result: Dict, curr_topo: nx.Graph):
        """计算 GED 并转化为相似度 = 1 - GED / max(|V|+|E|)。

        小规模图 (≤ 50) 使用精确解，大规模图回退到启发式近似。
        """
        if self.ref_topology is None:
            return

        ref_size = (self.ref_topology.number_of_nodes() +
                    self.ref_topology.number_of_edges())
        cur_size = (curr_topo.number_of_nodes() +
                    curr_topo.number_of_edges())
        max_size = max(ref_size, cur_size, 1)

        # NetworkX 的 GED 对大图可能较慢, 限制规模
        if ref_size > 50 or cur_size > 50:
            # 大规模电路: 启发式近似 GED
            similarity = self._approximate_ged_similarity(curr_topo)
            result['similarity'] = max(result.get('similarity', 0), similarity)
            result['errors'].append(
                f"📏 Circuit similarity (approx): {similarity:.0%}")
            return

        try:
            # 精确 GED（仅限小规模电路）
            # 自定义代价函数: 极性错误代价低于类型错误
            def _node_subst_cost(a, b):
                if a.get('kind') != b.get('kind'):
                    return 2.0  # 类型不同: 高代价
                if a.get('kind') == 'comp':
                    if a.get('ctype') != b.get('ctype'):
                        return 1.5  # 同为元件但类型不同
                    if a.get('polarity', 'none') != b.get('polarity', 'none'):
                        return 0.5  # 类型对但极性不同: 低代价
                    return 0.0
                return 0.0

            def _node_del_cost(a):
                return 1.0

            def _node_ins_cost(a):
                return 1.0

            def _edge_subst_cost(a, b):
                ref_role = a.get('pin_role')
                cur_role = b.get('pin_role')
                if ref_role and cur_role and ref_role != cur_role:
                    return 0.5
                return 0.0

            def _edge_del_cost(a):
                return 1.0

            def _edge_ins_cost(a):
                return 1.0

            # 使用 optimize_graph_edit_distance 迭代获取更优解
            best_ged = max_size  # 初始上界
            for ged in nx.optimize_graph_edit_distance(
                self.ref_topology, curr_topo,
                node_subst_cost=_node_subst_cost,
                node_del_cost=_node_del_cost,
                node_ins_cost=_node_ins_cost,
                edge_subst_cost=_edge_subst_cost,
                edge_del_cost=_edge_del_cost,
                edge_ins_cost=_edge_ins_cost,
            ):
                best_ged = ged
                break  # 取首个优化解即可

            similarity = max(0.0, 1.0 - best_ged / max_size)
            result['similarity'] = max(result.get('similarity', 0), similarity)
            result['errors'].append(
                f"📏 Circuit similarity: {similarity:.0%} "
                f"(GED={best_ged:.1f}, size={max_size})")

        except Exception as e:
            logger.warning(f"GED computation failed: {e}, using approximation")
            similarity = self._approximate_ged_similarity(curr_topo)
            result['similarity'] = max(result.get('similarity', 0), similarity)
            result['errors'].append(
                f"📏 Circuit similarity (approx): {similarity:.0%}")

    def _approximate_ged_similarity(self, curr_topo: nx.Graph) -> float:
        """近似 GED 相似度（用于大规模电路）。

        融合三个维度:
          1) 元件类型分布余弦相似度
          2) 度序列相似度
          3) 边数比值
        """
        if self.ref_topology is None:
            return 0.0

        # 维度1: 元件类型分布余弦相似度
        ref_types = Counter(
            d.get('ctype', 'NET')
            for _, d in self.ref_topology.nodes(data=True)
            if d.get('kind') == 'comp'
        )
        cur_types = Counter(
            d.get('ctype', 'NET')
            for _, d in curr_topo.nodes(data=True)
            if d.get('kind') == 'comp'
        )
        all_types = set(ref_types.keys()) | set(cur_types.keys())
        if not all_types:
            return 1.0

        dot = sum(ref_types.get(t, 0) * cur_types.get(t, 0) for t in all_types)
        norm_r = math.sqrt(sum(v ** 2 for v in ref_types.values())) or 1
        norm_c = math.sqrt(sum(v ** 2 for v in cur_types.values())) or 1
        type_sim = dot / (norm_r * norm_c)

        # 维度2: 度序列相似度 (L1 归一化差异)
        ref_deg = sorted([d for _, d in self.ref_topology.degree()], reverse=True)
        cur_deg = sorted([d for _, d in curr_topo.degree()], reverse=True)
        max_len = max(len(ref_deg), len(cur_deg), 1)
        # 填充到相同长度
        ref_deg.extend([0] * (max_len - len(ref_deg)))
        cur_deg.extend([0] * (max_len - len(cur_deg)))
        deg_diff = sum(abs(a - b) for a, b in zip(ref_deg, cur_deg))
        deg_sum = sum(ref_deg) + sum(cur_deg) or 1
        deg_sim = 1.0 - deg_diff / deg_sum

        # 维度3: 边数比值
        ref_e = self.ref_topology.number_of_edges() or 1
        cur_e = curr_topo.number_of_edges() or 1
        edge_sim = min(ref_e, cur_e) / max(ref_e, cur_e)

        # 加权融合
        similarity = 0.5 * type_sim + 0.3 * deg_sim + 0.2 * edge_sim
        return max(0.0, min(1.0, similarity))

    # ----------------------------------------------------------------
    # 精确差异定位
    # ----------------------------------------------------------------

    def _localize_errors(self, result: Dict,
                         curr_analyzer: CircuitAnalyzer,
                         curr_topo: nx.Graph):
        """对比参考图与当前图的边集差异，将缺失/多余连接映射回面包板位置。"""
        if self.ref_topology is None or curr_topo is None:
            return

        # 提取参考图的元件连接签名 (ctype-net 对)
        ref_connections = self._extract_connection_signatures(self.ref_topology)
        cur_connections = self._extract_connection_signatures(curr_topo)

        missing_conns = ref_connections - cur_connections
        extra_conns = cur_connections - ref_connections

        for conn in missing_conns:
            result['errors'].append(f"🔗 Missing connection: {conn}")
        for conn in extra_conns:
            result['errors'].append(f"🔗 Extra connection: {conn}")

        # 将缺失连接映射回面包板位置
        for ref_c in self.ref_components:
            # 检查该元件的连接是否在当前电路中存在
            found = False
            for curr_c in curr_analyzer.components:
                if (curr_c.type == ref_c.type and
                        curr_c.pin1_loc == ref_c.pin1_loc and
                        curr_c.pin2_loc == ref_c.pin2_loc):
                    found = True
                    break
            if not found and ref_c.pin1_loc and ref_c.pin2_loc:
                result['missing_links'].append((ref_c.pin1_loc, ref_c.pin2_loc))

    @staticmethod
    def _extract_connection_signatures(topo: nx.Graph) -> Set[str]:
        """提取拓扑图的连接签名集，格式: "COMP_TYPE[邻居网络特征]" 。

        用于粗粒度差异对比: 缺失/多余的签名意味着对应连接存在问题。
        """
        signatures = set()
        for node, data in topo.nodes(data=True):
            if data.get('kind') == 'comp':
                ctype = data.get('ctype', '?')
                # 收集该元件连接到的网络的特征
                neighbors = []
                for nb in topo.neighbors(node):
                    nb_data = topo.nodes[nb]
                    if nb_data.get('kind') == 'net':
                        power = nb_data.get('power', '')
                        degree = topo.degree(nb)
                        neighbors.append(f"{power}d{degree}")
                neighbors.sort()
                sig = f"{ctype}[{'|'.join(neighbors)}]"
                signatures.add(sig)
        return signatures

    # ----------------------------------------------------------------
    # 位置启发式匹配 (兆底补充)
    # ----------------------------------------------------------------

    def _heuristic_position_match(self, result: Dict,
                                  curr_analyzer: CircuitAnalyzer):
        """布局相关的位置匹配: 按元件类型+行号最近邻进行贪心匹配。"""
        matched = set()
        for ref_c in self.ref_components:
            best_idx, min_dist = None, 999
            try:
                ref_row = int(ref_c.pin1_loc[0])
            except (ValueError, TypeError):
                continue

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
                if ref_c.pin1_loc and ref_c.pin2_loc:
                    result['missing_links'].append(
                        (ref_c.pin1_loc, ref_c.pin2_loc))


    def _check_polarity_errors(self, result: Dict, curr_topo: nx.Graph):
        """极性专项诊断: 不带极性做 VF2++，若拓扑匹配但极性不配，
        则说明电路连接正确但元件方向反了。
        """
        if self.ref_topology is None:
            return

        from networkx.algorithms.isomorphism import GraphMatcher
        gm = GraphMatcher(self.ref_topology, curr_topo,
                          node_match=self._node_match_no_polarity)

        if gm.is_isomorphic():
            # 拓扑正确但带极性匹配失败 → 遍历映射找出极性不一致的元件
            mapping = gm.mapping
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
                        if ctype in {'LED', 'Diode', 'Electrolytic_Capacitor'}:
                            result['errors'].append(
                                f"🔴 {ctype} ({curr_node}) 接反了！"
                                f"请将阳极(+)和阴极(-)对调")
                        elif ctype in {'Transistor', 'NPN', 'PNP'}:
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
        """添加详细的拓扑差异诊断: 节点/边计数 + 连接模式差异。"""
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

        # 比较每种元件类型的度分布，发现连接模式差异
        for ctype in sorted(set(rc.keys()) & set(cc.keys())):
            ref_degrees = sorted([
                self.ref_topology.degree(n)
                for n, d in self.ref_topology.nodes(data=True)
                if d.get('ctype') == ctype
            ])
            cur_degrees = sorted([
                curr_topo.degree(n)
                for n, d in curr_topo.nodes(data=True)
                if d.get('ctype') == ctype
            ])
            if ref_degrees != cur_degrees:
                result['errors'].append(
                    f"   {ctype} connectivity differs: "
                    f"ref degrees={ref_degrees}, current={cur_degrees}")

    # ----------------------------------------------------------------
    # 独立电路诊断 (无需参考电路)
    # ----------------------------------------------------------------

    @staticmethod
    def diagnose(analyzer: CircuitAnalyzer) -> List[str]:
        """基于拓扑的独立电路诊断，不依赖参考电路。

        检查项:
          1. LED 缺少限流电阻 (同网络中无 RESISTOR)
          2. 有极性元件极性未知
          3. 元件两引脚在同一导通组 (短路/未跨行)
          4. 孤立元件 (只有一个连接, 无法形成回路)
          5. 三极管引脚缺失 (只检测到2个引脚)
          6. 开路检测 (网络只连接一个元件端子)
        """
        issues = []
        g = analyzer.graph

        for comp in analyzer.components:
            ctype = norm_component_type(comp.type)

            # --- 1. LED 限流电阻检查 (通过图路径) ---
            if ctype == "LED" and comp.pin2_loc:
                n1 = analyzer._get_node_name(comp.pin1_loc)
                n2 = analyzer._get_node_name(comp.pin2_loc)
                # 检查 LED 所在的两个网络中是否有直接相邻的电阻
                has_resistor = False
                for node in (n1, n2):
                    if node not in g:
                        continue
                    for neighbor in g.neighbors(node):
                        edge_data = g.get_edge_data(node, neighbor)
                        if edge_data and norm_component_type(
                                edge_data.get('type', '')) == "Resistor":
                            has_resistor = True
                            break
                    if has_resistor:
                        break
                if not has_resistor:
                    issues.append(
                        f"{comp.name}: LED所在网络中未检测到限流电阻, "
                        f"建议在{n1}或{n2}串联220Ω-1kΩ电阻")

            # --- 2. 极性未知 ---
            if ctype in POLARIZED_TYPES and comp.polarity == Polarity.UNKNOWN:
                issues.append(f"{comp.name}: {ctype}极性未确定, 请目视检查安装方向")

            # --- 3. 同导通组短路 ---
            if comp.pin2_loc:
                n1 = analyzer._get_node_name(comp.pin1_loc)
                n2 = analyzer._get_node_name(comp.pin2_loc)
                if n1 == n2 and ctype not in ("Wire",):
                    issues.append(
                        f"{comp.name}: {ctype}两引脚在同一导通组({n1}), "
                        f"元件被短路或未正确跨行插入")

            # --- 4. 三极管引脚缺失 ---
            if ctype in THREE_PIN_TYPES and comp.pin3_loc is None:
                issues.append(
                    f"{comp.name}: 三极管仅检测到2个引脚, "
                    f"无法确定B/C/E, 请检查是否正确跨行插入")

        # --- 5. 孤立元件 (度 = 1,只有一端接入网络) ---
        for comp in analyzer.components:
            ctype = norm_component_type(comp.type)
            if ctype == "Wire":
                continue
            nodes_of_comp = set()
            nodes_of_comp.add(analyzer._get_node_name(comp.pin1_loc))
            if comp.pin2_loc:
                nodes_of_comp.add(analyzer._get_node_name(comp.pin2_loc))
            if comp.pin3_loc:
                nodes_of_comp.add(analyzer._get_node_name(comp.pin3_loc))

            for node in nodes_of_comp:
                if node in g and g.degree(node) == 1:
                    # 该节点只有一条边 (就是这个元件自身), 悬空端
                    issues.append(
                        f"{comp.name}: 引脚{node}仅连接到该元件自身, "
                        f"可能为悬空引脚, 无法形成回路")
                    break  # 每个元件只报一次

        # --- 6. 网络连通性: 非连通图意味着电路未闭合 ---
        if g.number_of_nodes() > 0:
            n_components = nx.number_connected_components(g)
            if n_components > 1:
                issues.append(
                    f"电路图有 {n_components} 个独立子网络, "
                    f"可能存在断路或缺少连线")

        return issues



# 全局单例
validator = CircuitValidator()
