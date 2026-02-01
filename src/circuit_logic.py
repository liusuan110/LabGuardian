import networkx as nx
import json
from datetime import datetime
from networkx.readwrite import json_graph

class CircuitComponent:
    def __init__(self, name, type, pin1_loc, pin2_loc=None):
        self.name = name          # e.g., "R1", "LED1"
        self.type = type          # e.g., "resistor", "wire", "chip"
        self.pin1_loc = pin1_loc  # (Row, Col) e.g., ("15", "a")
        self.pin2_loc = pin2_loc  # (Row, Col) e.g., ("20", "a")
        
    def __repr__(self):
        return f"{self.name}({self.pin1_loc}-{self.pin2_loc})"

class CircuitAnalyzer:
    def __init__(self):
        self.graph = nx.Graph()
        self.components = []
        
    def reset(self):
        self.graph.clear()
        self.components = []

    def add_component(self, comp: CircuitComponent):
        self.components.append(comp)
        
        # 核心逻辑：面包板的电气规则 (升级版)
        # 规则1：同一行的 a-e (Left) 是导通的节点
        # 规则2：同一行的 f-j (Right) 是导通的节点
        # 规则3：Left 和 Right 之间默认断路 (由沟槽隔离)，除非元件跨接
        
        def get_node_name(loc_tuple):
            # loc_tuple = ('15', 'a')
            row = loc_tuple[0]
            col = loc_tuple[1]
            if col in ['a', 'b', 'c', 'd', 'e']:
                return f"Row{row}_L" # 左侧节点
            else:
                return f"Row{row}_R" # 右侧节点

        node1 = get_node_name(comp.pin1_loc)
        
        if comp.pin2_loc:
            node2 = get_node_name(comp.pin2_loc)
            # 在图中添加一条边，代表这个元件连接了两个节点
            self.graph.add_edge(node1, node2, component=comp.name, type=comp.type)
        else:
            self.graph.add_node(node1, component=comp.name)

    @staticmethod
    def _norm_type(t: str) -> str:
        if not t:
            return "UNKNOWN"
        u = str(t).strip().upper()
        # normalize common variants
        if "RESIST" in u:
            return "RESISTOR"
        if u == "WIRE" or "WIRE" in u:
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
        """Build a layout-independent topology graph.

        Output is a bipartite undirected graph with:
        - Net nodes: kind='net'
        - Component nodes: kind='comp', ctype='RESISTOR'|'LED'|'SWITCH'|...

        Wires are treated as ideal conductors and are collapsed into nets.
        Breadboard intrinsic connectivity is already collapsed by node naming (RowX_L / RowX_R).
        """
        # 1) Build conductor-only graph (collapse nets)
        conductor = nx.Graph()
        conductor.add_nodes_from(self.graph.nodes())

        for u, v, data in self.graph.edges(data=True):
            et = self._norm_type(data.get("type", ""))
            if et == "WIRE":
                conductor.add_edge(u, v)

        # Each connected component in conductor graph is an electrical net
        net_groups = list(nx.connected_components(conductor))
        node_to_net = {}
        for i, group in enumerate(net_groups):
            for n in group:
                node_to_net[n] = f"N{i}"

        topo = nx.Graph()

        # Add net nodes
        for i in range(len(net_groups)):
            topo.add_node(f"N{i}", kind="net")

        # 2) Add component nodes (exclude wires)
        comp_idx = 0
        for comp in self.components:
            ctype = self._norm_type(comp.type)
            if ctype == "WIRE":
                continue

            # Map component pins -> net ids via Row nodes
            def _node_name(loc_tuple):
                row, col = loc_tuple
                if col in ['a', 'b', 'c', 'd', 'e']:
                    return f"Row{row}_L"
                return f"Row{row}_R"

            try:
                n1 = node_to_net.get(_node_name(comp.pin1_loc))
                n2 = node_to_net.get(_node_name(comp.pin2_loc)) if comp.pin2_loc else None
            except Exception:
                n1, n2 = None, None

            if n1 is None:
                # Component pin not mapped; skip it from topology
                continue

            cid = f"C{comp_idx}"
            comp_idx += 1

            if n2 is None:
                topo.add_node(cid, kind="comp", ctype=ctype, pins=1)
                topo.add_edge(cid, n1)
            else:
                if n1 == n2:
                    # Both pins in same net; represent as 1-net attachment (still a useful error signal)
                    topo.add_node(cid, kind="comp", ctype=ctype, pins=2, same_net=True)
                    topo.add_edge(cid, n1)
                else:
                    topo.add_node(cid, kind="comp", ctype=ctype, pins=2)
                    topo.add_edge(cid, n1)
                    topo.add_edge(cid, n2)

        return topo

    def analyze_circuit(self):
        """
        分析电路连接情况
        """
        connected_groups = list(nx.connected_components(self.graph))
        print("--- Circuit Analysis Report ---")
        print(f"Number of electrical nets: {len(connected_groups)}")
        
        for idx, group in enumerate(connected_groups):
            print(f"Net {idx + 1}: {group}")

    def get_circuit_description(self):
        """
        生成自然语言描述的电路网表，供 LLM 理解
        """
        if not self.components:
            return "No circuit components detected."
            
        desc = "Current Detected Circuit Topology:\n"
        connected_groups = list(nx.connected_components(self.graph))
        
        for idx, group in enumerate(connected_groups):
            # 将集合转换为排序列表以保持稳定性
            nodes = sorted(list(group))
            # 过滤出 Row 节点 (代表连接点)
            # 这里的节点名已经变成了 Row15_L, Row20_R 等
            rows = [n for n in nodes if n.startswith("Row")]
            desc += f"- Net {idx+1} connects: {', '.join(rows)}\n"
            
        # 列出具体的元件连接
        desc += "\nComponent Connections:\n"
        for comp in self.components:
            # 格式: R1 (resistor) connects Row15_L to Row20_L
            side1 = "Left" if comp.pin1_loc[1] <= 'e' else "Right"
            side2 = "Left" if comp.pin2_loc[1] <= 'e' else "Right"
            desc += f"- {comp.name} connects Row{comp.pin1_loc[0]} ({side1}) to Row{comp.pin2_loc[0]} ({side2})\n"
            
        return desc

            
    def validate_connection(self, pin1_row, pin2_row):
        """
        检查两个行是否导通 (Legacy Support, defaults to Left side check)
        """
        node1 = f"Row{pin1_row}_L"
        node2 = f"Row{pin2_row}_L" 
        
        # 尝试左右两边
        if node1 not in self.graph: node1 = f"Row{pin1_row}_R"
        if node2 not in self.graph: node2 = f"Row{pin2_row}_R"
        
        if node1 not in self.graph or node2 not in self.graph:
            return False
            
        return nx.has_path(self.graph, node1, node2)

    def find_missing_link(self, target_netlist):
        """
        对比当前电路和目标电路 (Diff)
        target_netlist: [('Row10_L', 'Row15_L'), ('Row15_R', 'Row20_R')]
        """
        missing = []
        for start, end in target_netlist:
             # 这里假设输入已经包含 _L 或 _R
             if not (self.graph.has_node(start) and self.graph.has_node(end) and nx.has_path(self.graph, start, end)):
                 missing.append(f"{start} -> {end}")
        return sorted(missing)

class CircuitValidator:
    def __init__(self):
        self.ref_graph = None
        self.ref_components = []
        self.ref_topology = None  # layout-independent topology (bipartite)
        # 定义一个简单的“标准网表”结构，用于硬编码的实验
        # 格式: { "Experiment A": [ ("10", "15", "resistor"), ("15", "20", "led") ] }
        self.predefined_experiments = {}

    def set_reference(self, analyzer):
        """
        Record the current circuit state as the 'Golden Reference'
        """
        self.ref_graph = analyzer.graph.copy()
        # Deep copy components list to avoid reference issues
        self.ref_components = [
            CircuitComponent(c.name, c.type, c.pin1_loc, c.pin2_loc) 
            for c in analyzer.components
        ]
        try:
            self.ref_topology = analyzer.build_topology_graph()
        except Exception:
            self.ref_topology = None
        print(f"DEBUG: Reference set with {len(self.ref_components)} components.")

    def save_reference(self, file_path):
        """Save current Golden Reference to a JSON file."""
        if not self.ref_components:
            raise ValueError("No reference circuit set.")

        topo_payload = None
        if self.ref_topology is not None:
            topo_payload = json_graph.node_link_data(self.ref_topology)

        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "format": "labguardian_ref_v1",
            },
            "components": [
                {
                    "name": c.name,
                    "type": c.type,
                    "pin1_loc": list(c.pin1_loc) if c.pin1_loc is not None else None,
                    "pin2_loc": list(c.pin2_loc) if c.pin2_loc is not None else None,
                }
                for c in self.ref_components
            ],
            # New: layout-independent topology template
            "topology": topo_payload,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_reference(self, file_path):
        """Load Golden Reference from a JSON file and rebuild reference graph."""
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        comps = []
        for item in payload.get("components", []):
            pin1 = tuple(item.get("pin1_loc")) if item.get("pin1_loc") else None
            pin2 = tuple(item.get("pin2_loc")) if item.get("pin2_loc") else None
            if pin1 is None:
                continue
            comps.append(CircuitComponent(item.get("name", "UNKNOWN"), item.get("type", "UNKNOWN"), pin1, pin2))

        self.ref_components = comps

        # Rebuild reference graph using the same rules as CircuitAnalyzer
        tmp = CircuitAnalyzer()
        for c in self.ref_components:
            tmp.add_component(c)
        self.ref_graph = tmp.graph.copy()

        topo_payload = payload.get("topology")
        if topo_payload:
            try:
                self.ref_topology = json_graph.node_link_graph(topo_payload)
            except Exception:
                self.ref_topology = None
        else:
            # Backward compatibility for older templates
            try:
                self.ref_topology = tmp.build_topology_graph()
            except Exception:
                self.ref_topology = None

    def compare(self, curr_analyzer):
        """
        Compare current circuit with reference.
        Returns a dictionary with 'errors' (list of strings) and 'missing_links' (list of tuples for AR).
        """
        result = {
            'errors': [],
            'missing_links': [] # List of (Row_Start_Loc, Row_End_Loc) for drawing ghost wires
        }
        
        if not self.ref_components:
            result['errors'].append("No reference circuit set. Cannot validate.")
            return result

        # --- Topology-equivalence check (layout-independent) ---
        # If topology graphs exist, use graph isomorphism to accept different (but equivalent) wiring.
        try:
            if self.ref_topology is not None:
                curr_topo = curr_analyzer.build_topology_graph()

                def _node_match(a, b):
                    if a.get('kind') != b.get('kind'):
                        return False
                    if a.get('kind') == 'comp':
                        return a.get('ctype') == b.get('ctype') and a.get('pins', 2) == b.get('pins', 2)
                    return True

                from networkx.algorithms.isomorphism import GraphMatcher
                gm = GraphMatcher(self.ref_topology, curr_topo, node_match=_node_match)

                if gm.is_isomorphic():
                    result['errors'].append("✅ Topology matches lesson template (layout-independent).")
                    # If topology matches, ignore hole/row placement differences.
                    result['missing_links'] = []
                    return result
                else:
                    # Provide compact diagnostics
                    def _counts(G):
                        comps = [d.get('ctype') for _, d in G.nodes(data=True) if d.get('kind') == 'comp']
                        return len([1 for _, d in G.nodes(data=True) if d.get('kind') == 'net']), comps

                    ref_net_count, ref_comps = _counts(self.ref_topology)
                    cur_net_count, cur_comps = _counts(curr_topo)
                    from collections import Counter
                    rc = Counter(ref_comps)
                    cc = Counter(cur_comps)
                    result['errors'].append("❌ Topology mismatch vs lesson template.")
                    result['errors'].append(f"   Nets: expected {ref_net_count}, found {cur_net_count}")
                    for t in sorted(set(rc.keys()) | set(cc.keys())):
                        if rc[t] != cc[t]:
                            result['errors'].append(f"   {t}: expected {rc[t]}, found {cc[t]}")
                    # Fall through to legacy/heuristic checks for AR hints
        except Exception as e:
            result['errors'].append(f"⚠️ Topology check failed (fallback to heuristic): {e}")

        # 1. Component Count Check
        from collections import Counter
        ref_counts = Counter([c.type for c in self.ref_components])
        curr_counts = Counter([c.type for c in curr_analyzer.components])
        
        all_types = set(ref_counts.keys()) | set(curr_counts.keys())
        for t in all_types:
            r_c = ref_counts[t]
            c_c = curr_counts[t]
            if c_c < r_c:
                result['errors'].append(f"❌ Missing {r_c - c_c} x {t}")
            elif c_c > r_c:
                result['errors'].append(f"⚠️ Extra {c_c - r_c} x {t}")

        # 2. Topology / Connection Check (Heuristic)
        
        matched_curr_indices = set()
        
        for ref_c in self.ref_components:
            # A. Find best match in current
            best_match = None
            min_dist = 999 
            
            ref_row = int(ref_c.pin1_loc[0])
            
            for idx, curr_c in enumerate(curr_analyzer.components):
                if idx in matched_curr_indices: continue
                if curr_c.type != ref_c.type: continue
                
                try:
                    curr_row = int(curr_c.pin1_loc[0])
                    dist = abs(curr_row - ref_row)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = idx
                except:
                    continue
            
            if best_match is not None:
                # Match Found
                matched_curr_indices.add(best_match)
                curr_c = curr_analyzer.components[best_match]
                
                # Check 1: Position Deviation (Tolerance +/- 2 rows)
                if min_dist > 2:
                    result['errors'].append(f"⚠️ {ref_c.type} pos mismatch: Expected Row~{ref_row}, Found Row{curr_c.pin1_loc[0]}")
            else:
                # If no match found for this specific component (even if count is ok-ish), 
                # we can assume this specific connection is missing.
                # Adding to missing_link for AR visualization
                # We want to draw a ghost wire where the Ref component was.
                result['missing_links'].append( (ref_c.pin1_loc, ref_c.pin2_loc) )


        if not result['errors']:
            result['errors'].append("✅ Circuit matches Reference!")
            
        return result

# Global Validator Instance
validator = CircuitValidator()

# --- 单元测试 ---
if __name__ == "__main__":
    analyzer = CircuitAnalyzer()
    
    # 模拟 YOLO + Calibration 识别到的元件
    # 场景：Row10 和 Row15 之间插了一个电阻，Row15 和 Row20 之间插了一个导线
    comps = [
        CircuitComponent("R1", "resistor", ("10", "a"), ("15", "a")),
        CircuitComponent("W1", "wire",     ("15", "e"), ("20", "e")),
        CircuitComponent("LED1", "led",    ("20", "f"), ("21", "f")) # LED串联在后面
    ]
    
    for c in comps:
        analyzer.add_component(c)
        
    analyzer.analyze_circuit()
    
    # 测试连通性
    print(f"Row 10 connected to Row 20? {analyzer.validate_connection('10', '20')}") # 应该 True
    print(f"Row 10 connected to Row 21? {analyzer.validate_connection('10', '21')}") # 应该 True (因为 21连着20的LED... 等等，LED不算导线，但物理上确实连着)
    
    # 注意：NetworkX 的 has_path 只是在图拓扑上查找路径。
    # 在真实电路中，元件也是路径的一部分。所以 validate_connection 返回 True 意味着“物理上连接在一起”。
    
    # 测试纠错
    # 假设目标是 10 -> 25 通路
    missing = analyzer.find_missing_link([('10', '25')])
    print(f"Missing links: {missing}")
