import schemdraw
import schemdraw.elements as elm
import networkx as nx
import matplotlib.pyplot as plt

class SchematicGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def generate_schematic(self, show=True):
        """
        根据 CircuitAnalyzer 的组件列表绘制原理图
        采用“物理拓扑映射”策略：直接将面包板坐标映射为原理图坐标
        """
        if not self.analyzer.components:
            print("No components to draw.")
            return

        with schemdraw.Drawing(show=False) as d:
            d.config(fontsize=12)
            
            # --- 坐标映射配置 ---
            # 【Day 3 适配】 如果 breadboard 是 Vertical Conducting (Node = Column / Strip index),
            # 那么 row_num (1..30) 实际上对应 X 轴坐标
            
            # X轴：对应面包板 Strip Index 
            X_SCALE = 1.0 
            
            # Y轴：对应 Position within strip (Top side / Bottom side)
            # 我们把 Top side (a-e) 放在上面, Bottom side (f-j) 放在下面
            # 中间沟槽
            
            Y_OFFSET_TOP = 0
            Y_OFFSET_BOTTOM = -4
            
            # 辅助函数：将逻辑坐标 ('15', 'a') 转为绘图坐标 (x, y)
            def get_xy(loc_tuple):
                row_str, col_str = loc_tuple
                try:
                    # row_str is technically the "Strip Index" (X) now
                    strip_idx = int(row_str)
                except:
                    strip_idx = 0
                
                # X轴展开：Strip 1 在左，Strip 30 在右
                x = strip_idx * X_SCALE
                
                # Y轴分上下
                # a-e 在上 (Y=0 area), f-j 在下 (Y=-4 area)
                if col_str <= 'e':
                    y = Y_OFFSET_TOP
                else:
                    y = Y_OFFSET_BOTTOM
                
                return (x, y)

            # --- 绘制元件 ---
            # 记录已使用的节点坐标，用于绘制连接点(Dot)
            used_nodes = set()

            for i, comp in enumerate(self.analyzer.components):
                
                start_xy = get_xy(comp.pin1_loc)
                used_nodes.add(start_xy)
                
                # 如果是双端元件
                if comp.pin2_loc:
                    end_xy = get_xy(comp.pin2_loc)
                    used_nodes.add(end_xy)
                    
                    # 优化标签显示：避免 "Resistor\n(Resistor)" 这种重复
                    if comp.name == comp.type:
                        label = f"{comp.name} {i+1}" # e.g. Resistor 1
                    else:
                        label = f"{comp.name}\n{comp.type}"
                    
                    # 根据元件类型选择符号
                    element = None
                    # 注意：SchemDraw 的 .to() 方法会自动计算角度和长度
                    
                    if "RESISTOR" in comp.type:
                        element = elm.Resistor().at(start_xy).to(end_xy).label(label)
                    elif "LED" in comp.type:
                        element = elm.LED().at(start_xy).to(end_xy).label(label)
                    elif "CAPACITOR" in comp.type:
                        element = elm.Capacitor().at(start_xy).to(end_xy).label(label)
                    elif "Button" in comp.type or "Switch" in comp.type:
                        element = elm.Switch().at(start_xy).to(end_xy).label(label)
                    elif "Wire" in comp.type:
                        # 导线用直线，颜色也可以区分
                        element = elm.Line().at(start_xy).to(end_xy).color('blue')
                    else:
                        # 默认用方框或通用阻抗
                        element = elm.ResistorBox().at(start_xy).to(end_xy).label(label)
                    
                    d.add(element)
                    
                    # 标注节点名称 (可选)
                    # d.add(elm.Dot().at(start_xy))
                    # d.add(elm.Dot().at(end_xy))

            # --- 绘制连接点 (Junctions) ---
            # 在所有有引脚连接的地方画黑点，强调电气连接
            for xy in used_nodes:
                d.add(elm.Dot(radius=0.12).at(xy))

            # --- 绘制电源轨 (示意) ---
            # 在最左侧绘制 VCC/GND 提示
            # d.add(elm.Vdd().at((-2, -1)).label("Power Rail"))
            
            if show:
                # 使用 Matplotlib 后端显示
                d.draw()
                # 能够保存图片
                # d.save('current_circuit.png')
                
        return d
