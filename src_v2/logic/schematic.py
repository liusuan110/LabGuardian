"""
原理图可视化模块
职责：根据 CircuitAnalyzer 的元件列表绘制电路原理图
"""

import schemdraw
import schemdraw.elements as elm

from .circuit import CircuitAnalyzer


class SchematicGenerator:
    """基于 schemdraw 的电路原理图生成器"""

    X_SCALE = 1.0
    Y_OFFSET_TOP = 0
    Y_OFFSET_BOTTOM = -4

    def __init__(self, analyzer: CircuitAnalyzer):
        self.analyzer = analyzer

    def generate_schematic(self, show: bool = True, save_path: str = None):
        """
        根据元件列表绘制原理图
        
        Args:
            show: 是否弹窗显示
            save_path: 保存路径 (可选, e.g. "circuit.png")
        """
        if not self.analyzer.components:
            print("[Schematic] No components to draw.")
            return None

        with schemdraw.Drawing(show=False) as d:
            d.config(fontsize=12)

            used_nodes = set()

            for i, comp in enumerate(self.analyzer.components):
                start_xy = self._get_xy(comp.pin1_loc)
                used_nodes.add(start_xy)

                if comp.pin2_loc:
                    end_xy = self._get_xy(comp.pin2_loc)
                    used_nodes.add(end_xy)

                    label = (f"{comp.name} {i + 1}" if comp.name == comp.type
                             else f"{comp.name}\n{comp.type}")

                    element = self._type_to_element(comp.type, start_xy, end_xy, label)
                    d.add(element)

            for xy in used_nodes:
                d.add(elm.Dot(radius=0.12).at(xy))

            if save_path:
                d.save(save_path)

            if show:
                d.draw()

        return d

    @classmethod
    def _get_xy(cls, loc_tuple):
        """将逻辑坐标 ('15', 'a') 转为绘图坐标 (x, y)"""
        row_str, col_str = loc_tuple
        try:
            strip_idx = int(row_str)
        except (ValueError, TypeError):
            strip_idx = 0

        x = strip_idx * cls.X_SCALE
        y = cls.Y_OFFSET_TOP if col_str <= 'e' else cls.Y_OFFSET_BOTTOM
        return (x, y)

    @staticmethod
    def _type_to_element(comp_type, start, end, label):
        """根据元件类型选择 schemdraw 符号"""
        t = comp_type.upper()

        if "RESISTOR" in t or "RESIST" in t:
            return elm.Resistor().at(start).to(end).label(label)
        elif "LED" in t:
            return elm.LED().at(start).to(end).label(label)
        elif "CAPACITOR" in t or "CAP" in t:
            return elm.Capacitor().at(start).to(end).label(label)
        elif "BUTTON" in t or "SWITCH" in t:
            return elm.Switch().at(start).to(end).label(label)
        elif "WIRE" in t:
            return elm.Line().at(start).to(end).color('blue')
        else:
            return elm.ResistorBox().at(start).to(end).label(label)
