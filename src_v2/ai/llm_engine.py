"""
LLM 推理引擎模块 (v2)
职责：提供统一的 AI 问答接口, 自动选择最优后端

后端优先级 (三级降级策略):
  1. Cloud  — DeepSeek / Qwen 等 OpenAI 兼容云端 API (最强, 需联网)
  2. Local  — Qwen2.5-1.5B / MiniCPM 等小模型, 通过 openvino_genai 在 NPU/GPU 上运行
  3. Rule   — 基于电路领域规则的模板引擎 (零依赖, 离线兜底)

模型选型建议 (Intel Core Ultra 5 225U · DK-2500):
  - 首选: Qwen2.5-1.5B-Instruct (INT4) — 中文最好的 1B 级模型
  - 次选: MiniCPM-1B-sft (INT4) — 清华/面壁, 专为端侧设计
  - 备选: Phi-3-mini-4k-instruct (INT4) — 推理能力强, 中文略弱
  - 弃用: TinyLlama-1.1B — 中文几乎不可用

转换命令 (在有网环境执行一次):
  optimum-cli export openvino \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --weight-format int4 --sym --ratio 1.0 --group-size 128 \\
    --trust-remote-code \\
    models/qwen2.5_1.5b_ov

参考:
  - OpenVINO GenAI NPU Guide: docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide-npu.html
  - openvino_genai LLMPipeline: github.com/openvinotoolkit/openvino.genai
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import llm as llm_cfg

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """LLM 后端抽象基类"""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """生成回复"""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """是否已加载就绪"""
        ...


class CloudLLMBackend(LLMBackend):
    """云端 API 后端 (DeepSeek / Moonshot / Qwen 等 OpenAI 兼容接口)"""

    def __init__(self):
        self.client = None

    def load(self) -> bool:
        if not llm_cfg.is_cloud_ready:
            print("[LLM-Cloud] API Key 未配置. 设置环境变量 LG_API_KEY 或 DEEPSEEK_API_KEY")
            return False
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=llm_cfg.cloud_api_key,
                base_url=llm_cfg.cloud_base_url,
            )
            print(f"[LLM-Cloud] 已连接: {llm_cfg.cloud_model_name} @ {llm_cfg.cloud_base_url}")
            return True
        except Exception as e:
            print(f"[LLM-Cloud] 连接失败: {e}")
            return False

    def is_ready(self) -> bool:
        return self.client is not None

    def generate(self, system_prompt: str, user_message: str) -> str:
        if not self.is_ready():
            return "[Error] Cloud LLM not ready."
        try:
            response = self.client.chat.completions.create(
                model=llm_cfg.cloud_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=llm_cfg.max_tokens,
                temperature=llm_cfg.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Cloud Error] {e}"


class LocalLLMBackend(LLMBackend):
    """
    本地 LLM 后端 (OpenVINO GenAI)

    支持两种加载方式:
    1. openvino_genai.LLMPipeline (推荐, 支持 NPU)
    2. optimum-intel OVModelForCausalLM (回退)

    推荐模型 (INT4 对称量化):
      - Qwen2.5-1.5B-Instruct  → 中文优秀, ~1GB
      - MiniCPM-1B-sft          → 端侧设计, ~0.6GB
      - Phi-3-mini-4k-instruct  → 推理强, 中文弱
    """

    def __init__(self):
        self._pipe = None         # openvino_genai.LLMPipeline
        self._model = None        # OVModelForCausalLM (fallback)
        self._tokenizer = None    # transformers tokenizer (fallback)
        self._backend_type = None # "genai" | "optimum"

    def load(self) -> bool:
        model_path = llm_cfg.local_model_path
        device = llm_cfg.local_device

        if not Path(model_path).exists():
            logger.warning(f"[LLM-Local] 模型路径不存在: {model_path}")
            return False

        # 策略 1: 优先用 openvino_genai (更轻量, 支持 NPU, 无 transformers 依赖)
        if self._try_load_genai(model_path, device):
            return True

        # 策略 2: 回退到 optimum-intel
        if self._try_load_optimum(model_path, device):
            return True

        logger.error("[LLM-Local] 所有加载方式均失败")
        return False

    def _try_load_genai(self, model_path: str, device: str) -> bool:
        """尝试用 openvino_genai.LLMPipeline 加载"""
        try:
            import openvino_genai as ov_genai

            pipeline_config = {
                "MAX_PROMPT_LEN": 1024,
                "MIN_RESPONSE_LEN": 256,
                "NPUW_CACHE_DIR": str(Path(model_path) / ".npucache"),
            }

            logger.info(f"[LLM-Local] 使用 openvino_genai 加载: {model_path} → {device}")
            self._pipe = ov_genai.LLMPipeline(model_path, device, pipeline_config)
            self._backend_type = "genai"
            logger.info(f"[LLM-Local] ✅ openvino_genai LLMPipeline 加载成功 (device={device})")
            return True
        except ImportError:
            logger.info("[LLM-Local] openvino_genai 未安装, 尝试 optimum-intel")
            return False
        except Exception as e:
            logger.warning(f"[LLM-Local] openvino_genai 加载失败: {e}")
            return False

    def _try_load_optimum(self, model_path: str, device: str) -> bool:
        """回退: 用 optimum-intel 加载"""
        try:
            from optimum.intel.openvino import OVModelForCausalLM
            from transformers import AutoTokenizer

            # optimum 的 device 格式不同
            ov_device = "GPU" if device == "GPU" else "CPU"
            logger.info(f"[LLM-Local] 使用 optimum-intel 加载: {model_path} → {ov_device}")
            self._model = OVModelForCausalLM.from_pretrained(model_path, device=ov_device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._backend_type = "optimum"
            logger.info("[LLM-Local] ✅ optimum-intel 加载成功")
            return True
        except Exception as e:
            logger.warning(f"[LLM-Local] optimum-intel 加载失败: {e}")
            return False

    def is_ready(self) -> bool:
        if self._backend_type == "genai":
            return self._pipe is not None
        elif self._backend_type == "optimum":
            return self._model is not None and self._tokenizer is not None
        return False

    def generate(self, system_prompt: str, user_message: str) -> str:
        if not self.is_ready():
            return "[Error] Local LLM not loaded."

        if self._backend_type == "genai":
            return self._generate_genai(system_prompt, user_message)
        else:
            return self._generate_optimum(system_prompt, user_message)

    def _generate_genai(self, system_prompt: str, user_message: str) -> str:
        """使用 openvino_genai LLMPipeline 生成"""
        try:
            # GenAI Pipeline 使用纯文本 prompt (不走 chat template, 手动拼接)
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            import openvino_genai as ov_genai
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = llm_cfg.max_tokens
            # NPU 目前仅支持 greedy decoding
            config.do_sample = False

            result = self._pipe.generate(prompt, config)
            return result.strip()
        except Exception as e:
            logger.error(f"[LLM-Local-GenAI] 生成失败: {e}")
            return f"[Local Error] {e}"

    def _generate_optimum(self, system_prompt: str, user_message: str) -> str:
        """使用 optimum-intel 生成"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self._tokenizer(
                input_text, return_tensors="pt"
            ).input_ids.to(self._model.device)

            output = self._model.generate(
                input_ids,
                max_new_tokens=llm_cfg.max_tokens,
                temperature=llm_cfg.temperature,
            )
            answer = self._tokenizer.decode(output[0], skip_special_tokens=True)
            # 尝试多种分隔符提取 assistant 回复
            for sep in ("<|im_start|>assistant\n", "<|assistant|>", "assistant\n"):
                if sep in answer:
                    return answer.split(sep)[-1].strip()
            return answer.strip()
        except Exception as e:
            logger.error(f"[LLM-Local-Optimum] 生成失败: {e}")
            return f"[Local Error] {e}"


# ====================================================================
# 第三级: 规则引擎后端 (零依赖, 离线兜底)
# ====================================================================

class RuleBasedBackend(LLMBackend):
    """
    基于领域规则的模板回复引擎

    当云端不可用且本地模型也失败时, 使用此后端提供基本的电路分析能力。
    不需要任何 ML 依赖, 纯 Python 字符串处理。

    能力范围:
    - 解读电路网表, 回答 "XX连在哪里"
    - 识别常见接线错误并给出提示
    - 回答元件基础知识 (离线知识库)
    """

    # 元件基础知识库
    KNOWLEDGE_BASE = {
        "RESISTOR": "电阻器: 限制电流/分压。无极性, 可双向安装。色环读数从左到右。",
        "LED": "发光二极管: 有极性! 长脚为阳极(+), 短脚为阴极(-)。必须串联限流电阻(通常220Ω-1kΩ)。",
        "DIODE": "二极管: 有极性! 有银色/白色色环的一端为阴极(-)。电流只能从阳极流向阴极。",
        "CAPACITOR": "电容器: 电解电容有极性(长脚正极)! 陶瓷电容无极性。注意耐压值。",
        "WIRE": "导线/跳线: 无极性, 用于连接面包板不同行。",
        "Push_Button": "按钮开关: 按下导通, 松开断开。四脚按钮注意对角连通。",
        "TRANSISTOR": "三极管(BJT): TO-92封装, 平面朝自己时引脚从左到右为 E/B/C。注意型号(NPN/PNP)。",
        "NPN": "NPN三极管: 基极(B)高电平时导通, 电流从集电极(C)流向发射极(E)。",
        "PNP": "PNP三极管: 基极(B)低电平时导通, 电流从发射极(E)流向集电极(C)。",
    }

    # 常见错误检查规则
    ERROR_PATTERNS = {
        "LED无限流电阻": "⚠️ 检测到LED直接跨接在电源轨之间, 缺少限流电阻! 建议串联220Ω-1kΩ电阻。",
        "二极管可能反接": "⚠️ 二极管极性可能接反。请检查: 银色/白色环标记的一端应接向低电位(GND)方向。",
        "短路风险": "🔴 检测到电源正负极之间缺少负载, 存在短路风险!",
    }

    def __init__(self):
        self._ready = True

    def load(self) -> bool:
        self._ready = True
        logger.info("[LLM-Rule] 规则引擎已就绪 (离线模式)")
        return True

    def is_ready(self) -> bool:
        return self._ready

    def generate(self, system_prompt: str, user_message: str) -> str:
        """基于规则和模板的回复生成"""
        question = user_message.strip()
        context = system_prompt  # system_prompt 中包含电路网表

        # 1. 元件知识查询
        for comp_type, info in self.KNOWLEDGE_BASE.items():
            if comp_type.lower() in question.lower():
                return self._format_knowledge_reply(comp_type, info, context)

        # 2. 连接查询 ("XX连在哪里", "XX怎么接的")
        if any(kw in question for kw in ("连接", "连在", "怎么接", "接在", "连到")):
            return self._parse_connections_from_context(question, context)

        # 3. 检查/验证类问题
        if any(kw in question for kw in ("检查", "有没有错", "对不对", "正确", "问题")):
            return self._check_circuit_issues(context)

        # 4. 通用回复
        return (
            "我是 LabGuardian 电路助手 (离线模式)。\n"
            "你可以问我:\n"
            "• 某个元件的基础知识 (如 '电阻是什么')\n"
            "• 当前电路的连接情况 (如 'LED连在哪里')\n"
            "• 电路检查 (如 '检查一下有没有错')\n\n"
            "提示: 联网后可使用更强大的 AI 问答。"
        )

    def _format_knowledge_reply(self, comp_type: str, info: str, context: str) -> str:
        """格式化元件知识回复, 结合当前电路上下文"""
        reply = f"📖 {info}\n"

        # 从 context 中找到该元件的连接信息
        connections = []
        for line in context.split("\n"):
            if comp_type in line.upper() and "Row" in line:
                connections.append(line.strip().lstrip("- "))

        if connections:
            reply += f"\n在当前电路中:\n"
            for c in connections[:5]:
                reply += f"  • {c}\n"

        return reply

    def _parse_connections_from_context(self, question: str, context: str) -> str:
        """从电路上下文中解析连接信息"""
        if not context or "暂无电路数据" in context:
            return "当前暂无电路检测数据。请确保摄像头对准面包板, 并已完成校准。"

        # 提取 Component Connections 部分
        lines = []
        in_comp_section = False
        for line in context.split("\n"):
            if "Component Connections" in line:
                in_comp_section = True
                continue
            if in_comp_section and line.startswith("-"):
                lines.append(line.strip())
            elif in_comp_section and not line.strip():
                break

        if not lines:
            return "未检测到元件连接信息。"

        return "当前检测到的电路连接:\n" + "\n".join(lines)

    def _check_circuit_issues(self, context: str) -> str:
        """基于规则检查常见电路错误"""
        issues = []

        # 检查 LED 是否有限流电阻
        if "LED" in context:
            has_resistor_near_led = "RESISTOR" in context
            if not has_resistor_near_led:
                issues.append(self.ERROR_PATTERNS["LED无限流电阻"])

        # 检查极性问题 (从 context 中寻找极性标记)
        if "polarity: UNKNOWN" in context:
            issues.append("⚠️ 部分有极性元件的方向无法确定, 请目视检查。")

        if "接反" in context or "REVERSE" in context:
            issues.append(self.ERROR_PATTERNS["二极管可能反接"])

        if not issues:
            return "✅ 基于规则检查, 未发现明显的接线错误。(注: 离线模式仅检查常见问题)"

        return "检查结果:\n" + "\n".join(issues)


class LLMEngine:
    """
    统一 LLM 推理引擎 (v2) + RAG 知识增强

    三级降级策略:
      1. Cloud  → DeepSeek / Qwen 等 (需联网, 最强)
      2. Local  → Qwen2.5 / MiniCPM 等 via OpenVINO (离线, 中等)
      3. Rule   → 领域规则模板引擎 (零依赖, 兜底)

    RAG 增强:
      - 自动检索本地知识库 (ChromaDB) 中的相关片段
      - 将检索结果注入 system prompt, 提升回答准确性
      - RAG 不可用时优雅降级, 不影响主流程

    使用方式:
        engine = LLMEngine()
        engine.load()
        answer = engine.ask("8050三极管引脚怎么接？", circuit_context="...")
    """

    def __init__(self, enable_rag: bool = True):
        self.cloud = CloudLLMBackend()
        self.local = LocalLLMBackend()
        self.rules = RuleBasedBackend()
        self._active: Optional[LLMBackend] = None
        self._enable_rag = enable_rag
        self.rag = None  # RAGEngine instance (lazy init)

    def load(self) -> str:
        """
        加载 LLM 后端 + RAG 知识库, 按优先级尝试, 返回加载状态描述
        """
        # 0. 初始化 RAG 知识库 (非阻塞, 失败不影响 LLM)
        rag_status = self._init_rag()

        # 1. 优先尝试云端
        if llm_cfg.use_cloud and self.cloud.load():
            self._active = self.cloud
            status = f"☁️ Cloud AI Ready: {llm_cfg.cloud_model_name}"
            logger.info(status)
            return f"{status}\n{rag_status}"

        # 2. 本地 OpenVINO 模型
        if self.local.load():
            self._active = self.local
            status = f"💻 Local AI Ready ({self.local._backend_type})"
            logger.info(status)
            return f"{status}\n{rag_status}"

        # 3. 规则引擎兜底 (永远成功)
        self.rules.load()
        self._active = self.rules
        status = "📋 Rule-Based AI Ready (离线模式)"
        logger.info(status)
        return f"{status}\n{rag_status}"

    def _init_rag(self) -> str:
        """初始化 RAG 引擎 (优雅降级)"""
        if not self._enable_rag:
            return "📚 RAG: 已禁用"
        try:
            from ai.rag_engine import RAGEngine
            self.rag = RAGEngine()
            if self.rag.initialize():
                # 如果知识库为空, 自动构建索引
                if self.rag.doc_count == 0:
                    logger.info("[RAG] 知识库为空, 自动构建索引...")
                    self.rag.build_index()
                count = self.rag.doc_count
                return f"📚 RAG Ready: {count} 知识块已加载"
            else:
                self.rag = None
                return "📚 RAG: 初始化失败 (降级为无知识增强)"
        except ImportError as e:
            logger.warning(f"[RAG] 依赖缺失: {e}")
            self.rag = None
            return "📚 RAG: 依赖未安装 (pip install chromadb sentence-transformers)"
        except Exception as e:
            logger.warning(f"[RAG] 初始化异常: {e}")
            self.rag = None
            return f"📚 RAG: 初始化异常 ({e})"

    @property
    def is_ready(self) -> bool:
        return self._active is not None and self._active.is_ready()

    @property
    def backend_name(self) -> str:
        if self._active is self.cloud:
            return f"Cloud ({llm_cfg.cloud_model_name})"
        elif self._active is self.local:
            return f"Local-{self.local._backend_type}"
        elif self._active is self.rules:
            return "Rule-Based (Offline)"
        return "None"

    @property
    def is_offline_fallback(self) -> bool:
        """当前是否在使用规则引擎兜底"""
        return self._active is self.rules

    def switch_backend(self, backend: str) -> str:
        """
        手动切换后端 (用于调试或演示)

        Args:
            backend: "cloud" | "local" | "rules"
        """
        if backend == "cloud" and self.cloud.is_ready():
            self._active = self.cloud
            return f"已切换到 Cloud 后端"
        elif backend == "local" and self.local.is_ready():
            self._active = self.local
            return f"已切换到 Local 后端"
        elif backend == "rules":
            self.rules.load()
            self._active = self.rules
            return "已切换到 Rule-Based 后端"
        return f"切换失败: {backend} 后端不可用"

    @property
    def rag_ready(self) -> bool:
        """RAG 知识库是否可用"""
        return self.rag is not None and self.rag.is_ready

    def ask(self, question: str, circuit_context: str = "") -> str:
        """
        向 AI 提问 (带电路上下文 + RAG 知识增强)
        
        Args:
            question: 用户问题
            circuit_context: 当前电路的网表描述
            
        Returns:
            AI 回复文本
        """
        if not self.is_ready:
            return "AI 引擎未就绪。请检查 API Key 配置或本地模型。"

        # RAG 检索: 从知识库获取相关上下文
        rag_context = ""
        if self.rag_ready:
            try:
                rag_context = self.rag.get_context(question, top_k=3, min_score=0.35)
            except Exception as e:
                logger.warning(f"[RAG] 检索失败: {e}")

        system_prompt = self._build_system_prompt(circuit_context, rag_context)
        return self._active.generate(system_prompt, question)

    def ask_about_component(self, component_name: str, circuit_context: str = "") -> str:
        """快捷方法：询问某个元件的信息"""
        question = f"我正在看一个 {component_name}，请告诉我它在这个电路中是怎么连接的？"
        return self.ask(question, circuit_context)

    @staticmethod
    def _build_system_prompt(circuit_context: str, rag_context: str = "") -> str:
        # 构建知识库参考部分
        knowledge_section = ""
        if rag_context:
            knowledge_section = f"""\n\n你还拥有以下参考资料 (来自本地芯片手册/实验知识库):
{rag_context}
请优先参考这些资料回答, 并在回答中简要注明出处。"""

        return f"""你是一个专业的电子实验室助手 (LabGuardian)。
你拥有计算机视觉系统提供的实时电路网表数据：
{circuit_context if circuit_context else '(当前暂无电路数据)'}
{knowledge_section}
请基于此电路状态和参考资料回答用户问题。
- 如果被问及连接，请根据 Net (网络) 信息判断。
- Push_Button = 按钮开关, Wire = 导线。
- 请用中文回答，像一个人类助教一样自然。
- 回答简洁，不要超过 300 字。"""
