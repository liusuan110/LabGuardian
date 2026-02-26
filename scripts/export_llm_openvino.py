#!/usr/bin/env python3
"""
LabGuardian — 本地 LLM 模型转换脚本

将 HuggingFace 模型转换为 OpenVINO INT4 格式, 用于 NPU/GPU 离线推理。

使用方法:
  1. 安装依赖:
     pip install optimum[openvino] nncf

  2. 运行转换 (需联网, 仅执行一次):
     python scripts/export_llm_openvino.py --model qwen2.5-1.5b
     python scripts/export_llm_openvino.py --model minicpm-1b
     python scripts/export_llm_openvino.py --model phi3-mini

  3. 转换完成后, 将 models/<model_name> 目录拷贝到 DK-2500 即可离线使用。

推荐模型 (按优先级):
  ┌──────────────────────────┬──────────┬──────────┬────────────┐
  │ 模型                      │ 参数量    │ 中文能力  │ INT4 大小   │
  ├──────────────────────────┼──────────┼──────────┼────────────┤
  │ Qwen2.5-1.5B-Instruct   │ 1.5B     │ ★★★★★   │ ~1.0 GB    │
  │ MiniCPM-1B-sft-bf16     │ 1.2B     │ ★★★★    │ ~0.7 GB    │
  │ Phi-3-mini-4k-instruct  │ 3.8B     │ ★★★     │ ~2.2 GB    │
  │ Qwen2.5-0.5B-Instruct   │ 0.5B     │ ★★★     │ ~0.4 GB    │
  └──────────────────────────┴──────────┴──────────┴────────────┘

技术参考:
  - OpenVINO GenAI NPU Guide:
    docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide-npu.html
  - NPU 要求对称 INT4 量化 (--sym --ratio 1.0)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 模型注册表
MODELS = {
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "output_dir": "qwen2.5_1.5b_ov",
        "group_size": 128,
        "description": "中文最佳 1.5B 级模型 (首选)",
    },
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_dir": "qwen2.5_0.5b_ov",
        "group_size": 128,
        "description": "极小模型, 适合内存紧张的场景",
    },
    "minicpm-1b": {
        "hf_id": "openbmb/MiniCPM-1B-sft-bf16",
        "output_dir": "minicpm_1b_ov",
        "group_size": 128,
        "description": "清华/面壁端侧模型, 中文优秀",
    },
    "phi3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "output_dir": "phi3_mini_ov",
        "group_size": 128,
        "description": "微软 Phi-3, 推理能力强",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def export_model(model_key: str, device_target: str = "NPU"):
    """执行模型转换"""
    if model_key not in MODELS:
        print(f"❌ 未知模型: {model_key}")
        print(f"   可选: {', '.join(MODELS.keys())}")
        return False

    info = MODELS[model_key]
    output_path = MODELS_DIR / info["output_dir"]

    if output_path.exists() and any(output_path.glob("*.xml")):
        print(f"⚠️ 模型已存在: {output_path}")
        print("   如需重新转换, 请先删除该目录")
        return True

    print(f"{'='*60}")
    print(f"📦 转换模型: {info['hf_id']}")
    print(f"   {info['description']}")
    print(f"   输出目录: {output_path}")
    print(f"   量化: INT4 对称, group_size={info['group_size']}")
    print(f"   目标设备: {device_target}")
    print(f"{'='*60}")

    # NPU 需要对称 INT4 量化
    cmd = [
        sys.executable, "-m", "optimum.exporters.openvino",
        "--model", info["hf_id"],
        "--weight-format", "int4",
        "--sym",
        "--ratio", "1.0",
        "--group-size", str(info["group_size"]),
        "--trust-remote-code",
        str(output_path),
    ]

    # 等效的 CLI 命令 (供手动执行):
    cli_cmd = (
        f"optimum-cli export openvino "
        f"--model {info['hf_id']} "
        f"--weight-format int4 --sym --ratio 1.0 "
        f"--group-size {info['group_size']} "
        f"--trust-remote-code "
        f"{output_path}"
    )
    print(f"\n💡 等效 CLI 命令:\n   {cli_cmd}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 转换成功: {output_path}")
        print(f"   模型大小: {_dir_size_mb(output_path):.1f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 转换失败: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ 未找到 optimum。请先安装:")
        print("   pip install optimum[openvino] nncf")
        return False


def verify_model(model_key: str):
    """验证已转换的模型可否加载"""
    if model_key not in MODELS:
        return

    info = MODELS[model_key]
    output_path = MODELS_DIR / info["output_dir"]

    if not output_path.exists():
        print(f"❌ 模型不存在: {output_path}")
        return

    print(f"\n🔍 验证模型: {output_path}")

    # 尝试 openvino_genai
    try:
        import openvino_genai as ov_genai
        pipe = ov_genai.LLMPipeline(str(output_path), "CPU")  # CPU 验证即可
        result = pipe.generate("你好", max_new_tokens=20, do_sample=False)
        print(f"   ✅ openvino_genai 验证通过")
        print(f"   回复: {result[:100]}")
        return
    except ImportError:
        print("   ⚠️ openvino_genai 未安装, 尝试 optimum")
    except Exception as e:
        print(f"   ⚠️ openvino_genai 加载失败: {e}")

    # 回退 optimum
    try:
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        model = OVModelForCausalLM.from_pretrained(str(output_path), device="CPU")
        tokenizer = AutoTokenizer.from_pretrained(str(output_path))
        inputs = tokenizer("你好", return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=20)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"   ✅ optimum-intel 验证通过")
        print(f"   回复: {result[:100]}")
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")


def _dir_size_mb(path: Path) -> float:
    """计算目录大小 (MB)"""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="LabGuardian LLM 模型转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/export_llm_openvino.py --model qwen2.5-1.5b
  python scripts/export_llm_openvino.py --model qwen2.5-1.5b --verify
  python scripts/export_llm_openvino.py --list
        """,
    )
    parser.add_argument(
        "--model", type=str, choices=list(MODELS.keys()),
        help="要转换的模型"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="转换后验证模型"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用模型"
    )
    parser.add_argument(
        "--device", type=str, default="NPU",
        choices=["CPU", "GPU", "NPU"],
        help="目标推理设备"
    )

    args = parser.parse_args()

    if args.list:
        print("\n📋 可用模型列表:")
        print(f"{'─'*60}")
        for key, info in MODELS.items():
            output_path = MODELS_DIR / info["output_dir"]
            status = "✅ 已转换" if output_path.exists() else "⬜ 未转换"
            print(f"  {key:20s} {status}  {info['description']}")
        print(f"{'─'*60}")
        print(f"\n模型存储目录: {MODELS_DIR}")
        return

    if not args.model:
        parser.print_help()
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    success = export_model(args.model, args.device)

    if success and args.verify:
        verify_model(args.model)


if __name__ == "__main__":
    main()
