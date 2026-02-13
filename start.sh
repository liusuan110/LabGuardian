#!/usr/bin/env bash
# ============================================================
# LabGuardian 一键启动脚本 — DK-2500 / Ubuntu
# ============================================================
# 用法:
#   chmod +x start.sh     (首次)
#   ./start.sh             正常启动
#   ./start.sh --diag      仅诊断
#   ./start.sh --watchdog  评审模式（崩溃自动重启）
# ============================================================

set -euo pipefail

# --- 定位项目根目录 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src_v2"

echo "╔══════════════════════════════════════╗"
echo "║      LabGuardian — 启动中...         ║"
echo "╚══════════════════════════════════════╝"
echo "项目根: $SCRIPT_DIR"

# --- 激活虚拟环境 ---
VENV_DIR=""
for candidate in "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/venv" "$SCRIPT_DIR/env"; do
    if [ -f "$candidate/bin/activate" ]; then
        VENV_DIR="$candidate"
        break
    fi
done

if [ -n "$VENV_DIR" ]; then
    echo "激活虚拟环境: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "⚠  未找到虚拟环境，使用系统 Python"
fi

# --- 验证 Python ---
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
if [ -z "$PYTHON" ]; then
    echo "❌ 未找到 Python，请安装 Python 3.10+"
    exit 1
fi
echo "Python: $($PYTHON --version) → $PYTHON"

# --- 快速依赖检查 ---
echo "检查核心依赖..."
$PYTHON -c "
import sys
missing = []
for mod in ['PySide6', 'cv2', 'numpy', 'ultralytics', 'networkx']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f'❌ 缺少依赖: {missing}')
    print('   请执行: pip install ' + ' '.join(missing))
    sys.exit(1)
print('✔ 核心依赖完备')
" || exit 1

# --- 设置 DK-2500 特定环境变量 ---
# 如果在 DK-2500 上且存在 NPU，设置 OpenVINO 设备
if [ -d "/opt/intel" ]; then
    echo "检测到 Intel 平台"
    export LG_OV_DEVICE="${LG_OV_DEVICE:-NPU}"
fi

# 确保摄像头使用 V4L2
if [ -e "/dev/video0" ]; then
    echo "摄像头: /dev/video0 (V4L2)"
    export LG_CAMERA_BACKEND="${LG_CAMERA_BACKEND:-v4l2}"
fi

# --- 字体检查（CJK） ---
if command -v fc-list &>/dev/null; then
    CJK_FONTS=$(fc-list :lang=zh 2>/dev/null | head -1)
    if [ -z "$CJK_FONTS" ]; then
        echo "⚠  未检测到中文字体，UI 文字可能异常"
        echo "   建议安装: sudo apt install fonts-noto-cjk"
    fi
fi

# --- 启动 ---
cd "$SRC_DIR"
echo ""
echo "════════ 启动 LabGuardian ════════"
exec $PYTHON launcher.py "$@"
