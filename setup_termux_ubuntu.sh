#!/usr/bin/env bash
# =============================================================================
# Gemini Editor — AnLinux Ubuntu (Termux PRoot ARM64) One-Command Setup
# =============================================================================
# Usage inside AnLinux Ubuntu (Termux):
#   bash setup_termux_ubuntu.sh
#   source .venv/bin/activate && python main.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# AnLinux Ubuntu usually runs as root inside PRoot (sudo may not be installed)
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "============================================================"
echo " Gemini Editor — AnLinux Ubuntu (Termux) Setup"
echo " Arch: $(uname -m) | OS: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-Ubuntu}" || echo "Linux")"
echo "============================================================"

# 1. Install FFmpeg + fonts + prebuilt ARM64 Python native packages via apt
#    (avoids compiling OpenCV / NumPy / Pillow / Cryptography / PyNaCl on phone)
echo "[1/4] Installing FFmpeg, fonts, and prebuilt ARM64 Python binaries..."
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
  ffmpeg \
  fonts-dejavu-core \
  git \
  curl \
  ca-certificates \
  python3 \
  python3-pip \
  python3-venv \
  python3-opencv \
  python3-numpy \
  python3-pil \
  python3-cryptography

# 2. Create virtualenv sharing system site-packages (uses apt binaries directly)
echo "[2/4] Setting up Python virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel

# 3. Install pure-Python mobile requirements (zero Rust/C++ compilation)
echo "[3/4] Installing pure-Python dependencies (requirements-termux.txt)..."
pip install --prefer-binary -r requirements-termux.txt

# Optional lightweight faster-whisper ('tiny' model ~75MB, corrected by Gemini downstream)
echo "Installing faster-whisper (tiny model default; non-fatal if wheel unavailable on this Python/arch)..."
pip install --prefer-binary "faster-whisper>=1.0.0" || \
  echo "WARNING: faster-whisper skipped (no prebuilt wheel); pipeline will still run cleanly."

# Safety fallback: verify native apt modules exist; pip-install only if missing
python - << 'EOF'
import importlib, subprocess, sys

native_map = {
    "cv2": "opencv-python-headless>=4.9.0.80,<5.0.0",
    "numpy": "numpy>=1.26.0,<2.2.0",
    "PIL": "Pillow>=10.0.0",
    "cryptography": "cryptography>=42.0.0",
}
missing = []
for mod, pkg in native_map.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(pkg)

if missing:
    print("Installing missing native packages via pip:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--prefer-binary", *missing])
else:
    print("All native ARM64 modules verified via Ubuntu apt packages.")
EOF

# 4. Verify FFmpeg drawtext & core pipeline imports
echo "[4/4] Verifying FFmpeg and core imports..."
ffmpeg -version | head -n 1
if ffmpeg -hide_banner -filters 2>/dev/null | grep -qw drawtext; then
  echo "OK: FFmpeg drawtext filter available."
else
  echo "WARNING: FFmpeg drawtext filter not detected."
fi

python - << 'EOF'
import sys
for mod in [
    "cv2", "numpy", "PIL", "dotenv", "requests", "httpx",
    "telegram", "pyrogram", "google.genai", "json_repair", "jsonschema",
    "edge_tts", "yt_dlp", "cryptography",
    "Gemini_Modules.gemini_ffmpeg_synthesis",
    "Phase_1.phase1_orchestrator",
    "Phase_2.phase2_orchestrator",
    "Phase_3.phase3_orchestrator",
]:
    __import__(mod)
print("OK: All core Gemini Editor modules imported cleanly!")
EOF

mkdir -p Credentials downloads cache logs temp output

echo ""
echo "============================================================"
echo " Setup Complete!"
echo " Next steps:"
echo "   1. Place your keys in Credentials/.env (GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, etc.)"
echo "   2. Start the bot:"
echo "      source .venv/bin/activate"
echo "      python main.py"
echo "   Tip: Run 'termux-wake-lock' in Termux before launching Ubuntu"
echo "        so Android doesn't sleep the process while rendering."
echo "============================================================"
