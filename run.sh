#!/usr/bin/env bash
# Convenience launcher for the Raspberry Pi.
# Usage: ./run.sh [--video rtsp://...] [--test] [--ocr-image path.jpg]
set -e
cd "$(dirname "$0")"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
export PYTHONUTF8=1
python3 -m detection.main "$@"
