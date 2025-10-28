#!/usr/bin/env bash
set -euo pipefail

COMPETITORS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
CONFIG="$COMPETITORS_DIR/config.yaml"

python "$COMPETITORS_DIR/scripts/setup_repo.py" --config "$CONFIG"
python "$COMPETITORS_DIR/scripts/setup_tasks.py" --config "$CONFIG"

echo "[setup] Done"
