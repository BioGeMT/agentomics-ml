#!/usr/bin/env bash
# Docker-only script.
# Local-mode execution is not supported yet and must be added
set -euo pipefail

AGENT_ID="${1:-}"
if [[ -z "$AGENT_ID" ]]; then
  echo "Usage: $0 <AGENT_ID>"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR_HOST="$REPO_ROOT/outputs/$AGENT_ID"

if [[ ! -d "$AGENT_DIR_HOST" ]]; then
  echo "ERROR: Output folder not found at: $AGENT_DIR_HOST"
  echo "Available folders:"
  ls -1 "$REPO_ROOT/outputs" || true
  exit 1
fi

if ! docker image inspect agentomics_img >/dev/null 2>&1; then
  echo "agentomics_img not found, building..."
  docker build -t agentomics_img -f Dockerfile .
fi

echo "Generating PDFs for: $AGENT_ID"
echo "Host outputs: $AGENT_DIR_HOST"

# Matplotlib warning fix: make config/cache writable in container
MPLCONFIGDIR_IN_CONTAINER="/tmp/mplconfig"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e MPLCONFIGDIR="$MPLCONFIGDIR_IN_CONTAINER" \
  -v "$REPO_ROOT":/repository \
  -v "$AGENT_DIR_HOST":/agent_out \
  --entrypoint /opt/conda/envs/agentomics-env/bin/python \
  agentomics_img /repository/src/generate_final_reports.py \
    --agent-dir /agent_out

echo "Done. See: outputs/$AGENT_ID/final_reports/"
