#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPO_ROOT"
./download_example_datasets.sh
PYTHONPATH=src conda run -n agentomics-datasets python src/agentomics/prepare_datasets.py --prepare-all
