#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPETITORS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$COMPETITORS_DIR/data"

MODE="${1:-all}"
case "$MODE" in
    all|proteingym|polaris)
        ;;
    *)
        echo "[prepare_biomlbench_tasks] Invalid mode '$MODE'. Use: all | proteingym | polaris"
        exit 1
        ;;
esac

mapfile -t TASKS < <(
    python "$SCRIPT_DIR/get_biomlbench_tasks.py" \
        --config "$COMPETITORS_DIR/config.yaml" \
        --mode "$MODE"
)

if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "[prepare_biomlbench_tasks] No tasks found for mode '$MODE' in $COMPETITORS_DIR/config.yaml"
    exit 1
fi

for task in "${TASKS[@]}"; do
    echo "[prepare_biomlbench_tasks] Preparing $task"
    if [[ "$task" == polarishub/* ]]; then
        python "$SCRIPT_DIR/fetch_polaris_leaderboard.py" \
            --task-id "$task" \
            --competitors-dir "$COMPETITORS_DIR"
    fi
    python "$SCRIPT_DIR/prepare_biomlbench_task.py" \
        --task-id "$task" \
        --data-dir "$DATA_DIR"
done
