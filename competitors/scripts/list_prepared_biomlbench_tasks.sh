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
        echo "[list_prepared_biomlbench_tasks] Invalid mode '$MODE'. Use: all | proteingym | polaris"
        exit 1
        ;;
esac

mapfile -t TASKS < <(
    python "$SCRIPT_DIR/get_biomlbench_tasks.py" \
        --config "$COMPETITORS_DIR/config.yaml" \
        --mode "$MODE"
)

if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "[list_prepared_biomlbench_tasks] No tasks found for mode '$MODE' in $COMPETITORS_DIR/config.yaml"
    exit 1
fi

is_task_prepared() {
    local task="$1"
    local prepared_root="$DATA_DIR/$task/prepared"
    local public_dir="$prepared_root/public"
    local private_dir="$prepared_root/private"

    [[ -d "$public_dir" && -d "$private_dir" ]] || return 1
    [[ -n "$(find "$public_dir" -mindepth 1 -print -quit 2>/dev/null)" ]] || return 1
    [[ -n "$(find "$private_dir" -mindepth 1 -print -quit 2>/dev/null)" ]] || return 1
}

for task in "${TASKS[@]}"; do
    if is_task_prepared "$task"; then
        echo "PREPARED  $task"
    else
        echo "MISSING   $task"
    fi
done
