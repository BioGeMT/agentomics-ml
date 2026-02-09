#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPETITORS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$COMPETITORS_DIR/data"

PROTEINGYM_TASKS=(
    "proteingym-dms/SPIKE_SARS2_Starr_2020_binding"
    "proteingym-dms/SPA_STAAU_Tsuboyama_2023_1LP1"
    "proteingym-dms/PSAE_PICP2_Tsuboyama_2023_1PSE_indels"
    "proteingym-dms/CBX4_HUMAN_Tsuboyama_2023_2K28"
    "proteingym-dms/Q8EG35_SHEON_Campbell_2022_indels"
    "proteingym-dms/CSN4_MOUSE_Tsuboyama_2023_1UFM_indels"
)

POLARIS_TASKS=(
    "polarishub/polaris-pkis2-egfr-wt-c-1"
    "polarishub/polaris-adme-fang-hclint-1"
    "polarishub/polaris-adme-fang-hppb-1"
    "polarishub/polaris-adme-fang-solu-1"
    "polarishub/tdcommons-cyp2d6-substrate-carbonmangels"
    "polarishub/tdcommons-lipophilicity-astrazeneca"
    "polarishub/tdcommons-herg"
    "polarishub/tdcommons-bbb-martins"
    "polarishub/tdcommons-caco2-wang"
)

MODE="${1:-all}"
case "$MODE" in
    all)
        TASKS=("${PROTEINGYM_TASKS[@]}" "${POLARIS_TASKS[@]}")
        ;;
    proteingym)
        TASKS=("${PROTEINGYM_TASKS[@]}")
        ;;
    polaris)
        TASKS=("${POLARIS_TASKS[@]}")
        ;;
    *)
        echo "[prepare_biomlbench_tasks] Invalid mode '$MODE'. Use: all | proteingym | polaris"
        exit 1
        ;;
esac

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
