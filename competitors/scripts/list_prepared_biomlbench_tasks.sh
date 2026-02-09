#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPETITORS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$COMPETITORS_DIR/data"

TASKS=(
    "proteingym-dms/SPIKE_SARS2_Starr_2020_binding"
    "proteingym-dms/SPA_STAAU_Tsuboyama_2023_1LP1"
    "proteingym-dms/PSAE_PICP2_Tsuboyama_2023_1PSE_indels"
    "proteingym-dms/CBX4_HUMAN_Tsuboyama_2023_2K28"
    "proteingym-dms/Q8EG35_SHEON_Campbell_2022_indels"
    "proteingym-dms/CSN4_MOUSE_Tsuboyama_2023_1UFM_indels"
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
