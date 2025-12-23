#!/usr/bin/env bash
set -e

# Tags to filter experiments by
TAGS=("ismb2026_v1" "ismb2026_v2" "ismb2026_v2_4hsplit" "ismb2026_v3" "ismb2026_v4")

# Directories to search
SEARCH_DIRS=(
    "/SCRATCH/biomlbench/runs"
    "/SCRATCH/agentomics-ml/outputs"
)

echo "Searching for experiments with tags: ${TAGS[@]}"
echo "----------------------------------------"

# Function to check if config has any of the specified tags
has_matching_tag() {
    local config_file="$1"
    if [[ ! -f "$config_file" ]]; then
        return 1
    fi

    local config_tags=$(jq -r '.tags[]' "$config_file" 2>/dev/null)
    for tag in "${TAGS[@]}"; do
        if echo "$config_tags" | grep -q "^${tag}$"; then
            return 0
        fi
    done
    return 1
}

# Find all experiments with matching tags
MATCHING_EXPERIMENTS=()
CONFIG_FILES=()
declare -A EXPERIMENT_DATASETS
for search_dir in "${SEARCH_DIRS[@]}"; do
    if [[ ! -d "$search_dir" ]]; then
        echo "Warning: Directory $search_dir not found, skipping..."
        continue
    fi

    echo "Searching in $search_dir..."

    # Find all potential agent folders (both biomlbench at depth 3 and agentomics at depth 1)
    while IFS= read -r agent_folder; do
        # Skip if this is not a valid agent folder
        [[ ! -d "$agent_folder" ]] && continue

        agent_id=$(basename "$agent_folder")

        # Check for config in extras first (priority), then fall back to best_run_files
        if [[ -f "$agent_folder/extras/config.json" ]]; then
            config_file="$agent_folder/extras/config.json"
        elif [[ -f "$agent_folder/best_run_files/config.json" ]]; then
            config_file="$agent_folder/best_run_files/config.json"
        elif [[ -d "$agent_folder/code/best_run_files" ]]; then
            # biomlbench structure with code folder
            config_file=$(find "$agent_folder/code/best_run_files" -maxdepth 2 -name "config.json" -type f 2>/dev/null | head -1)
            [[ -z "$config_file" ]] && continue
        elif [[ -d "$agent_folder/best_run_files" ]]; then
            # agentomics structure without code folder
            config_file=$(find "$agent_folder/best_run_files" -maxdepth 2 -name "config.json" -type f 2>/dev/null | head -1)
            [[ -z "$config_file" ]] && continue
        else
            continue
        fi

        if has_matching_tag "$config_file"; then
            tags=$(jq -r '.tags | join(", ")' "$config_file" 2>/dev/null)
            config_agent_id=$(jq -r '.agent_id' "$config_file" 2>/dev/null)
            dataset=$(jq -r '.dataset' "$config_file" 2>/dev/null)

            # Create exp_folder path with agent_id from config at the end
            parent_folder=$(dirname "$agent_folder")
            exp_folder="$parent_folder/$config_agent_id"
            MATCHING_EXPERIMENTS+=("$exp_folder")
            CONFIG_FILES+=("$config_file")
            EXPERIMENT_DATASETS["$exp_folder"]="$dataset"

            echo "  Found: $exp_folder"
            echo "    Agent ID: $config_agent_id, Dataset: $dataset, Tags: [$tags]"
        fi
    done < <(find "$search_dir" -maxdepth 3 -type d -path "*/best_run_files" -prune -o -path "*/run_files" -prune -o -path "*/extras" -prune -o -path "*/reports" -prune -o -type d -print 2>/dev/null | grep -v "^\." | while read dir; do
        # Filter to only directories that have code or best_run_files subdirectories
        [[ -d "$dir/code/best_run_files" || -d "$dir/best_run_files" ]] && echo "$dir"
    done)
done

echo "----------------------------------------"
echo "Found ${#MATCHING_EXPERIMENTS[@]} matching experiments"
echo "----------------------------------------"

# Run compute_stealth_test.sh for each matching experiment
FAILED_EXPERIMENTS=()
for idx in "${!MATCHING_EXPERIMENTS[@]}"; do
    exp_folder="${MATCHING_EXPERIMENTS[$idx]}"
    config_file="${CONFIG_FILES[$idx]}"
    
    echo ""
    echo "========================================"
    echo "Processing experiment: $exp_folder"
    echo "========================================"

    PROTEINGYM_DATASETS=(
        "SPIKE_SARS2_Starr_2020_binding"
        "SPA_STAAU_Tsuboyama_2023_1LP1"
        "PSAE_PICP2_Tsuboyama_2023_1PSE_indels"
        "CBX4_HUMAN_Tsuboyama_2023_2K28"
        "Q8EG35_SHEON_Campbell_2022_indels"
        "CSN4_MOUSE_Tsuboyama_2023_1UFM_indels"
    )

    dataset="${EXPERIMENT_DATASETS[$exp_folder]}"
    skip_experiment=false
    for proteingym_dataset in "${PROTEINGYM_DATASETS[@]}"; do
        if [[ "$dataset" == *"$proteingym_dataset"* ]]; then
            echo "Skipping proteingym dataset: $dataset"
            skip_experiment=true
            break
        fi
    done
    [[ "$skip_experiment" == true ]] && continue

    echo "Using dataset: $dataset"
    cp "$config_file" "$exp_folder/extras/config.json"

    if ./compute_stealth_test.sh --exp-folder "$exp_folder" --agentomics-dir "SCRATCH/agentomics-ml"; then
        echo "✓ Stealth test completed successfully for $exp_folder"
    else
        echo "✗ Stealth test failed for $exp_folder"
        FAILED_EXPERIMENTS+=("$exp_folder")
    fi
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Total experiments processed: ${#MATCHING_EXPERIMENTS[@]}"
echo "Successful: $((${#MATCHING_EXPERIMENTS[@]} - ${#FAILED_EXPERIMENTS[@]}))"
echo "Failed: ${#FAILED_EXPERIMENTS[@]}"

if [[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed experiments:"
    for failed_exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $failed_exp"
    done
    exit 1
fi

echo ""
echo "All stealth tests completed successfully!"