#!/usr/bin/env bash

# Common shell utilities for agentomics bash scripts

# Ensure running under bash even if invoked via sh/zsh
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

# Print error message in red and exit with status 1
die() {
    echo -e "${RED}Error: $*${NOCOLOR}" >&2
    exit 1
}

# Check if a command exists, die with helpful message if not
need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# Validate that an option has a non-empty value
require_opt_value() {
    local opt="$1"
    local val="${2:-}"
    [[ -n "$val" && "$val" != --* ]] || die "Missing value for $opt"
}

# Check if stdin and stdout are connected to a TTY (interactive terminal)
has_tty() {
    [[ -t 0 && -t 1 ]]
}

# Run a command and suppress its output. Still prints errors if the command fails
printless_command(){
    "$@" > /dev/null 
}

# Setup a trap to clean up the Docker volume on script exit
cleanup_volume_on_finish() {
    trap 'docker volume rm temp_agentomics_volume_${AGENT_ID} >/dev/null 2>&1 || true' EXIT
}

# Print info message in green
info() {
    echo -e "${GREEN}$*${NOCOLOR}"
}

# Print warning message in yellow
warn() {
    echo -e "${YELLOW}Warning: $*${NOCOLOR}" >&2
}

docker_has_gpu() {
    # Check if nvidia-container-toolkit is installed and nvidia-smi works
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        # Also verify docker can use GPUs by checking for nvidia runtime
        if docker info 2>/dev/null | grep -q nvidia; then
            return 0
        fi
    fi
    return 1
}

load_string_field_from_run_config() {
    local source_workspace="$1"
    local field_name="$2"
    local config_path="$source_workspace/run/shared/config.json"
    local field_line=""
    local field_value=""

    [[ -f "$config_path" ]] || die "Source config not found at $config_path"

    field_line="$(grep -m1 -E "^[[:space:]]*\"${field_name}\"[[:space:]]*:" "$config_path" || true)"
    [[ -n "$field_line" ]] || die "${field_name} is missing from $config_path"

    if [[ "$field_line" =~ :[[:space:]]*null([[:space:]]*,?[[:space:]]*)$ ]]; then
        return 0
    fi

    field_value="$(printf '%s\n' "$field_line" | sed -nE "s/^[[:space:]]*\"${field_name}\"[[:space:]]*:[[:space:]]*\"([^\"]*)\"[[:space:]]*,?[[:space:]]*$/\\1/p")"
    [[ -n "$field_value" ]] || die "Could not parse ${field_name} from $config_path"
    printf '%s\n' "$field_value"
}

resolve_effective_string_field() {
    local explicit_value="${1:-}"
    local fork_from_run="${2:-}"
    local field_name="$3"
    local prefer_fork_value="${4:-false}"

    if [[ "$prefer_fork_value" = true && -n "$fork_from_run" ]]; then
        load_string_field_from_run_config "$fork_from_run" "$field_name"
        return 0
    fi
    if [[ -n "$explicit_value" ]]; then
        printf '%s\n' "$explicit_value"
        return 0
    fi
    if [[ -n "$fork_from_run" ]]; then
        load_string_field_from_run_config "$fork_from_run" "$field_name"
        return 0
    fi
}

build_setup_fork_args() {
    local source_workspace="$1"
    local target_workspace="$2"
    local agent_id="$3"
    local fork_from_step="${4:-}"
    local fork_from_iteration="${5:-}"

    SETUP_FORK_ARGS=(
        --source-workspace "$source_workspace"
        --target-workspace "$target_workspace"
        --agent-id "$agent_id"
    )
    [ -n "$fork_from_step" ] && SETUP_FORK_ARGS+=(--fork-from-step "$fork_from_step")
    [ -n "$fork_from_iteration" ] && SETUP_FORK_ARGS+=(--fork-from-iteration "$fork_from_iteration")
    return 0
}

write_outputs_readme() {
    local agent_id="$1"
    local best_iter="No best iteration found"
    local metadata_path="outputs/${agent_id}/best_iteration_snapshot/runtime_info/iteration_metadata.json"

    if [[ -f "$metadata_path" ]]; then
        best_iter="$(sed -nE 's/.*"iteration"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' "$metadata_path" | head -n 1)"
        [[ -z "$best_iter" ]] && best_iter="No best iteration found"
    fi

    cat > "outputs/${agent_id}/README.md" << EOF

This directory contains the full results of one run (**AGENT_ID: ${agent_id}**).
It includes: (1) the **best model chosen across iterations**, (2) per-iteration run artifacts,
(3) reports, and (4) logs/extras.

---

**Best iteration selected:** **${best_iter}**

**Note:** Iterations are **0-indexed** (first iteration is \`0\`).

What to use:
- **Best model + code:** \`best_iteration_snapshot/\` (train script: \`model_training/train.py\`, artifacts: \`model_training/training_artifacts/\`)
- **Best report:** \`reports/markdown/run_report_iter_${best_iter}.md\`

---
\`\`\`
outputs/${agent_id}/
├── best_iteration_snapshot/                 # Snapshot exported from the selected best iteration
│   ├── model_training/
│   │   ├── train.py                # Training script used to produce the best model
│   │   └── training_artifacts/     # Serialized artifacts (e.g. \`model.joblib\`)
│   │       └── model.joblib
│   │       └── ...
│   ├── model_inference/
│   │   └── inference.py            # Inference script for predictions on new data
│   ├── validation_evaluation/
│   │   ├── output.json             # Contains train/validation metrics
│   │   ├── eval_predictions_train.csv
│   │   └── eval_predictions_validation.csv
│   ├── <other_step_dirs>/          # output.json + step scripts per step
│   ├── runtime_info/
│   │   └── iteration_metadata.json
│   ├── environment.yml
│   ├── eval_predictions_test.csv   # Predictions on held-out test set
│   └── test_metrics.json           # Metrics on held-out test set
│
├── run/                            # Run working directory
│   ├── shared/splits/              # Versioned train/validation split folders
│   ├── iteration_0/                # Archive of iteration 0
│   ├── iteration_1/                # Archive of iteration 1
│   └── ...                         # Additional iterations if present
│
├── reports/
│   ├── markdown/                   # Human-readable markdown reports per iteration
│   │   ├── run_report_iter_0.md
│   │   ├── run_report_iter_1.md
│   │   └── ...
│   └── pdf/                        # PDF reports + plots per iteration (auto-generated)
│       ├── iteration_0.pdf
│       ├── iteration_1.pdf
│       └── ...
│       └── plots/
│           └── iter_<i>_<split>_*.png
│
├── extras/                         # Logs and debugging information
│   └── run_logs/
│
└── README.md                       # This file
\`\`\`
---

## Running inference on new data

\`\`\`bash
./scripts/inference.sh --agent-dir outputs/${agent_id} --input <path_to_input_folder> --output <path_to_output_csv>
\`\`\`

Inference relies on:
- \`best_iteration_snapshot/model_inference/inference.py\`
- \`best_iteration_snapshot/model_training/training_artifacts/\`

EOF
}

ensure_agentomics_docker_image() {
    # Check the existence of the agentomics docker image. Prioritize locally built images, then dockerhub ones
    local dockerhub_username="$1"
    local dockerhub_img="${dockerhub_username}/agentomics:latest"

    if docker image inspect agentomics_img >/dev/null 2>&1; then
        AGENTOMICS_IMAGE="agentomics_img"
        return
    fi
    if docker image inspect "$dockerhub_img" >/dev/null 2>&1; then
        AGENTOMICS_IMAGE="$dockerhub_img"
        return
    fi
    if docker pull "$dockerhub_img" >/dev/null 2>&1; then
        AGENTOMICS_IMAGE="$dockerhub_img"
        return
    fi

    warn "Docker image not found locally or on Dockerhub. Building 'agentomics_img'..."
    [[ -f "Dockerfile" ]] || die "Dockerfile not found in repository root; cannot build 'agentomics_img'."
    docker build -t agentomics_img -f Dockerfile .
    AGENTOMICS_IMAGE="agentomics_img"
}
