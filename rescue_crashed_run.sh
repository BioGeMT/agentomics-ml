#!/usr/bin/env bash
# Rescues a crashed Docker run by extracting its workspace volume to outputs/.
# The Docker volume persists after crashes and power loss; use this script to
# recover what was completed before the failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source scripts/bash_helpers.sh

show_help() {
    cat <<'EOF'
Usage: ./rescue_crashed_run.sh --agent-id <id> [OPTIONS]

Extracts the workspace of a crashed run from its Docker volume and saves the
results to outputs/<agent_id>/, regenerating markdown reports from whatever
iterations completed before the crash.

Required (unless --list is given):
  --agent-id <id>     Agent ID of the crashed run.
                      Use --list to see available volumes.

Options:
  --list              List agent IDs with surviving Docker volumes, then exit.
  --skip-reports      Copy raw files only; skip report generation.
  -h, --help          Show this help and exit.

The rescued outputs land in outputs/<agent_id>/. From there you can inspect
completed iterations or fork from the last checkpoint:
  ./run.sh --fork-from-run outputs/<agent_id> --iterations <N>
EOF
}

AGENT_ID=""
LIST_MODE=false
SKIP_REPORTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent-id)
            require_opt_value "$1" "${2:-}"
            AGENT_ID="$2"
            shift 2
            ;;
        --list)
            LIST_MODE=true
            shift
            ;;
        --skip-reports)
            SKIP_REPORTS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            die "Unknown argument: $1. Run with --help for usage."
            ;;
    esac
done

# ── List mode ─────────────────────────────────────────────────────────────────
if [ "$LIST_MODE" = true ]; then
    echo "Agent IDs with surviving Docker volumes:"
    docker volume ls --filter "name=temp_agentomics_volume_" --format "{{.Name}}" \
        | sed 's/^temp_agentomics_volume_//' \
        | while read -r id; do echo "  $id"; done \
        || echo "  (none found)"
    exit 0
fi

[[ -n "$AGENT_ID" ]] || die "--agent-id is required. Use --list to see available volumes, or --help for usage."

VOLUME_NAME="temp_agentomics_volume_${AGENT_ID}"
OUTPUT_DIR="outputs/${AGENT_ID}"

if ! docker volume ls --format "{{.Name}}" | grep -q "^${VOLUME_NAME}$"; then
    die "Volume '${VOLUME_NAME}' not found. Use --list to see available volumes."
fi

echo "Rescuing run '${AGENT_ID}' from volume '${VOLUME_NAME}'..."

# ── Extract workspace from volume ─────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}/best_iteration_snapshot" \
         "${OUTPUT_DIR}/run" \
         "${OUTPUT_DIR}/reports" \
         "${OUTPUT_DIR}/extras"

echo "Copying run directory..."
docker run --rm \
    -v "${VOLUME_NAME}:/workspace" \
    busybox chmod -R a+rX /workspace/run/ || true

docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${VOLUME_NAME}:/source" \
    -v "$(pwd)/${OUTPUT_DIR}:/dest" \
    busybox cp -r /source/run/. /dest/run/

docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${VOLUME_NAME}:/source" \
    -v "$(pwd)/${OUTPUT_DIR}:/dest" \
    busybox sh -c 'if [ -d /source/reports ]; then cp -r /source/reports/. /dest/reports/; fi'

docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${VOLUME_NAME}:/source" \
    -v "$(pwd)/${OUTPUT_DIR}:/dest" \
    busybox sh -c 'if [ -d /source/extras ]; then cp -r /source/extras/. /dest/extras/; fi'

# ── Best iteration snapshot (only present if at least one iteration completed) ─
HAS_SNAPSHOT=false
if docker run --rm -v "${VOLUME_NAME}:/workspace" busybox test -d /workspace/best_iteration_snapshot 2>/dev/null; then
    echo "Found best iteration snapshot — copying..."
    docker run --rm -v "${VOLUME_NAME}:/workspace" busybox chmod -R a+rX /workspace/best_iteration_snapshot/ || true
    docker run --rm \
        -u "$(id -u):$(id -g)" \
        -v "${VOLUME_NAME}:/source" \
        -v "$(pwd)/${OUTPUT_DIR}:/dest" \
        busybox cp -r /source/best_iteration_snapshot/. /dest/best_iteration_snapshot/
    HAS_SNAPSHOT=true
fi

# ── Report generation ─────────────────────────────────────────────────────────
if [ "$SKIP_REPORTS" = false ]; then
    echo "Generating iteration reports..."
    PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python -m runtime.iteration_reports \
        --agent-dir "${OUTPUT_DIR}"

    if [ "$HAS_SNAPSHOT" = true ]; then
        AGENTOMICS_IMAGE="agentomics_img"
        if docker image inspect "$AGENTOMICS_IMAGE" > /dev/null 2>&1; then
            echo "Generating PDF reports..."
            docker run --rm \
                -u "$(id -u):$(id -g)" \
                -e MPLCONFIGDIR="/tmp/mplconfig" \
                -e PYTHONPATH=/repository/src \
                -v "$(pwd)":/repository \
                -v "$(pwd)/${OUTPUT_DIR}":/agent_out \
                --entrypoint /opt/conda/envs/agentomics-env/bin/python \
                "$AGENTOMICS_IMAGE" /repository/src/runtime/generate_final_reports.py \
                    --agent-dir /agent_out \
                    --prepared-datasets /repository/prepared_datasets \
                    --prepared-tests /repository/prepared_test_sets
        else
            warn "Docker image '${AGENTOMICS_IMAGE}' not found — skipping PDF reports."
            warn "Build it first with: ./run.sh --build-images --list-datasets"
        fi
    fi
fi

write_outputs_readme "${AGENT_ID}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Rescue complete. Files saved to: ${OUTPUT_DIR}/"
if [ "$HAS_SNAPSHOT" = false ]; then
    warn "No best iteration snapshot found — the run crashed before completing its first iteration."
fi
echo ""
echo "To fork from the last checkpoint and continue:"
echo "  ./run.sh --fork-from-run ${OUTPUT_DIR} --iterations <N>"
