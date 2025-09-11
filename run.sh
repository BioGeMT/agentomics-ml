set -euo pipefail

AGENTOMICS_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --model $2"
            shift 2
            ;;
        --dataset)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --dataset $2"
            shift 2
            ;;
        --iterations)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --iterations $2"
            shift 2
            ;;
        --val-metric)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --val-metric $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

docker build -t agentomics_prepare_img -f Dockerfile.prepare .
docker run \
    --rm \
    -it \
    --name agentomics_prepare_cont \
    -v $(pwd):/repository \
    agentomics_prepare_img

docker build -t agentomics_img .
docker volume create temp_agentomics_volume

docker run \
    -it \
    --rm \
    --name agentomics_cont \
    -v $(pwd):/repository:ro \
    -v temp_agentomics_volume:/workspace \
    agentomics_img $AGENTOMICS_ARGS

# Copy best-run files and report
docker run --rm -v temp_agentomics_volume:/source -v $(pwd)/outputs:/dest busybox cp -r /source/snapshots/. /dest/best_run_files/

# Copy reports from all iterations
docker run --rm -v temp_agentomics_volume:/source -v $(pwd)/outputs:/dest busybox cp -r /source/reports/. /dest/reports/

docker volume rm temp_agentomics_volume
