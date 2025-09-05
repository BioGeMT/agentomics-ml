set -euo pipefail

docker build -t agentomics_prepare_img -f Dockerfile.prepare .
docker run \
    --rm \
    -it \
    --name agentomics_prepare_cont \
    -v $(pwd):/repository \
    agentomics_prepare_img

docker build -t agentomics_img .
docker volume create agentomics_volume

docker run \
    -it \
    --rm \
    --name agentomics_cont \
    -v $(pwd):/repository:ro \
    -v agentomics_volume:/workspace/runs \
    agentomics_img
