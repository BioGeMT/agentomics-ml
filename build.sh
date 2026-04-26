#!/bin/bash

# Multi-architecture (amd64, arm64) Docker build and push script for Agentomics
# Builds all foundation model variants of the main image + the prepare image.
#
# Usage:
#   ./build.sh <username>              # Builds and pushes with 'latest' tag
#   ./build.sh <username> v1.0         # Builds and pushes with 'v1.0' tag

set -e
AGENTOMICS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$AGENTOMICS_DIR/bash_helpers.sh"

DRIVER_OPTS=()
if [ -n "${HTTP_PROXY:-}" ]; then
    DRIVER_OPTS+=(--driver-opt "env.HTTP_PROXY=$HTTP_PROXY")
fi
if [ -n "${HTTPS_PROXY:-}" ]; then
    DRIVER_OPTS+=(--driver-opt "env.HTTPS_PROXY=$HTTPS_PROXY")
fi
if [ -n "${http_proxy:-}" ]; then
    DRIVER_OPTS+=(--driver-opt "env.http_proxy=$http_proxy")
fi
if [ -n "${https_proxy:-}" ]; then
    DRIVER_OPTS+=(--driver-opt "env.https_proxy=$https_proxy")
fi

PROXY_BUILD_ARGS=()
if [ -n "${HTTP_PROXY:-}" ]; then
    PROXY_BUILD_ARGS+=(--build-arg "HTTP_PROXY=$HTTP_PROXY")
fi
if [ -n "${HTTPS_PROXY:-}" ]; then
    PROXY_BUILD_ARGS+=(--build-arg "HTTPS_PROXY=$HTTPS_PROXY")
fi
if [ -n "${http_proxy:-}" ]; then
    PROXY_BUILD_ARGS+=(--build-arg "http_proxy=$http_proxy")
fi
if [ -n "${https_proxy:-}" ]; then
    PROXY_BUILD_ARGS+=(--build-arg "https_proxy=$https_proxy")
fi

USERNAME=${1:-}
VERSION=${2:-latest}

if [ -z "$USERNAME" ]; then
    echo "Usage: $0 <username> [version]"
    echo "Example: $0 myusername v1.0"
    exit 1
fi

FM_VARIANTS=("NONE" "DNA" "RNA" "MOLECULE" "PROTEIN" "ALL")

declare -A FM_BUILD_ARG_MAP
FM_BUILD_ARG_MAP["NONE"]=""
FM_BUILD_ARG_MAP["DNA"]="dna"
FM_BUILD_ARG_MAP["RNA"]="rna"
FM_BUILD_ARG_MAP["MOLECULE"]="molecule"
FM_BUILD_ARG_MAP["PROTEIN"]="protein"
FM_BUILD_ARG_MAP["ALL"]="all"

PREPARE_IMAGE="${USERNAME}/agentomics-prepare:${VERSION}"

need_cmd docker

docker buildx rm multiplatform 2>/dev/null || true
docker buildx create \
    --name multiplatform \
    --driver docker-container \
    ${DRIVER_OPTS[@]+"${DRIVER_OPTS[@]}"} \
    --use
docker buildx inspect --bootstrap

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -f Dockerfile.prepare \
    -t "$PREPARE_IMAGE" \
    ${PROXY_BUILD_ARGS[@]+"${PROXY_BUILD_ARGS[@]}"} \
    --push .

for VARIANT in "${FM_VARIANTS[@]}"; do
    IMAGE_TAG="${USERNAME}/agentomics:FM-${VARIANT}-${VERSION}"
    BUILD_ARG_VALUE="${FM_BUILD_ARG_MAP[$VARIANT]}"

    echo ""
    echo "============================================"
    echo "Building: ${IMAGE_TAG}"
    if [ -n "$BUILD_ARG_VALUE" ]; then
        echo "Foundation model type: ${BUILD_ARG_VALUE}"
    else
        echo "Foundation model type: none"
    fi
    echo "============================================"

    BUILD_CMD=(
        docker buildx build
        --platform linux/amd64,linux/arm64
        -t "$IMAGE_TAG"
        ${PROXY_BUILD_ARGS[@]+"${PROXY_BUILD_ARGS[@]}"}
    )

    if [ -n "$BUILD_ARG_VALUE" ]; then
        BUILD_CMD+=(--build-arg "FOUNDATION_MODEL_TYPE=$BUILD_ARG_VALUE")
    fi

    BUILD_CMD+=(--push .)

    if "${BUILD_CMD[@]}"; then
        echo "Done: ${IMAGE_TAG}"
    else
        echo "FAILED: ${IMAGE_TAG}"
        exit 1
    fi
done
