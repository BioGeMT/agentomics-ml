#!/bin/bash
# Multi-architecture (amd64, arm64) Docker build and push script for Agentomics
# It assumes already logged in to Docker: docker login -u <username>
#
# Usage:
#   ./scripts/build.sh <username>              # Builds and pushes with 'latest' tag
#   ./scripts/build.sh <username> v1.0         # Builds and pushes with 'v1.0' tag

set -e
AGENTOMICS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$AGENTOMICS_DIR/scripts/bash_helpers.sh"

cd "$AGENTOMICS_DIR"

# Forward proxy settings into the buildx builder container if present
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

# Forward proxy into Dockerfile RUN commands via build args
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

AGENTOMICS_IMAGE="${USERNAME}/agentomics:${VERSION}"

need_cmd docker

# Recreate buildx builder to ensure current proxy settings are applied
docker buildx rm multiplatform 2>/dev/null || true
docker buildx create \
    --name multiplatform \
    --driver docker-container \
    ${DRIVER_OPTS[@]+"${DRIVER_OPTS[@]}"} \
    --use
docker buildx inspect --bootstrap

echo ""
echo "============================================"
echo "Building: ${AGENTOMICS_IMAGE}"
echo "============================================"

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t "$AGENTOMICS_IMAGE" \
    ${PROXY_BUILD_ARGS[@]+"${PROXY_BUILD_ARGS[@]}"} \
    --push .
