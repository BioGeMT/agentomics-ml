#!/bin/bash

# Simple multi-architecture Docker build and push script
# Usage: ./build.sh <username> [version]

set -e

USERNAME=${1:-}
VERSION=${2:-latest}

if [ -z "$USERNAME" ]; then
    echo "Usage: $0 <username> [version]"
    echo "Example: $0 myusername v1.0"
    exit 1
fi

MAIN_IMAGE="$USERNAME/agentomics:$VERSION"
PREPARE_IMAGE="$USERNAME/agentomics-prepare:$VERSION"

echo "Building and pushing multi-architecture images:"
echo "  $MAIN_IMAGE"
echo "  $PREPARE_IMAGE"

# Setup buildx
docker buildx create --name multiplatform --driver docker-container --use 2>/dev/null || docker buildx use multiplatform
docker buildx inspect --bootstrap

# Build and push main image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t "$MAIN_IMAGE" \
  --push .

# Build and push preparation image  
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.prepare \
  -t "$PREPARE_IMAGE" \
  --push .

echo "Complete! Images pushed:"
echo "  $MAIN_IMAGE"
echo "  $PREPARE_IMAGE"
