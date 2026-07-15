#!/usr/bin/env bash
# Release Agentomics: build + push the multi-arch Docker image from the git tag
# v<version>, then build + upload the pip package. Both come from the same commit,
# so tag the release commit, push the tag, and run this from that clean checkout.
#
# Prerequisites:
#   - docker login with push access to the agentomics image repo
#   - a PyPI token for twine (~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD)
#
# Usage:
#   ./scripts/release.sh

set -euo pipefail
cd "$(dirname "$0")/.."

command -v docker >/dev/null || { echo "Error: docker is required" >&2; exit 1; }

VERSION="$(PYTHONPATH=src python3 -c 'import agentomics; print(agentomics.__version__)')"
TAG="v${VERSION}"
# Must match docker_utils.DEFAULT_IMAGE so the release publishes the image the
# installed package pulls.
IMAGE="biogemt/agentomics:${VERSION}"
REPO_URL="${AGENTOMICS_REPO_URL:-https://github.com/BioGeMT/Agentomics-ML.git}"

echo "Releasing Agentomics ${TAG}"

# Forgot to bump __version__? Refuse to re-release a version already on PyPI.
if curl -sfI "https://pypi.org/pypi/agentomics/${VERSION}/json" >/dev/null 2>&1; then
    echo "Error: agentomics ${VERSION} is already on PyPI; bump __version__." >&2
    exit 1
fi

# Release the tagged commit exactly, so the image (built from the tag) and the
# pip package (built from the tree) are the same code.
[ -z "$(git status --porcelain)" ] \
    || { echo "Error: working tree is not clean." >&2; exit 1; }
git ls-remote --exit-code origin "refs/tags/${TAG}" >/dev/null 2>&1 \
    || { echo "Error: tag ${TAG} is not on origin. Create and push it: git tag ${TAG} && git push origin ${TAG}" >&2; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse "${TAG}^{commit}")" ] \
    || { echo "Error: HEAD is not at ${TAG}; check out the tagged commit first." >&2; exit 1; }

# Release only integrated code: the tag must sit on the release branch (main).
RELEASE_BRANCH="${AGENTOMICS_RELEASE_BRANCH:-main}"
git fetch --quiet origin "$RELEASE_BRANCH"
git merge-base --is-ancestor "${TAG}^{commit}" FETCH_HEAD \
    || { echo "Error: ${TAG} is not on origin/${RELEASE_BRANCH}; merge the release commit to ${RELEASE_BRANCH} first." >&2; exit 1; }

# Forward any proxy settings into the buildx builder and the image build.
DRIVER_OPTS=()
BUILD_ARGS=(--build-arg "REPOSITORY_SOURCE=${REPO_URL}#${TAG}")
for proxy_var in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
    proxy_value="${!proxy_var:-}"
    [ -n "$proxy_value" ] || continue
    DRIVER_OPTS+=(--driver-opt "env.${proxy_var}=${proxy_value}")
    BUILD_ARGS+=(--build-arg "${proxy_var}=${proxy_value}")
done

# Recreate the buildx builder so current proxy settings apply.
docker buildx rm multiplatform 2>/dev/null || true
docker buildx create --name multiplatform --driver docker-container \
    ${DRIVER_OPTS[@]+"${DRIVER_OPTS[@]}"} --use
docker buildx inspect --bootstrap

echo "Building and pushing ${IMAGE} from ${TAG}"
docker buildx build --platform linux/amd64,linux/arm64 \
    -t "$IMAGE" "${BUILD_ARGS[@]}" --push .

# Build and publish the pip package.
python3 -m pip install --upgrade build twine
rm -rf dist
python3 -m build
python3 -m twine check dist/*
python3 -m twine upload dist/*

echo "Released Agentomics ${TAG}: image ${IMAGE} and pip package."
