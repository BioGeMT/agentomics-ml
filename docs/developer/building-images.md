# Building Docker Images

Build and distribute custom Docker images for Agentomics-ML.

## Quick Build (Multi-Arch + Push)

Use the provided build script to build and push multi-architecture images:

```bash
./build.sh <dockerhub-username> [version]
```

This script always pushes to the registry and requires `docker login` first.

## Build Script Usage

```bash
# Build and push to Docker Hub (latest)
./build.sh myusername

# Build and push a specific version
./build.sh myusername v1.0
```

## Images

Two Docker images are built:

| Image | Purpose | Dockerfile |
|-------|---------|------------|
| `agentomics` | Main agent execution | `Dockerfile` |
| `agentomics-prepare` | Dataset preparation | `Dockerfile.prepare` |

## Manual Build (Local Only)

### Main Image

```bash
docker build -t agentomics_img .
```

### Preparation Image

```bash
docker build -f Dockerfile.prepare -t agentomics_prepare_img .
```

## Multi-Architecture Builds

For production deployments supporting both Intel/AMD and ARM processors.

### Prerequisites

Enable Docker's multi-platform builder:

```bash
# Create and use a multi-platform builder
docker buildx create --name multiplatform --use
docker buildx inspect --bootstrap
```

### Build Both Architectures

**Main agent image:**

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myusername/agentomics:latest \
  --push .
```

**Preparation image:**

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.prepare \
  -t myusername/agentomics-prepare:latest \
  --push .
```

### With Version Tags

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myusername/agentomics:v1.0 \
  -t myusername/agentomics:latest \
  --push .
```

## Architecture-Specific Builds

Build for specific architectures only:

```bash
# ARM64 only (Apple Silicon, ARM servers)
docker buildx build \
  --platform linux/arm64 \
  -t myusername/agentomics:arm64 \
  --push .

# AMD64 only (Intel/AMD processors)
docker buildx build \
  --platform linux/amd64 \
  -t myusername/agentomics:amd64 \
  --push .
```

## Verification

Check that images support multiple architectures:

```bash
docker buildx imagetools inspect myusername/agentomics:latest
```

Expected output shows both:
- `linux/amd64`
- `linux/arm64`

## Build with Proxy

If behind a corporate proxy:

```bash
docker build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -t agentomics_img .
```
If you need proxy settings for multi-arch builds, configure the Docker daemon
proxy or use manual `docker buildx` commands with `--build-arg`.

## Image Contents

### Main Image (Dockerfile)

The multi-stage build includes:

1. **Stage 1:** Create conda environment (`agentomics-env`)
2. **Stage 2:** Pre-download foundation models
3. **Stage 3:** Pack environment with conda-pack
4. **Final:** Runtime image with entry point

Key contents:
- Python 3.12 with ML libraries
- PyTorch, scikit-learn, transformers
- pydantic-ai for agent framework
- Weave and W&B for logging

### Preparation Image (Dockerfile.prepare)

Lightweight image for dataset preparation:
- Python 3.10
- pandas, rich
- Minimal dependencies

## Customization

### Adding Dependencies

Edit `environment.yaml` and rebuild:

```yaml
dependencies:
  - your-new-package==1.0.0
```

### Modifying Entry Point

The entry point is `python -m agentomics.run_agent_interactive`.

To change, edit the `ENTRYPOINT` in the Dockerfile.

## Storage Requirements

| Image | Size (approx) |
|-------|---------------|
| agentomics | 8-12 GB |
| agentomics-prepare | 1-2 GB |

Includes conda environments and pre-downloaded models.

## Pushing to Registry

### Docker Hub

```bash
docker login
docker push myusername/agentomics:latest
```

### Other Registries

```bash
# AWS ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/agentomics:latest

# Google Container Registry
docker push gcr.io/my-project/agentomics:latest
```

## Troubleshooting

### Build fails with memory error

Increase Docker memory limit or use `--memory` flag:

```bash
docker build --memory=8g -t agentomics .
```

### Build takes too long

Use BuildKit for faster builds:

```bash
DOCKER_BUILDKIT=1 docker build -t agentomics .
```

### Platform mismatch

Ensure you're building for the correct platform:

```bash
docker build --platform linux/amd64 -t agentomics .
```
