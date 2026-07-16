# GPU Settings

Configure NVIDIA GPU support for accelerated training.

## Overview

GPU acceleration significantly speeds up:
- Neural network training
- Large dataset processing

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- NVIDIA Container Toolkit

## Checking GPU Availability

### Host System

```bash
nvidia-smi
```

Should show your GPU(s) with driver version and CUDA version.

### In Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Installing NVIDIA Container Toolkit

Follow the official guide:
[NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Quick Install (Ubuntu/Debian)

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verification

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Running with GPU

### Default (GPU Enabled)

```bash
agentomics-run
```

GPU is detected and used automatically.

### CPU Only

```bash
agentomics-run --cpu-only
```

Disables GPU even if available.

### Specific GPUs

The launcher forwards `CUDA_VISIBLE_DEVICES` into the container:

```bash
# Use only GPU 0
CUDA_VISIBLE_DEVICES=0 agentomics-run

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 agentomics-run

# Use no GPU (equivalent to --cpu-only)
CUDA_VISIBLE_DEVICES="" agentomics-run
```

## GPU Memory

### Monitoring

During training, monitor GPU memory:

```bash
watch -n 1 nvidia-smi
```

### Out of Memory

If you encounter OOM errors:

1. Use `--cpu-only` to switch to CPU
2. Reduce batch size in generated training scripts
3. Use a smaller model architecture
4. Use gradient accumulation

## Multi-GPU

Agentomics-ML supports multi-GPU training:

- Agent-generated scripts may use DataParallel or DistributedDataParallel
- GPUs exposed to the container are available to generated training scripts

```bash
CUDA_VISIBLE_DEVICES=0,1 agentomics-run
```

`agentomics-run` requests all GPUs from Docker (`--gpus all`) unless you pass
`--cpu-only`. To restrict the run to specific GPUs, set `CUDA_VISIBLE_DEVICES`
as shown above; the launcher forwards it into the container.

## Troubleshooting

### "nvidia-smi not found"

NVIDIA drivers not installed. Install from:
[NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

### "docker: Error response from daemon: could not select device driver"

NVIDIA Container Toolkit not installed or configured. Follow installation steps above.

### GPU not detected in container

1. Verify host GPU works: `nvidia-smi`
2. Verify Docker can see GPU: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
3. Check Docker runtime: `docker info | grep -i runtime`

### CUDA version mismatch

The container uses a specific CUDA version. If your driver is older:

1. Update NVIDIA drivers
2. Or use `--cpu-only` mode

### Performance is slow

- Check GPU utilization with `nvidia-smi`
- Ensure you're not CPU-bound (data loading)
- Verify training is actually using GPU (check nvidia-smi during training)

## Cloud GPU Instances

### AWS

Use GPU instances (p3, p4, g4, g5 series) with Deep Learning AMI.

### Google Cloud

Use GPU instances with Deep Learning VM.

### Azure

Use NC/ND series VMs with GPU support.

## Related

- [Installation](../getting-started/installation.md)
