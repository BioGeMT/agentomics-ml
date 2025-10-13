# macOS and ollama Issues 

## Issue 1: Docker GPU Detection Failure

**Error:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

- Running `./run.sh --ollama` fails immediately after building Docker images
- The script unconditionally adds `--gpus all` flag when `CPU_ONLY` is false
- The `--gpus` flag is NVIDIA-specific and doesn't exist on macOS
- macOS uses Metal/Apple Silicon, not NVIDIA CUDA
- Docker Desktop on macOS doesn't support the `--gpus` flag at all

---

## Issue 2: README Ollama Instructions are Linux-Only

```bash
systemctl edit ollama.service
systemctl daemon-reload
systemctl restart ollama.service
```

- `systemctl` is a Linux-specific command (systemd)
- macOS doesn't use systemd
- No alternative instructions provided for macOS users
- The README specifies setting `OLLAMA_HOST=172.17.0.1:11434` which is Linux Docker's default bridge IP
- macOS Docker uses different networking (`host.docker.internal`)


---

## Issue 3: Ollama Doesn't Appear in Provider List


- Ollama is installed and running with `ollama serve`
- `./run.sh --ollama --cpu-only` executes successfully
- Provider selection menu shows: OpenAI, Anthropic, OpenRouter
- Ollama is missing from the list

The code in `src/utils/providers/provider.py` checks for `OLLAMA_BASE_URL` environment variable:
```python
if (api_key_env and os.getenv(api_key_env)) or (provider_name.lower() == "ollama" and os.getenv("OLLAMA_BASE_URL")):
    provided_keys[provider_name] = os.getenv(api_key_env, "")
```

Confusion:
- Ollama's base URL is hardcoded in `configured_providers.yaml`: `base_url: "http://host.docker.internal:11434/v1"`
- The hardcoded URL is used for connecting, but `OLLAMA_BASE_URL` env var is required for *detection*
- The README says `OLLAMA_BASE_URL` is only needed for "local mode (unsafe)"
- But it's actually required for both modes - just for provider detection


---

## Issue 4: Misleading README Documentation

**What the README says:**
> If you're running agentomics in docker mode (recommended), additional steps are needed:
> [systemctl instructions for Linux]
>
> If you're running agentomics in local mode (unsafe), `OLLAMA_BASE_URL` environment variable needs to be provided


- Implies `OLLAMA_BASE_URL` is only for local/unsafe mode
- Actually needed for *both* Docker and local modes (for provider detection to work)
- The hardcoded URL in `configured_providers.yaml` is only used *after* selection
- macOS users need completely different setup than Linux users
- No mention that default `ollama serve` listens only on localhost


---

## Issue 5: Missing macOS-Specific Documentation


- How to start Ollama on macOS to accept connections from Docker
- That `export OLLAMA_HOST=0.0.0.0:11434` is needed before `ollama serve`
- That Docker containers on macOS cannot access GPU/Metal (architecture limitation)
- Ollama runs on ARM GPU but agent does not
- How `host.docker.internal` works on macOS vs Linux bridge networking


---

## Additional Context: macOS Docker Architecture


- Docker containers run natively on ARM architecture (Apple Silicon M1/M2/M3/M4)
- Uses ARM-optimized packages from conda-forge
- Cannot access Metal/Apple Silicon GPU (Docker Desktop limitation)
- ML training happens on CPU only inside containers
- Ollama (running outside Docker on the host) *can* use Metal/GPU

**Not a bug, but worth documenting for user expectations.**

---

