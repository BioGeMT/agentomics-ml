# Environment Variables

Configure Agentomics-ML using environment variables or a `.env` file.

## Setting Variables

### Option 1: Export in Shell

```bash
export OPENROUTER_API_KEY="your-key-here"
./run.sh
```

### Option 2: .env File

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env with your values
```

Docker mode expects a `.env` file to exist, even if you export variables in your shell.

## LLM Provider Keys

At least one API key is required:

| Variable | Provider | Get Key |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter | [openrouter.ai](https://openrouter.ai/) |
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com/) |
| `ANTHROPIC_API_KEY` | Anthropic | [console.anthropic.com](https://console.anthropic.com/) |

The Codex provider uses `codex login` and reads `~/.codex/auth.json`. Ollama does not use an API key, but set `OLLAMA_BASE_URL` so the provider is selectable.

### Provisioning Key (Optional)

If you use `--use-provisioning-key`, set:

| Variable | Description |
|----------|-------------|
| `PROVISIONING_OPENROUTER_API_KEY` | OpenRouter provisioning key used to mint temporary keys |

### Example

```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

## Weights & Biases (W&B)

Enable experiment tracking and logging:

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | Your W&B API key |
| `WANDB_PROJECT_NAME` | Project name for runs |
| `WANDB_ENTITY` | Team or username |

### Example

```bash
# .env file
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT_NAME=agentomics-experiments
WANDB_ENTITY=my-team
```

Get your W&B API key at [wandb.ai/authorize](https://wandb.ai/authorize).

## Proxy Settings

For corporate networks or firewalls:

| Variable | Description |
|----------|-------------|
| `HTTP_PROXY` | HTTP proxy URL |
| `HTTPS_PROXY` | HTTPS proxy URL |
| `NO_PROXY` | Hosts to bypass proxy |

### Example

```bash
# .env file
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080
NO_PROXY=localhost,127.0.0.1
```

See [Proxy Settings](../developer/proxy-settings.md) for Docker proxy configuration.

## Ollama Configuration

Ollama uses the base URL defined in `src/utils/providers/configured_providers.yaml`.
Edit the `Ollama` provider entry to point to your Ollama server.

## GPU Configuration

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Specify which GPUs to use |

### Example

```bash
# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
```

## Complete .env Example

```bash
# LLM Provider (choose one or more)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx

# Weights & Biases (optional)
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT_NAME=agentomics
WANDB_ENTITY=my-team

# Proxy (optional)
# HTTP_PROXY=http://proxy.company.com:8080
# HTTPS_PROXY=http://proxy.company.com:8080
# NO_PROXY=localhost,127.0.0.1

# Ollama (optional)
# Configure in src/utils/providers/configured_providers.yaml
# OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Security Notes

!!! warning "Keep Keys Secret"
    - Never commit `.env` to version control
    - `.env` is already in `.gitignore`
    - Use environment-specific keys for production

## Variable Precedence

1. Command-line arguments (highest)
2. Environment variables
3. `.env` file
4. Default values (lowest)
