# LLM Providers

Agentomics-ML supports multiple LLM providers out of the box.

## Supported Providers

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| [OpenRouter](https://openrouter.ai/) | `OPENROUTER_API_KEY` | 100+ models |
| [OpenAI](https://openai.com/) | `OPENAI_API_KEY` | gpt-5.2-codex |
| [Anthropic](https://anthropic.com/) | `ANTHROPIC_API_KEY` | Claude 3.5, Claude 3, etc. |
| [Ollama](https://ollama.ai/) | Local setup | Local models |

## OpenRouter

**Recommended for beginners** - Access to 100+ models with one API key.

### Setup

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxx"
./run.sh
```

### Available Models

```bash
./run.sh --list-models
```

Default model:

- `openai/gpt-5.2-codex` - Recommended for best results

### Provisioning Key

For temporary access without your own key:

```bash
./run.sh --use-provisioning-key
```

---

## OpenAI

Direct access to OpenAI models.

### Setup

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxx"
./run.sh
```

### Available Models

- `gpt-5.2-codex` - Default, recommended

---

## Anthropic

Direct access to Claude models.

### Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxx"
./run.sh
```

### Available Models

Available via OpenRouter with `anthropic/` prefix.

---

## Ollama (Local Models)

Run models locally for privacy or offline use.

### Requirements

1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull <model-name>`

### Docker Mode (Recommended)

Configure Ollama to accept Docker connections:

```bash
sudo systemctl edit ollama.service
```

Add:
```ini
[Service]
Environment="OLLAMA_HOST=172.17.0.1:11434"
```

Restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama.service
```

Run:
```bash
./run.sh --ollama
```

### Local Mode

```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
./run.sh --local
```

### Popular Models

Run `ollama list` to see available models.

---

## Custom Providers

Add custom providers in `src/utils/providers/configured_providers.yaml`:

```yaml
providers:
  - name: "MyProvider"
    base_url: "https://api.myprovider.com/v1"
    apikey: "${MY_PROVIDER_API_KEY}"
```

Then set the API key:
```bash
export MY_PROVIDER_API_KEY="your-key"
```

For custom providers, use `--model` explicitly:
```bash
./run.sh --model my-custom-model
```

---

## Provider Selection

When multiple providers are configured, they're all available. Use `--list-models` to see all options:

```bash
./run.sh --list-models
```

The interactive mode groups models by provider for easy selection.

---

## Model Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Default | `openai/gpt-5.2-codex` |
| Privacy/Offline | Ollama local models |

---

## Troubleshooting

### "API key not found"

Ensure your key is set:
```bash
echo $OPENROUTER_API_KEY  # Should show your key
```

### "Model not available"

Check available models:
```bash
./run.sh --list-models
```

### "Rate limit exceeded"

- Wait and retry
- Use a different provider
- Check your API plan limits

### Ollama connection refused

Ensure Ollama is running:
```bash
ollama list  # Should show pulled models
```

For Docker mode, verify the OLLAMA_HOST setting.
