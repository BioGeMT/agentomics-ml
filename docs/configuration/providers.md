# LLM Providers

Agentomics-ML supports multiple LLM providers out of the box.

## Supported Providers

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| [OpenRouter](https://openrouter.ai/) | `OPENROUTER_API_KEY` | 100+ models |
| [OpenAI](https://openai.com/) | `OPENAI_API_KEY` | Use `--list-models` to see available models |
| [Anthropic](https://anthropic.com/) | `ANTHROPIC_API_KEY` | Claude models available to your account |
| OpenAI Codex | `codex login` | Uses your local Codex/ChatGPT login |
| [Ollama](https://ollama.ai/) | Local setup | Local models |

## OpenRouter

**Recommended for beginners** - Access to 100+ models with one API key.

### Setup

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxx"
agentomics-run
```

### Available Models

```bash
agentomics-run --list-models
```

Model availability depends on your provider and API plan. Use `agentomics-run --list-models`
to see what is available.

### Provisioning Key

For temporary access without your own key:

```bash
agentomics-run --use-provisioning-key
```

This requires `PROVISIONING_OPENROUTER_API_KEY` in your `.env`.

---

## OpenAI

Direct access to OpenAI models.

### Setup

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxx"
agentomics-run
```

### Available Models

Use `agentomics-run --list-models` to see what your API key can access.

---

## Anthropic

Direct access to Anthropic models.

### Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxx"
agentomics-run --provider anthropic --list-models
```

Use `--provider anthropic` explicitly when other provider keys are also set.

---

## Codex (ChatGPT OAuth)

Experimental support for the local Codex CLI login flow.

### Setup

First, sign into Codex on the same machine:

```bash
codex login
```

Then run Agentomics with the `codex` provider:

```bash
agentomics-run --provider codex --list-models
agentomics-run --provider codex --model gpt-5.4 --dataset my_data
```

This provider reads your local Codex auth state from `~/.codex/auth.json` and
uses the ChatGPT Codex backend instead of `OPENAI_API_KEY`.

`agentomics-run` automatically mounts your host `~/.codex` login into the
container (read-only), so no extra setup is needed beyond `codex login`.

If you are also setting `OPENAI_API_KEY` or `OPENROUTER_API_KEY`, pass
`--provider codex` explicitly for non-interactive runs.

---

## Ollama (Local Models)

Run models locally for privacy or offline use.

### Requirements

1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull <model-name>`
3. Set `OLLAMA_BASE_URL` so Agentomics considers Ollama available:
   `export OLLAMA_BASE_URL=http://localhost:11434/v1`

The launcher enables host networking so the container can reach Ollama:

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
agentomics-run --provider ollama --model <ollama-model> --dataset <dataset>
```

The server URL Agentomics connects to is the `Ollama` entry's `base_url` in
`src/agentomics/utils/providers/configured_providers.yaml` (default
`http://localhost:11434/v1`); edit it if your server differs. See
[Ollama Configuration](environment.md#ollama-configuration).

### Popular Models

Run `ollama list` to see available models.

---

## Custom Providers

Add custom providers in `src/agentomics/utils/providers/configured_providers.yaml`:

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
agentomics-run --model my-custom-model
```

---

## Provider Selection

When multiple providers are configured, they're all available. Use `--list-models` to see all options:

```bash
agentomics-run --list-models
```

The interactive mode groups models by provider for easy selection.

---

## Model Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Default | Use `--list-models` to pick |
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
agentomics-run --list-models
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

For Docker mode, run with `--network host` so the container can reach the host's Ollama server, and verify the configured Ollama URL is reachable.
