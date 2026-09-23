# Proxy Settings

Configure Agentomics-ML to work behind corporate proxies and firewalls.

## Environment Variables

Set these variables for proxy access:

| Variable | Description | Example |
|----------|-------------|---------|
| `HTTP_PROXY` | HTTP proxy URL | `http://proxy.company.com:8080` |
| `HTTPS_PROXY` | HTTPS proxy URL | `http://proxy.company.com:8080` |
| `ALL_PROXY` | Fallback proxy URL for all protocols | `socks5://proxy.company.com:1080` |
| `NO_PROXY` | Hosts to bypass | `localhost,127.0.0.1,.internal` |

### Setting Variables

**Export in shell:**

```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1
```

**Or in `.env` file:**

```bash
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080
NO_PROXY=localhost,127.0.0.1
```

## Docker Configuration

Pulling images from a registry is handled by the Docker daemon. If registry pulls also require a proxy, configure the daemon separately from the proxy settings used by image build steps.

### Configure Docker Daemon

Create or edit the proxy configuration:

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
```

Add:

```ini
[Service]
Environment="HTTP_PROXY=http://proxy.company.com:8080"
Environment="HTTPS_PROXY=http://proxy.company.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1"
```

### Apply Changes

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Verify Configuration

```bash
sudo systemctl show --property=Environment docker
```

## Building Images with Proxy

When an Agentomics command is run with `--dev`, the launcher automatically forwards `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY` to the Dockerbuild. Uppercase and lowercase forms are supported.

Proxy values can be exported in the shell or placed in the `.env` file in the repository root. Exported values take precedence over `.env` values:

```bash
agentomics-run --dev --dataset my_dataset
```

## Running with Proxy

`agentomics-run` forwards the exported `HTTP_PROXY`, `HTTPS_PROXY`, and
`ALL_PROXY` variables (and their lowercase forms) into the container:

```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
agentomics-run --dataset my_dataset
```

To pass any other proxy variable (for example `NO_PROXY`), add it to your `.env`
file — the launcher forwards the whole `.env` to the container:

```bash
# .env
NO_PROXY=localhost,127.0.0.1,.internal
```

## Checking Proxy Variables

Verify your proxy settings:

```bash
# Check HTTP proxy
env | grep -i http_proxy

# Check HTTPS proxy
env | grep -i https_proxy
```

## Common Proxy Formats

```bash
# Basic
HTTP_PROXY=http://proxy.company.com:8080

# With authentication
HTTP_PROXY=http://username:password@proxy.company.com:8080

# SOCKS proxy
HTTP_PROXY=socks5://proxy.company.com:1080
```

## API-Specific Proxies

Different APIs may need different proxy settings:

```bash
# For LLM APIs
HTTPS_PROXY=http://proxy.company.com:8080

# Bypass for internal services
NO_PROXY=localhost,127.0.0.1,internal.company.com
```

## Troubleshooting

### Connection timeouts

Verify proxy is accessible:

```bash
curl -x http://proxy.company.com:8080 https://api.openai.com
```

### Certificate errors

Some proxies perform SSL inspection. You may need to:

1. Add the corporate CA certificate to the container
2. Or set `REQUESTS_CA_BUNDLE` environment variable

### Docker can't pull images

Ensure Docker daemon proxy is configured (not just shell environment).

### LLM API calls fail

Check that the proxy allows connections to:
- `api.openai.com`
- `api.anthropic.com`
- `openrouter.ai`

Add to `NO_PROXY` if these should bypass the proxy.

## Related

- [Environment Variables](../configuration/environment.md)
