# Proxy Settings

Configure Agentomics-ML to work behind corporate proxies and firewalls.

## Environment Variables

Set these variables for proxy access:

| Variable | Description | Example |
|----------|-------------|---------|
| `HTTP_PROXY` | HTTP proxy URL | `http://proxy.company.com:8080` |
| `HTTPS_PROXY` | HTTPS proxy URL | `http://proxy.company.com:8080` |
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

Docker requires separate proxy configuration.

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

Pass proxy settings as build arguments:

```bash
docker build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -t agentomics .
```

## Running with Proxy

In Docker mode, pass proxy settings to the container with `-e`:

```bash
docker run --rm \
  -e HTTP_PROXY=$HTTP_PROXY \
  -e HTTPS_PROXY=$HTTPS_PROXY \
  -e NO_PROXY=$NO_PROXY \
  --env-file .env \
  -v "$(pwd)/datasets:/repository/datasets" \
  -v "$(pwd)/outputs/my_run_1:/workspace" \
  biogemt/agentomics:latest --dataset my_dataset
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

## Local Mode

In local mode, proxy settings are inherited from your shell environment automatically.

## Related

- [Environment Variables](../configuration/environment.md)
