#!/usr/bin/env bash

# Common shell utilities for agentomics bash scripts

# Ensure running under bash even if invoked via sh/zsh
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

# Print error message in red and exit with status 1
die() {
    echo -e "${RED}Error: $*${NOCOLOR}" >&2
    exit 1
}

# Check if a command exists, die with helpful message if not
need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# Validate that an option has a non-empty value
require_opt_value() {
    local opt="$1"
    local val="${2:-}"
    [[ -n "$val" && "$val" != --* ]] || die "Missing value for $opt"
}

# Check if stdin and stdout are connected to a TTY (interactive terminal)
has_tty() {
    [[ -t 0 && -t 1 ]]
}

# Print info message in green
info() {
    echo -e "${GREEN}$*${NOCOLOR}"
}

# Print warning message in yellow
warn() {
    echo -e "${YELLOW}Warning: $*${NOCOLOR}" >&2
}

docker_has_gpu() {
    # Check if nvidia-container-toolkit is installed and nvidia-smi works
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        # Also verify docker can use GPUs by checking for nvidia runtime
        if docker info 2>/dev/null | grep -q nvidia; then
            return 0
        fi
    fi
    return 1
}