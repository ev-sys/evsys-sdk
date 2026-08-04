#!/bin/bash
# Claude Code on the web / Slack bootstrap for the trajectory-labs SDK (evsys-sdk).
# Runs at SessionStart in cloud sessions only and installs the SDK + dev deps so
# the repo is ready to run when opened directly (e.g. from Slack).
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

if [ "${CLAUDE_CODE_REMOTE:-false}" != "true" ]; then
  echo "[claude-session-start] local session detected, skipping remote bootstrap"
  exit 0
fi

echo "[claude-session-start] bootstrapping trajectory-labs SDK at $PROJECT_DIR"

if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev || uv sync || echo "[claude-session-start] WARNING: uv sync failed" >&2
  # skypilot cannot be a sync extra: it pins uvicorn<0.36 while harbor needs
  # >=0.38, so `uv sync --extra skypilot` is unresolvable. Installing it after
  # the sync downgrades uvicorn and works. Only bother if it is missing, since
  # this hook runs on every session and the setup script usually did it already.
  python -c "import sky" 2>/dev/null \
    || uv pip install "skypilot>=0.13" >/dev/null 2>&1 \
    || echo "[claude-session-start] note: skypilot unavailable; provisioning paths will not import" >&2
  if [ -n "${CLAUDE_ENV_FILE:-}" ] && [ -d "$PROJECT_DIR/.venv/bin" ]; then
    printf 'export PATH="%s/.venv/bin:${PATH}"\n' "$PROJECT_DIR" >> "$CLAUDE_ENV_FILE"
  fi
else
  echo "[claude-session-start] WARNING: uv not found; install it in the cloud setup script" >&2
fi

# Preflight the GPU vendors. None of these hosts is on the cloud environment's
# default Trusted allowlist, so a session that has not been given Custom network
# access can run the whole test suite (it is hermetic) but cannot price, check
# availability, or launch anything. Saying so here turns a confusing timeout
# later into one clear line now.
for host in api.verda.com console.vast.ai api.primeintellect.ai; do
  if curl -sS -o /dev/null -m 6 "https://$host" 2>/dev/null; then
    echo "[claude-session-start] vendor reachable: $host"
  else
    echo "[claude-session-start] vendor UNREACHABLE: $host — add it to the" \
         "environment's Custom allowed domains if this session needs live capacity" >&2
  fi
done

echo "[claude-session-start] trajectory-labs SDK ready"
