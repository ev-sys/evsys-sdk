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
  uv sync || echo "[claude-session-start] WARNING: uv sync failed" >&2
  if [ -n "${CLAUDE_ENV_FILE:-}" ] && [ -d "$PROJECT_DIR/.venv/bin" ]; then
    printf 'export PATH="%s/.venv/bin:${PATH}"\n' "$PROJECT_DIR" >> "$CLAUDE_ENV_FILE"
  fi
else
  echo "[claude-session-start] WARNING: uv not found; install it in the cloud setup script" >&2
fi

echo "[claude-session-start] trajectory-labs SDK ready"
