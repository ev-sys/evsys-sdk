#!/bin/bash
# Paste this into the *Setup script* field of a Claude Code cloud environment
# (claude.ai/code -> environment selector -> settings icon).
#
# Claude Code does NOT read this from the repo: a cloud environment keeps its
# setup script in the environment config. This file exists so the script is
# reviewable and versioned next to the code it installs.
#
# Runs once per environment, as root on Ubuntu 24.04, before Claude starts.
# Anthropic then snapshots the filesystem, so later sessions restore it and skip
# this entirely — which is why heavy installs belong here and not in the
# SessionStart hook, which runs every session.
#
# Two hard constraints: exit 0 or the session fails to start, and finish inside
# ~5 minutes or the cache never builds. Hence `|| true` on everything.
set -uxo pipefail

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh || true
export PATH="$HOME/.local/bin:$PATH"

cd "${CLAUDE_PROJECT_DIR:-/repo}" 2>/dev/null || cd /
uv sync --extra dev || true

# skypilot is installed SEPARATELY and deliberately. It cannot go in the
# `--extra skypilot` sync: skypilot 0.13 pins uvicorn<0.36 while harbor needs
# >=0.38, so `uv sync --extra skypilot` is unresolvable. Installing it after the
# sync downgrades uvicorn and works, which is how a working dev venv is built.
# Without it, every import of the provisioning stack fails rather than being
# tested.
uv pip install 'skypilot>=0.13' || true

mkdir -p "$HOME/.evsys" "$HOME/.sky" || true

# --- credentials -------------------------------------------------------------
# The SDK reads vendor credentials from files, not variables, so materialise
# them from the environment variables set on this cloud environment.
#
# Be clear-eyed about what this is: a cloud environment has no secrets store,
# and its variables are readable by anyone who can use the environment. These
# keys rent GPUs that bill by the hour. Scope them as narrowly as the vendor
# allows and rotate them when the work is done.
if [ -n "${VERDA_CLIENT_ID:-}" ] && [ -n "${VERDA_CLIENT_SECRET:-}" ]; then
  mkdir -p "$HOME/.verda"
  printf '{"client_id":"%s","client_secret":"%s"}\n' \
    "$VERDA_CLIENT_ID" "$VERDA_CLIENT_SECRET" > "$HOME/.verda/config.json"
  chmod 600 "$HOME/.verda/config.json"
  echo "[cloud-setup] wrote ~/.verda/config.json"
fi

if [ -n "${VAST_API_KEY:-}" ]; then
  mkdir -p "$HOME/.vast"
  printf '{"api_key":"%s"}\n' "$VAST_API_KEY" > "$HOME/.vast/config.json"
  chmod 600 "$HOME/.vast/config.json"
  echo "[cloud-setup] wrote ~/.vast/config.json"
fi

if [ -n "${PRIME_API_KEY:-}" ]; then
  mkdir -p "$HOME/.prime"
  printf '{"api_key":"%s"}\n' "$PRIME_API_KEY" > "$HOME/.prime/config.json"
  chmod 600 "$HOME/.prime/config.json"
  echo "[cloud-setup] wrote ~/.prime/config.json"
fi

exit 0
