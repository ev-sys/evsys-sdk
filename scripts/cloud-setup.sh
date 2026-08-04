#!/bin/bash
# Paste this into the *Setup script* field of a Claude Code cloud environment
# (claude.ai/code -> environment selector -> settings icon).
#
# It is NOT read from the repo: Claude Code cloud environments keep the setup
# script in the environment config, not in git. This file exists so the script
# is reviewable and versioned alongside the code it installs.
#
# It runs once per environment, as root on Ubuntu 24.04, before Claude starts,
# and Anthropic snapshots the filesystem afterwards. Later sessions restore the
# snapshot and skip this entirely, so heavy installs belong here rather than in
# the SessionStart hook, which runs every single session.
#
# Constraints worth knowing: it must exit 0 or the session fails to start, and
# it must finish inside ~5 minutes or the cache never builds.
set -uxo pipefail

# uv is pre-installed on the cloud image, but not guaranteed on every base.
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh || true
export PATH="$HOME/.local/bin:$PATH"

cd "${CLAUDE_PROJECT_DIR:-/repo}" 2>/dev/null || cd /
# `skypilot` matters as much as `dev`: without it every import of the
# provisioning stack is skipped rather than tested, which hides breakage until
# it reaches a real launch.
uv sync --extra dev --extra skypilot || true

mkdir -p "$HOME/.evsys" "$HOME/.sky" || true
exit 0
