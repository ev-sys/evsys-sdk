#!/usr/bin/env bash
# Build the environment a cloud agent starts from.
#
# Runs once when the snapshot is built, not on every prompt, so it is worth
# installing the heavy extras here rather than making each agent wait.
set -euxo pipefail

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# `dev` gets pytest; `skypilot` gets the provisioning stack the compute layer
# imports lazily. Without the latter, anything touching SkyPilot is skipped
# rather than tested, which hides breakage until it reaches a real launch.
uv sync --extra dev --extra skypilot

# State the SDK writes to. Creating them here means a first run does not race
# on mkdir, and makes it obvious where the ledgers live.
mkdir -p "$HOME/.evsys" "$HOME/.sky"

uv run python -c "import evsys_sdk, evsys_sdk.compute as c; print('evsys-sdk ok:', sorted(c.__all__)[:6])"
uv run pytest -q -x --timeout=300 2>/dev/null || uv run pytest -q -x
