#!/usr/bin/env bash
# Provision the cloud dev environment for fast-browser-use.
set -euxo pipefail

# uv (deterministic Python package manager)
pipx install uv || python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

# Python dependencies (locked)
uv sync --locked --python 3.12

# Headless browser + media deps used by the browser agent
uv run playwright install --with-deps chromium
sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg

# Codex CLI available inside the cloud environment
npm install -g @openai/codex@0.157.1
