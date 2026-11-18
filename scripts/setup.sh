#!/usr/bin/env bash
# Creates the Python environment and downloads Reson8/Rumble-2-Flash (~1 GB)
# into the Hugging Face cache.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
uv venv --allow-existing --python 3.12 .venv
uv sync
.venv/bin/hf download "${RUMBLE_MODEL:-Reson8/Rumble-2-Flash}"
echo "Setup complete. Next: ./build.sh && open RumbleBar.app"
