#!/usr/bin/env bash
# Portable entrypoint. With no action, print help; --submit is always explicit.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec "${OPS_LAUNCHER_PYTHON:-python3}" "$script_dir/launcher.py" "$@"
