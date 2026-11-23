#!/usr/bin/env bash
# Entry point of the TGV_StoreNone (tgvstorenone) decomposition sweep. With no command it prints help.
# Nothing is submitted to Slurm unless a command is given --submit (or is `submit`).
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
python_bin="${OPS_LAUNCHER_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "ERROR: $python_bin not found. Set OPS_LAUNCHER_PYTHON to a Python >= 3.6 interpreter." >&2
  exit 2
fi
exec "$python_bin" "$script_dir/launcher.py" "$@"
