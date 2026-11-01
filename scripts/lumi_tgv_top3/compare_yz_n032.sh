#!/usr/bin/env bash
# 32-node follow-up only; does not resubmit the 4/6-node experiments.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for arg in "$@"; do
  case "$arg" in
    --nodes|--nodes=*) echo 'ERROR: this entrypoint is fixed to 32 nodes' >&2; exit 2 ;;
  esac
done
exec bash "$script_dir/compare_yz_n016.sh" --nodes 32 "$@"
