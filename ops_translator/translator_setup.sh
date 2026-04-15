#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

make --silent -C "${SCRIPT_DIR}" python
exec "${SCRIPT_DIR}/.python/bin/python3" "${SCRIPT_DIR}/ops-translator" "$@"
