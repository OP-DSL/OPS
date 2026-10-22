#!/usr/bin/env bash
# TGV Top 3 tiling experiment: depth 6, tiles 8000^3, GPU Direct, OPS_DIAGS=2.
# Uses the same portable dataset, build, Slurm and output workflow as lumi_tgv_top3.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/../lumi_tgv_top3/run.sh" --tiled "$@"
