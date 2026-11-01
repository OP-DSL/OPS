#!/usr/bin/env bash
# Validate both profiles before submitting one job for 4 nodes and one for 6.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
engine="$script_dir/compare_yz_n016.sh"
action=dry-run
output_root=
common=()
while (( $# )); do
  case "$1" in
    --submit) action=submit; shift ;;
    --dry-run) action=dry-run; shift ;;
    --output-root)
      [[ $# -ge 2 && -n "$2" ]] || { echo 'ERROR: --output-root needs a path' >&2; exit 2; }
      output_root=$2; shift 2 ;;
    --help|-h)
      cat <<'HELP'
Usage: bash compare_yz_n004_n006.sh [--dry-run | --submit] [options]
4 nodes: 1280^3, 32 ranks, MPI 2x8x2 and 1x8x4.
6 nodes: 1452^3, 48 ranks, MPI 6x4x2.
Each shape: rank-0 autotuning once, then replay with its own blocks, 500 steps.
Options: --account, --campaign, --ops-root, --env-file, --executable,
         --repetitions (replay only), --output-root.
Historical default/Top3 are not rerun. No action defaults to dry-run.
For a single profile or catalog reuse, use compare_yz_n016.sh --nodes 4|6.
HELP
      exit 0 ;;
    --account|--campaign|--ops-root|--env-file|--executable|--repetitions)
      [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }
      common+=("$1" "$2"); shift 2 ;;
    *) echo "ERROR: Unsupported option: $1 (use --help)" >&2; exit 2 ;;
  esac
done
stamp=$(date +%Y%m%d_%H%M%S)_000000
run_profile() {
  local nodes=$1 mode=$2
  local extra=()
  if [[ -n "$output_root" ]]; then
    local tag
    printf -v tag 'N%03d' "$nodes"
    extra=(--output "$output_root/TGV_YZ_${tag}_${stamp}")
  fi
  bash "$engine" --nodes "$nodes" "${common[@]}" "${extra[@]}" "--$mode"
}
# If either preflight fails, no job is submitted.
run_profile 4 dry-run
run_profile 6 dry-run
if [[ "$action" == submit ]]; then
  run_profile 4 submit
  run_profile 6 submit
fi
