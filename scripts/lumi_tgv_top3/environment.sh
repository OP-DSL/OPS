#!/usr/bin/env bash
# Sourced inside build and compute jobs. Override with --env-file for pinned modules.
# Preserve a ready site environment; otherwise load the standard LUMI components.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
fi
if ! command -v hipcc >/dev/null 2>&1 || [[ -z "${CRAY_MPICH_DIR:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo 'ERROR: no HIP/Cray MPI environment. Supply --env-file with your LUMI module setup.' >&2
    return 1
  fi
  module load LUMI
  module load partition/G
  module load PrgEnv-gnu
  module load rocm
  module load craype-accel-amd-gfx90a
fi
export AMD_ARCH=gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
