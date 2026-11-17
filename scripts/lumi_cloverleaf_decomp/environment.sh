#!/usr/bin/env bash
# Default environment: LUMI-G, PrgEnv-gnu, ROCm, GPU-aware Cray MPICH, for the OPS C/C++ HIP
# build of CloverLeaf_3D. It is copied into the campaign by `run.sh prepare` and sourced before
# every build and job. To change modules or any compiler/flag, copy this file, edit it and pass
# it to prepare with --env-file. `run.sh build --print-toolchain` lists every CLOVER_* variable.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
fi
# A ready PrgEnv-gnu site environment is preserved; otherwise the standard LUMI components are
# loaded. A login shell that still has PrgEnv-cray (the LUMI default) is switched to PrgEnv-gnu,
# so that CRAY_MPICH_DIR and the GTL library are the GNU builds of Cray MPICH.
if [[ "${PE_ENV:-}" != "GNU" ]] || ! command -v hipcc >/dev/null 2>&1 || [[ -z "${CRAY_MPICH_DIR:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo 'ERROR: no HIP/Cray MPI environment and no "module" command. Pass --env-file with your module setup.' >&2
    return 1
  fi
  module load LUMI
  module load partition/G
  module load PrgEnv-gnu
  module load rocm
  module load craype-accel-amd-gfx90a
fi
# The OPS translator needs Python >= 3.8 (the launcher itself runs on the system 3.6).
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  if command -v module >/dev/null 2>&1; then module load cray-python 2>/dev/null || true; fi
fi
export AMD_ARCH=gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1

# ---- C/C++ toolchain ---------------------------------------------------------------
# Host C/C++ (ROCm clang/clang++), hipcc, Cray MPICH and libmpi_gtl_hsa are located
# automatically from the loaded modules. Defaults, overridable here or in the shell:
export CLOVER_CXXFLAGS="${CLOVER_CXXFLAGS--O3 -fPIC -std=c++11}"
export CLOVER_HIPFLAGS="${CLOVER_HIPFLAGS--O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__}"
# Other variables: CLOVER_CC, CLOVER_CXX, CLOVER_MPICXX, CLOVER_HIPCC, CLOVER_ROCM_ROOT, CLOVER_MPI_ROOT,
# CLOVER_GTL_DIR, CLOVER_MPI_HIP_LINK, CLOVER_HIP_LINK, CLOVER_HIPRAND_LINK, CLOVER_LINK_EXTRA.
