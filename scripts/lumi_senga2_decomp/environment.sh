#!/usr/bin/env bash
# Default environment: LUMI-G, PrgEnv-gnu (ftn = gfortran), ROCm, GPU-aware Cray MPICH.
# It is copied into the campaign by `run.sh prepare` and sourced before every build and
# job. To change modules or any compiler/flag, copy this file, edit it and pass it to
# prepare with --env-file. `run.sh build --print-toolchain` lists every SENGA_* variable.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
fi
# A ready site environment is preserved; otherwise the standard LUMI components are loaded.
if ! command -v hipcc >/dev/null 2>&1 || [[ -z "${CRAY_MPICH_DIR:-}" ]]; then
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

# ---- Fortran toolchain: ftn is gfortran under PrgEnv-gnu ---------------------------
export SENGA_F90="${SENGA_F90:-ftn}"
export SENGA_MPIF90="${SENGA_MPIF90:-ftn}"
# Flags of makefiles/Makefile.gnu; the OPS Fortran sources need free-form long lines.
export SENGA_FFLAGS="${SENGA_FFLAGS--O3 -g -ffree-form -ffree-line-length-none -fopenmp}"
# Application flags, including its optimisation level (SENGA2 src/Makefile).
export SENGA_APP_FLAGS="${SENGA_APP_FLAGS--g -cpp -O3 -fallow-argument-mismatch}"
export SENGA_LINK_EXTRA="${SENGA_LINK_EXTRA--fopenmp -lstdc++}"
# Host C/C++ (ROCm clang/clang++), hipcc, Cray MPICH and libmpi_gtl_hsa are located
# automatically from the loaded modules; override with SENGA_CC, SENGA_CXX, SENGA_HIPCC,
# SENGA_ROCM_ROOT, SENGA_MPI_ROOT, SENGA_GTL_DIR, SENGA_CXXFLAGS, SENGA_HIPFLAGS,
# SENGA_MPI_HIP_LINK, SENGA_HIP_LINK, SENGA_HIPRAND_LINK when needed.
