#!/usr/bin/env bash
# Documented fallback: Cray Fortran (PrgEnv-cray) for the Fortran sources, following
# SENGA2 src/lumi_compile.sh and makefiles/Makefile.cray. Use it when the PrgEnv-gnu build
# fails to compile or link:   run.sh prepare ... --env-file environment_cray.sh
# (a new campaign: an environment file is frozen at prepare time).
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
fi
if ! command -v module >/dev/null 2>&1; then
  echo 'ERROR: no "module" command. Pass --env-file with your own module setup.' >&2
  return 1
fi
module load LUMI
module load partition/G
module load PrgEnv-cray
module load rocm
module load craype-accel-amd-gfx90a
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load cray-python 2>/dev/null || true
fi
export AMD_ARCH=gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1

export SENGA_F90="${SENGA_F90:-ftn}"
export SENGA_MPIF90="${SENGA_MPIF90:-ftn}"
# Flags of makefiles/Makefile.cray (FFLAGS) and SENGA2 src/Makefile.codegen (-N 1023).
export SENGA_FFLAGS="${SENGA_FFLAGS--g -fopenmp -M 878 -f free}"
# Fortran optimisation level of the application ...
export SENGA_APP_FLAGS="${SENGA_APP_FLAGS--O3 -N 1023}"
# ... and of the 16 large generated files, which src/lumi_compile.sh compiles at -O1.
# Raise it to -O2/-O3 if the compiler copes; unset SENGA_APP_FLAGS_LARGE to use
# SENGA_APP_FLAGS for every file.
export SENGA_APP_FLAGS_LARGE="${SENGA_APP_FLAGS_LARGE--O1 -N 1023}"
export SENGA_LINK_EXTRA="${SENGA_LINK_EXTRA--fopenmp -lstdc++}"
# Host C/C++ stay ROCm clang/clang++ and the HIP compiler stays hipcc (located from the
# rocm module). To try the Cray C++ driver instead, as Makefile.cray does, set for example:
#   export SENGA_CC=cc SENGA_CXX=CC SENGA_HIPCC=CC
#   export SENGA_HIPFLAGS="-x hip -std=c++17 -D__HIP_ROCclr__ --rocm-path=$ROCM_PATH --offload-arch=gfx90a"
