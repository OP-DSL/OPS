#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hip
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5


HEADERS=calupwindeff_kernel.h drhouupdx_kernel.h limiter_kernel.h test_kernel.h updateRK3_kernel.h zerores_kernel.h drhoEpudx_kernel.h fact_kernel.h Riemann_kernel.h tvd_kernel.h vars.h drhoudx_kernel.h initialize_kernel.h save_kernel.h update_kernel.h vars_kernel.h

OPS_FILES=shsgc.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=


APP=shsgc
MAIN_SRC=shsgc

TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled ompoffload mpi_ompoffload mpi_ompoffload_tiled sycl mpi_sycl mpi_sycl_tiled openacc mpi_openacc mpi_openacc_tiled

#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app_legacy
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
