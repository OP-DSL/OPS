#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hip


HEADERS=mblock_populate_kernel.h

OPS_FILES=mblock.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=


APP=mblock
MAIN_SRC=mblock

TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled ompoffload sycl mpi_sycl mpi_sycl_tiled openacc mpi_openacc mpi_openacc_tiled

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
