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


HEADERS=multidim_print_kernel.h multidim_reduce_kernel.h multidim_kernel.h

OPS_FILES=multidim.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=


APP=multidim
MAIN_SRC=multidim

TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled ompoffload sycl mpi_sycl mpi_sycl_tiled

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
