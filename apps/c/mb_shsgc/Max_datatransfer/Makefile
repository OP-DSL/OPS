#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hip




HEADERS=conv_kernel.h tvdx_kernel.h gridgen_kernel.h init_kernel.h

OPS_FILES=shsgc.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=


APP=shsgc
MAIN_SRC=shsgc

TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled ompoffload

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
