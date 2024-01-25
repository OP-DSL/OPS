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
USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5




HEADERS=laplace_kernels.h

OPS_FILES=laplace2d.cpp

OPS_GENERATED_SUFFIX=_ops
OPS_GENERATED=laplace2d_ops.cpp

OTHER_FILES=


APP=laplace2d
MAIN_SRC=laplace2d

#OPS_GENERATOR_VERBOSE=1
TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled mpi_inline 
#ompoffload

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app

