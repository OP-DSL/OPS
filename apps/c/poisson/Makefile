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




HEADERS=poisson_kernel.h

OPS_FILES=poisson.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=


APP=poisson
MAIN_SRC=poisson

#OPS_GENERATOR_VERBOSE=1
TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
