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


HEADERS=bc_exchanges.h constants.h defdec_data_set.h io.h opensbliblock00_kernels.h stencils.h

OPS_FILES=opensbli.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=


APP=wave
MAIN_SRC=wave

#OPS_GENERATOR_VERBOSE=1
TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled sycl mpi_sycl mpi_sycl_tiled ompoffload mpi_ompoffload mpi_ompoffload_tiled

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app_legacy
#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
