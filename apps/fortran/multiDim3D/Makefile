include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

OTHER_FILES=constants.F90

OPS_FILES=multidim.F90

OPS_GENERATED_SUFFIX=_ops

APP = multidim
MAIN_SRC=multidim

TARGETS=seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled

#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app_legacy
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app
