include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

OTHER_FILES=constants.F90

OPS_FILES=lowdim.F90

OPS_GENERATED_SUFFIX=_ops

APP = lowdim
MAIN_SRC=lowdim

TARGETS=seq tiled openmp mpi mpi_tiled mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app_legacy
#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app
