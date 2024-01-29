include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

OTHER_FILES=constants.F90

OPS_FILES=mblock.F90

OPS_GENERATED_SUFFIX=_ops

APP = mblock
MAIN_SRC=mblock

TARGETS=seq tiled openmp mpi mpi_tiled mpi_openmp

#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app_legacy
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app
