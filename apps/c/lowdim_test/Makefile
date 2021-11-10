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




HEADERS=lowdim_kernels.h

OPS_FILES=lowdim.cpp

OPS_GENERATED=lowdim_ops.cpp

OTHER_FILES=


APP=lowdim
MAIN_SRC=lowdim

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
