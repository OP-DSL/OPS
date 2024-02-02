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


HEADERS=

OPS_FILES=read.cpp

OPS_GENERATED=read_ops.cpp

OTHER_FILES=

APP=read
MAIN_SRC=read

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app

