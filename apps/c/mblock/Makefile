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

HEADERS=mblock_populate_kernel.h

OPS_FILES=mblock.cpp

OPS_GENERATED=mblock_ops.cpp

OTHER_FILES=


APP=mblock
MAIN_SRC=mblock

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
