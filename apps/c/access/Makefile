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




HEADERS=access_populate_kernel.h

OPS_FILES=access.cpp

OPS_GENERATED=access_ops.cpp

OTHER_FILES = fetch_data.cpp


APP=access
MAIN_SRC=access

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
