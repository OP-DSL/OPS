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




HEADERS=multidim_print_kernel.h multidim_reduce_kernel.h multidim_kernel.h

OPS_FILES=multidim.cpp

OPS_GENERATED=multidim_ops.cpp

OTHER_FILES=


APP=multidim
MAIN_SRC=multidim

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
