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




HEADERS=conv_kernel.h tvdx_kernel.h gridgen_kernel.h init_kernel.h

OPS_FILES=shsgc.cpp

OPS_GENERATED=shsgc_ops.cpp

OTHER_FILES=


APP=shsgc
MAIN_SRC=shsgc

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
