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
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5




HEADERS=user_kernels.h

OPS_FILES=tilingcheck.cpp

OPS_GENERATED=tilingcheck_ops.cpp

OTHER_FILES=


APP=tilingcheck
MAIN_SRC=tilingcheck

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
