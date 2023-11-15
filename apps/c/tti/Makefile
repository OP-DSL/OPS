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




HEADERS=tti_kernels.h

OPS_FILES=tti.cpp

OPS_GENERATED=tti_ops.cpp

OTHER_FILES=


APP=tti
MAIN_SRC=tti

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
