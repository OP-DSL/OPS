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




HEADERS=laplace_kernels.h

OPS_FILES=laplace2d.cpp

OPS_GENERATED=laplace2d_ops.cpp

OTHER_FILES=


APP=laplace2d
MAIN_SRC=laplace2d

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
