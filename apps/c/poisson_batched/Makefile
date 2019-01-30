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


BATCHED=0

HEADERS=poisson_kernel.h

OPS_FILES=poisson.cpp

OPS_GENERATED=poisson_ops.cpp

OTHER_FILES=


APP=poisson
MAIN_SRC=poisson

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
