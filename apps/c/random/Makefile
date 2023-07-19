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

HEADERS=random_print_kernel.h

OPS_FILES=random.cpp

OPS_GENERATED=random_ops.cpp

OTHER_FILES=


APP=random
MAIN_SRC=random

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
