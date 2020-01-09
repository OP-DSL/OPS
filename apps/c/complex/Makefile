#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda



HEADERS=complex_populate_kernel.h

OPS_FILES=complex.cpp

OPS_GENERATED=complex_ops.cpp

OTHER_FILES=


APP=complex
MAIN_SRC=complex

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
