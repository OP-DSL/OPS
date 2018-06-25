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




HEADERS=complex_numbers_block_0_kernel.h

OPS_FILES=complex_numbers.cpp

OPS_GENERATED=complex_numbers_ops.cpp

OTHER_FILES=


APP=complex_numbers
MAIN_SRC=complex_numbers

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
