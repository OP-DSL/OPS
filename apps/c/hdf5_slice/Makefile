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




HEADERS=init_kernel.h data.h

OPS_FILES=hdf5_slice.cpp

OPS_GENERATED=hdf5_slice_ops.cpp

OTHER_FILES=


APP=hdf5_slice
MAIN_SRC=hdf5_slice

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app