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




HEADERS=data.h preproc_kernel.h init_kernel.h

OPS_FILES=compact_scheme.cpp

OPS_GENERATED=compact_scheme_ops.cpp

OTHER_FILES=


APP=Compact3D
MAIN_SRC=Compact3D

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
