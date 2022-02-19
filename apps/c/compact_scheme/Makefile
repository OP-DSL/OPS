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
USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

# This refers to the location of the submodule in the "master" pdetk repo
TRID_INSTALL_PATH = $(TDMA_INSTALL_PATH)
#../../../tridsolver/scalar

TRID_INC := -I$(TRID_INSTALL_PATH)/include
TRID_LIB := -L$(TRID_INSTALL_PATH)/lib



HEADERS=data.h preproc_kernel.h init_kernel.h

OPS_FILES=compact_scheme.cpp

OPS_GENERATED=compact_scheme_ops.cpp

OTHER_FILES=


APP=compact3d
MAIN_SRC=compact_scheme

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
