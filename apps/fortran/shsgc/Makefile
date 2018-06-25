include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

OTHER_FILES=constants.F90
OPS_FILES=shsgc.F90
OPS_GENERATED=shsgc_ops.F90

APP = shsgc
MAIN_SRC=shsgc
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app
