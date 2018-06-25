include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

OTHER_FILES=constants.F90
OPS_FILES=poisson.F90
OPS_GENERATED=poisson_ops.F90

APP = poisson
MAIN_SRC=poisson
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app
