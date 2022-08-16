
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

OTHER_FILES=constants.F90 start.F90 data.F90

OPS_FILES=clover_leaf.F90 opsdata.F90 build_field.F90 initialise_chunk.F90 initialise.F90

OPS_GENERATED=clover_leaf_ops.F90 opsdata_ops.F90 build_field_ops.F90 initialise_chunk_ops.F90 initialise_ops.F90

APP = cloverleaf
MAIN_SRC=clover_leaf
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app
