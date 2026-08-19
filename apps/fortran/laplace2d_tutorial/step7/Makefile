
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hip
#USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

OTHER_FILES=constants.F90

OPS_FILES=laplace2d.F90

OPS_GENERATED_SUFFIX=_ops

APP = laplace2d
MAIN_SRC=laplace2d

TARGETS=seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled ompoffload mpi_ompoffload mpi_ompoffload_tiled f2c_seq f2c_mpi f2c_mpi_tiled f2c_cuda f2c_mpi_cuda f2c_mpi_cuda_tiled f2c_hip f2c_mpi_hip f2c_mpi_hip_tiled f2c_sycl f2c_mpi_sycl f2c_mpi_sycl_tiled

#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app_legacy
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.f_app
