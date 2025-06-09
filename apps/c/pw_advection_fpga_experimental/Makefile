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

# PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
PLATFORM=xilinx_vck5000_gen4x8_xdma_2_202210_1
HLS_TARGET_MODE=hw
HLS_JOBS=128
include $(OPS_INSTALL_PATH)/../makefiles/Makefile_hls.mk

HEADERS=pw_advect_kernel.h

OPS_FILES=pw_advect.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=


APP=pw_advect3d
MAIN_SRC=pw_advect

# OPS_GENERATOR_VERBOSE=1
# OPS_GENERATOR_DEBUG=1
ifeq ($(PLATFORM),xilinx_u280_gen3x16_xdma_1_202211_1)
OPS_TARGET_CONFIG_FILE= ./config_u280.json #U280 config file not yet available
else
OPS_TARGET_CONFIG_FILE= ./config_vck5000.json
endif


TARGETS=hls #tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled ompoffload mpi_ompoffload mpi_ompoffload_tiled sycl mpi_sycl mpi_sycl_tiled openacc mpi_openacc mpi_openacc_tiled

########  PRFOFILE OPTION ######

#Enable if wanted to profile
CXXFLAGS += -DPROFILE

########  VERIIFICATION OPTIONS  ######
#Enable if wanted to verify
# CXXFLAGS += -DVERIFICATION

#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app_legacy
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app

