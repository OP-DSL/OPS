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


HEADERS=initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h

OPS_FILES=clover_leaf.cpp initialise_chunk.cpp generate.cpp ideal_gas.cpp \
                update_halo.cpp field_summary.cpp viscosity.cpp \
                calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp flux_calc.cpp \
                advec_cell.cpp advec_mom.cpp reset_field.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES=start.cpp time_step.cpp build_field.cpp read_input.cpp initialise.cpp advection.cpp

APP=cloverleaf
MAIN_SRC=clover_leaf

#OPS_GENERATOR_VERBOSE=1
TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled ompoffload mpi_ompoffload mpi_ompoffload_tiled sycl mpi_sycl mpi_sycl_tiled

#include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app_legacy
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
