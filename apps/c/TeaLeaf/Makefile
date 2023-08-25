#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hip


HEADERS = data.h generate_chunk_kernel.h tea_leaf_cg_kernels.h tea_leaf.h tea_leaf_ops_vars.h user_types.h \
		definitions.h initialise_chunk_kernel.h tea_leaf_cheby_kernels.h tea_leaf_jacobi_kernels.h tea_leaf_ppcg_kernels.h \
		field_summary_kernel.h set_field_kernels.h tea_leaf_common_kernels.h tea_leaf_kernels.h update_halo_kernels.h

OPS_FILES = tea_leaf.cpp field_summary.cpp  generate.cpp initialise_chunk.cpp start.cpp set_field.cpp  tea_leaf_cg.cpp tea_leaf_cheby.cpp tea_leaf_common.cpp tea_leaf_jacobi.cpp tea_leaf_ppcg.cpp update_halo.cpp

OPS_GENERATED_SUFFIX=_ops

OTHER_FILES = build_field.cpp diffuse.cpp initialise.cpp read_input.cpp tea_solve.cpp timestep.cpp

APP=tealeaf
MAIN_SRC=tea_leaf

TARGETS=dev_seq dev_mpi seq tiled openmp mpi mpi_tiled mpi_openmp cuda mpi_cuda mpi_cuda_tiled hip mpi_hip mpi_hip_tiled ompoffload mpi_ompoffload mpi_ompoffload_tiled sycl mpi_sycl mpi_sycl_tiled openacc mpi_openacc mpi_openacc_tiled

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
